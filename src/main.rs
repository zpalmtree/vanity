use std::fs;
use std::io::{self, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(all(feature = "cuda", target_os = "linux"))]
use std::ffi::CString;

use clap::Parser;
use curve25519_dalek::constants::{RISTRETTO_BASEPOINT_POINT, RISTRETTO_BASEPOINT_TABLE};
use curve25519_dalek::scalar::Scalar;
use rand::RngCore;
use serde::Serialize;

const BASE58_ALPHABET: &str = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

#[derive(Parser)]
#[command(name = "blocknet-vanity")]
struct Args {
    /// Prefix the address must start with (case-insensitive)
    #[arg(long, default_value = "")]
    prefix: String,

    /// Suffix the address must end with (case-insensitive)
    #[arg(long, default_value = "")]
    suffix: String,

    /// Number of worker threads [default: CPU core count]
    #[arg(long, short = 't')]
    threads: Option<usize>,

    /// Use CUDA backend (hybrid mode; currently skeleton)
    #[arg(long)]
    cuda: bool,
}

#[derive(Serialize, Clone)]
struct VanityWallet {
    address: String,
    spend_private_key: String,
    spend_public_key: String,
    view_private_key: String,
    view_public_key: String,
}

#[derive(Clone)]
struct SearchConfig {
    prefix_lower: String,
    suffix_lower: String,
    threads: usize,
}

trait SearchBackend: Send {
    fn name(&self) -> &'static str;
    fn start(
        &mut self,
        config: Arc<SearchConfig>,
        match_tx: mpsc::Sender<VanityWallet>,
        counter: Arc<AtomicU64>,
    ) -> Result<(), String>;
}

struct CpuBackend;

impl SearchBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn start(
        &mut self,
        config: Arc<SearchConfig>,
        match_tx: mpsc::Sender<VanityWallet>,
        counter: Arc<AtomicU64>,
    ) -> Result<(), String> {
        for _ in 0..config.threads {
            let config = config.clone();
            let match_tx = match_tx.clone();
            let counter = counter.clone();

            thread::spawn(move || {
                let mut rng = rand::thread_rng();

                // Fix view keypair once per thread.
                let view_priv = random_scalar(&mut rng);
                let view_pub = (&view_priv * RISTRETTO_BASEPOINT_TABLE).compress();
                let view_pub_bytes = view_pub.to_bytes();

                // Random starting point, then increment (avoids RNG overhead in hot loop)
                let mut spend_priv = random_scalar(&mut rng);
                let mut spend_pub_point = &spend_priv * RISTRETTO_BASEPOINT_TABLE;
                let mut local_count: u64 = 0;
                let mut combined = [0u8; 64];
                let mut address_bytes = [0u8; 96];
                let prefix_bytes = config.prefix_lower.as_bytes();
                let suffix_bytes = config.suffix_lower.as_bytes();
                combined[32..].copy_from_slice(&view_pub_bytes);

                loop {
                    let spend_pub = spend_pub_point.compress();
                    combined[..32].copy_from_slice(spend_pub.as_bytes());

                    let encoded_len = bs58::encode(&combined)
                        .onto(&mut address_bytes[..])
                        .expect("encoding to byte buffer cannot fail");

                    let address_view = &address_bytes[..encoded_len];
                    let prefix_ok = starts_with_ascii_lowered(address_view, prefix_bytes);
                    let suffix_ok = ends_with_ascii_lowered(address_view, suffix_bytes);

                    if prefix_ok && suffix_ok {
                        let address = std::str::from_utf8(address_view)
                            .expect("base58 output must be valid ASCII")
                            .to_owned();

                        let wallet = VanityWallet {
                            address,
                            spend_private_key: hex::encode(spend_priv.as_bytes()),
                            spend_public_key: hex::encode(spend_pub.as_bytes()),
                            view_private_key: hex::encode(view_priv.as_bytes()),
                            view_public_key: hex::encode(view_pub_bytes),
                        };

                        if match_tx.send(wallet).is_err() {
                            return;
                        }
                    }

                    spend_priv += Scalar::ONE;
                    spend_pub_point += RISTRETTO_BASEPOINT_POINT;
                    local_count += 1;
                    if local_count & 0x3FF == 0 {
                        counter.fetch_add(1024, Ordering::Relaxed);
                    }
                }
            });
        }

        Ok(())
    }
}

#[allow(dead_code)]
struct CudaBackend {
    batch_size: usize,
}

impl CudaBackend {
    fn new(batch_size: usize) -> Self {
        Self { batch_size }
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
unsafe extern "C" {
    fn cuda_match_batch(
        inputs_64: *const u8,
        batch_size: i32,
        prefix_lower: *const i8,
        prefix_len: i32,
        suffix_lower: *const i8,
        suffix_len: i32,
        out_flags: *mut u8,
    ) -> i32;
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn cuda_match_batch_host(
    inputs: &[u8],
    batch_size: usize,
    prefix: &str,
    suffix: &str,
    out_flags: &mut [u8],
) -> Result<(), String> {
    let prefix_c =
        CString::new(prefix).map_err(|_| "prefix contains unsupported null byte".to_string())?;
    let suffix_c =
        CString::new(suffix).map_err(|_| "suffix contains unsupported null byte".to_string())?;

    let rc = unsafe {
        cuda_match_batch(
            inputs.as_ptr(),
            batch_size as i32,
            prefix_c.as_ptr(),
            prefix.len() as i32,
            suffix_c.as_ptr(),
            suffix.len() as i32,
            out_flags.as_mut_ptr(),
        )
    };

    if rc == 0 {
        Ok(())
    } else {
        Err(format!("cuda_match_batch failed with code {}", rc))
    }
}

impl SearchBackend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda-hybrid"
    }

    fn start(
        &mut self,
        config: Arc<SearchConfig>,
        match_tx: mpsc::Sender<VanityWallet>,
        counter: Arc<AtomicU64>,
    ) -> Result<(), String> {
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        {
            let _ = (config, match_tx, counter);
            return Err(
                "CUDA backend requires Linux build with --features cuda and nvcc toolchain"
                    .to_string(),
            );
        }

        #[cfg(all(feature = "cuda", target_os = "linux"))]
        {
            let batch_size = self.batch_size;
            let worker_count = config.threads.max(1);

            for _ in 0..worker_count {
                let config = config.clone();
                let match_tx = match_tx.clone();
                let counter = counter.clone();

                thread::spawn(move || {
                    let mut rng = rand::thread_rng();
                    let view_priv = random_scalar(&mut rng);
                    let view_pub = (&view_priv * RISTRETTO_BASEPOINT_TABLE).compress();
                    let view_pub_bytes = view_pub.to_bytes();

                    let mut spend_priv = random_scalar(&mut rng);
                    let mut spend_pub_point = &spend_priv * RISTRETTO_BASEPOINT_TABLE;

                    let mut inputs = vec![0u8; batch_size * 64];
                    let mut flags = vec![0u8; batch_size];
                    let mut spend_priv_batch = vec![[0u8; 32]; batch_size];
                    let mut spend_pub_batch = vec![[0u8; 32]; batch_size];

                    loop {
                        for i in 0..batch_size {
                            let spend_pub = spend_pub_point.compress();
                            spend_priv_batch[i] = *spend_priv.as_bytes();
                            spend_pub_batch[i] = *spend_pub.as_bytes();

                            let offset = i * 64;
                            inputs[offset..offset + 32].copy_from_slice(spend_pub.as_bytes());
                            inputs[offset + 32..offset + 64].copy_from_slice(&view_pub_bytes);

                            spend_priv += Scalar::ONE;
                            spend_pub_point += RISTRETTO_BASEPOINT_POINT;
                        }

                        if let Err(e) = cuda_match_batch_host(
                            &inputs,
                            batch_size,
                            &config.prefix_lower,
                            &config.suffix_lower,
                            &mut flags,
                        ) {
                            eprintln!("cuda worker error: {}", e);
                            return;
                        }

                        for i in 0..batch_size {
                            if flags[i] == 0 {
                                continue;
                            }

                            let offset = i * 64;
                            let address = bs58::encode(&inputs[offset..offset + 64]).into_string();
                            let wallet = VanityWallet {
                                address,
                                spend_private_key: hex::encode(spend_priv_batch[i]),
                                spend_public_key: hex::encode(spend_pub_batch[i]),
                                view_private_key: hex::encode(view_priv.as_bytes()),
                                view_public_key: hex::encode(view_pub_bytes),
                            };

                            if match_tx.send(wallet).is_err() {
                                return;
                            }
                        }

                        counter.fetch_add(batch_size as u64, Ordering::Relaxed);
                    }
                });
            }

            Ok(())
        }
    }
}

/// Generate a random Ristretto scalar (equivalent to Scalar::random)
fn random_scalar(rng: &mut impl RngCore) -> Scalar {
    let mut bytes = [0u8; 64];
    rng.fill_bytes(&mut bytes);
    Scalar::from_bytes_mod_order_wide(&bytes)
}

/// Check if every character in the pattern could appear in a base58 string
/// (case-insensitive: 'l' is valid because 'L' exists in base58, etc.)
fn validate_pattern(s: &str) -> Result<(), char> {
    let lower = BASE58_ALPHABET.to_ascii_lowercase();
    for c in s.chars() {
        if !lower.contains(c.to_ascii_lowercase()) {
            return Err(c);
        }
    }
    Ok(())
}

#[inline]
fn starts_with_ascii_lowered(haystack: &[u8], needle_lower: &[u8]) -> bool {
    if needle_lower.is_empty() {
        return true;
    }
    if needle_lower.len() > haystack.len() {
        return false;
    }
    for (i, &b) in needle_lower.iter().enumerate() {
        if haystack[i].to_ascii_lowercase() != b {
            return false;
        }
    }
    true
}

#[inline]
fn ends_with_ascii_lowered(haystack: &[u8], needle_lower: &[u8]) -> bool {
    if needle_lower.is_empty() {
        return true;
    }
    if needle_lower.len() > haystack.len() {
        return false;
    }
    let start = haystack.len() - needle_lower.len();
    for (i, &b) in needle_lower.iter().enumerate() {
        if haystack[start + i].to_ascii_lowercase() != b {
            return false;
        }
    }
    true
}

fn format_count(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i) % 3 == 0 {
            result.push(',');
        }
        result.push(b as char);
    }
    result
}

fn format_duration(secs: u64) -> String {
    if secs >= 86400 {
        format!(
            "{}d {:02}h {:02}m {:02}s",
            secs / 86400,
            (secs % 86400) / 3600,
            (secs % 3600) / 60,
            secs % 60
        )
    } else if secs >= 3600 {
        format!(
            "{}h {:02}m {:02}s",
            secs / 3600,
            (secs % 3600) / 60,
            secs % 60
        )
    } else if secs >= 60 {
        format!("{}m {:02}s", secs / 60, secs % 60)
    } else {
        format!("{}s", secs)
    }
}

fn spawn_writer_thread(
    output_dir: String,
    rx: mpsc::Receiver<VanityWallet>,
    found: Arc<AtomicU64>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        for wallet in rx {
            let address = wallet.address.clone();
            let json = serde_json::to_string_pretty(&wallet).unwrap();
            let path = format!("{}/{}.json", output_dir, address);

            match fs::write(&path, &json) {
                Ok(_) => {
                    let n = found.fetch_add(1, Ordering::Relaxed) + 1;
                    eprintln!("\r\x1b[K  found #{}: {}", n, wallet.address);
                    eprintln!("  saved: {}\n", path);
                }
                Err(e) => {
                    eprintln!("\r\x1b[Kerror writing {}: {}\n", path, e);
                }
            }
        }
    })
}

fn main() {
    let args = Args::parse();
    const MAX_PATTERN_LEN: usize = 8;

    if args.prefix.is_empty() && args.suffix.is_empty() {
        eprintln!("error: specify at least one of --prefix or --suffix");
        std::process::exit(1);
    }

    if args.prefix.len() > MAX_PATTERN_LEN {
        eprintln!("error: prefix cannot exceed {} characters", MAX_PATTERN_LEN);
        std::process::exit(1);
    }

    if args.suffix.len() > MAX_PATTERN_LEN {
        eprintln!("error: suffix cannot exceed {} characters", MAX_PATTERN_LEN);
        std::process::exit(1);
    }

    if !args.prefix.is_empty() {
        if let Err(c) = validate_pattern(&args.prefix) {
            eprintln!(
                "error: prefix contains '{}' which is not a valid base58 character",
                c
            );
            eprintln!("       base58 excludes: 0 (zero), O (uppercase o), I (uppercase i), l (lowercase L)");
            std::process::exit(1);
        }
    }

    if !args.suffix.is_empty() {
        if let Err(c) = validate_pattern(&args.suffix) {
            eprintln!(
                "error: suffix contains '{}' which is not a valid base58 character",
                c
            );
            eprintln!("       base58 excludes: 0 (zero), O (uppercase o), I (uppercase i), l (lowercase L)");
            std::process::exit(1);
        }
    }

    let output_dir = {
        let mut parts = Vec::new();
        if !args.prefix.is_empty() {
            parts.push(format!("{}-prefix", args.prefix.to_lowercase()));
        }
        if !args.suffix.is_empty() {
            parts.push(format!("{}-suffix", args.suffix.to_lowercase()));
        }
        format!("wallets/{}", parts.join("-"))
    };

    fs::create_dir_all(&output_dir).unwrap_or_else(|e| {
        let hint = match e.kind() {
            io::ErrorKind::PermissionDenied => "check directory permissions",
            _ => {
                if std::path::Path::new("wallets").exists()
                    && !std::path::Path::new("wallets").is_dir()
                {
                    "a file named 'wallets' exists and is not a directory"
                } else {
                    "check disk space and filesystem"
                }
            }
        };
        eprintln!("error: failed to create '{}: {} ({})", output_dir, e, hint);
        std::process::exit(1);
    });

    let num_threads = args.threads.unwrap_or_else(|| {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    });

    let config = Arc::new(SearchConfig {
        prefix_lower: args.prefix.to_ascii_lowercase(),
        suffix_lower: args.suffix.to_ascii_lowercase(),
        threads: num_threads,
    });

    let counter = Arc::new(AtomicU64::new(0));
    let found = Arc::new(AtomicU64::new(0));
    let start = Instant::now();
    let (match_tx, match_rx) = mpsc::channel::<VanityWallet>();
    let _writer = spawn_writer_thread(output_dir.clone(), match_rx, found.clone());

    // Case-insensitive matching: each position has ~35 unique case-folded chars
    // (9 digits + 26 letters), so difficulty is ~35^n instead of 58^n
    let pattern_len = config.prefix_lower.len() + config.suffix_lower.len();
    let expected = 35.0_f64.powi(pattern_len as i32);

    let mut backend: Box<dyn SearchBackend> = if args.cuda {
        Box::new(CudaBackend::new(4096))
    } else {
        Box::new(CpuBackend)
    };

    eprintln!("blocknet-vanity");
    eprintln!("───────────────");
    if !args.prefix.is_empty() {
        eprintln!("  prefix:     \"{}\" (case-insensitive)", args.prefix);
    }
    if !args.suffix.is_empty() {
        eprintln!("  suffix:     \"{}\" (case-insensitive)", args.suffix);
    }
    eprintln!(
        "  difficulty: ~{} attempts per match",
        format_count(expected as u64)
    );
    eprintln!("  threads:    {}", num_threads);
    eprintln!("  backend:    {}", backend.name());
    eprintln!("  output:     {}/", output_dir);
    eprintln!();

    if let Err(e) = backend.start(config, match_tx, counter.clone()) {
        eprintln!("error: {}", e);
        std::process::exit(2);
    }

    // Status display on main thread
    let stderr = io::stderr();
    loop {
        thread::sleep(Duration::from_millis(500));
        let count = counter.load(Ordering::Relaxed);
        let elapsed = start.elapsed();
        let secs = elapsed.as_secs_f64();
        let rate = if secs > 0.0 { count as f64 / secs } else { 0.0 };
        let estimate = if rate > 0.0 { (expected / rate) as u64 } else { 0 };
        let found_n = found.load(Ordering::Relaxed);

        let mut handle = stderr.lock();
        let _ = write!(
            handle,
            "\r\x1b[K  {} keys | {:.0}/s | {} found | {} ({} estimated)",
            format_count(count),
            rate,
            found_n,
            format_duration(elapsed.as_secs()),
            format_duration(estimate),
        );
        let _ = handle.flush();
    }
}
