use std::fs;
use std::io::{self, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use clap::Parser;
use curve25519_dalek::constants::RISTRETTO_BASEPOINT_TABLE;
use curve25519_dalek::scalar::Scalar;
use rand::RngCore;
use serde::Serialize;

const BASE58_ALPHABET: &str = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

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

/// Blocknet vanity address generator.
///
/// Searches for wallet addresses matching a given prefix and/or suffix.
/// Matching is case-insensitive. Found wallets are saved as JSON files
/// named after their address. Runs indefinitely until stopped with Ctrl+C.
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
}

#[derive(Serialize)]
struct VanityWallet {
    address: String,
    spend_private_key: String,
    spend_public_key: String,
    view_private_key: String,
    view_public_key: String,
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

    let counter = Arc::new(AtomicU64::new(0));
    let found = Arc::new(AtomicU64::new(0));
    let start = Instant::now();

    // Case-insensitive matching: each position has ~35 unique case-folded chars
    // (9 digits + 26 letters), so difficulty is ~35^n instead of 58^n
    let pattern_len = args.prefix.len() + args.suffix.len();
    let expected = 35.0_f64.powi(pattern_len as i32);

    // Lowercase versions for case-insensitive comparison
    let prefix_lower = args.prefix.to_ascii_lowercase();
    let suffix_lower = args.suffix.to_ascii_lowercase();

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
    eprintln!("  output:     {}/", output_dir);
    eprintln!();

    // Spawn worker threads
    for _ in 0..num_threads {
        let prefix_lower = prefix_lower.clone();
        let suffix_lower = suffix_lower.clone();
        let output_dir = output_dir.clone();
        let counter = counter.clone();
        let found = found.clone();

        thread::spawn(move || {
            let mut rng = rand::thread_rng();

            // Fix view keypair once per thread.
            // The address is base58(spend_pub || view_pub). Since base58 encodes
            // the full 64 bytes as one big number, changing spend_pub alone changes
            // every character of the output. So we only need ONE scalar multiplication
            // per candidate instead of two.
            let view_priv = random_scalar(&mut rng);
            let view_pub = (&view_priv * RISTRETTO_BASEPOINT_TABLE).compress();
            let view_pub_bytes = view_pub.to_bytes();

            // Random starting point, then increment (avoids RNG overhead in hot loop)
            let mut spend_priv = random_scalar(&mut rng);
            let mut local_count: u64 = 0;
            let mut combined = [0u8; 64];
            combined[32..].copy_from_slice(&view_pub_bytes);

            loop {
                let spend_pub = (&spend_priv * RISTRETTO_BASEPOINT_TABLE).compress();
                combined[..32].copy_from_slice(spend_pub.as_bytes());

                let address = bs58::encode(&combined).into_string();
                let addr_lower = address.to_ascii_lowercase();

                let prefix_ok = prefix_lower.is_empty() || addr_lower.starts_with(&prefix_lower);
                let suffix_ok = suffix_lower.is_empty() || addr_lower.ends_with(&suffix_lower);

                if prefix_ok && suffix_ok {
                    let wallet = VanityWallet {
                        address: address.clone(),
                        spend_private_key: hex::encode(spend_priv.as_bytes()),
                        spend_public_key: hex::encode(spend_pub.as_bytes()),
                        view_private_key: hex::encode(view_priv.as_bytes()),
                        view_public_key: hex::encode(&view_pub_bytes),
                    };

                    let json = serde_json::to_string_pretty(&wallet).unwrap();
                    let path = format!("{}/{}.json", output_dir, address);

                    match fs::write(&path, &json) {
                        Ok(_) => {
                            let n = found.fetch_add(1, Ordering::Relaxed) + 1;
                            eprintln!("\r\x1b[K  found #{}: {}", n, address);
                            eprintln!("  saved: {}\n", path);
                        }
                        Err(e) => {
                            eprintln!("\r\x1b[Kerror writing {}: {}\n", path, e);
                        }
                    }
                }

                spend_priv += Scalar::ONE;
                local_count += 1;
                if local_count & 0x3FF == 0 {
                    counter.fetch_add(1024, Ordering::Relaxed);
                }
            }
        });
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
