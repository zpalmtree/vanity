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
#[cfg(any(feature = "cuda", test))]
use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::scalar::Scalar;
use rand::RngCore;
use serde::Serialize;

const BASE58_ALPHABET: &str = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
const BASE58_CHARS: &[u8] = BASE58_ALPHABET.as_bytes();

/// Maximum number of generator table entries (supports batch up to 2^TABLE_BITS)
const TABLE_BITS: usize = 24;

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

    /// Use CUDA backend (full GPU key generation)
    #[arg(long)]
    cuda: bool,

    /// GPU batch size (keys per kernel launch) [default: 8388608]
    #[arg(long, default_value = "8388608")]
    batch_size: usize,
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
#[allow(dead_code)]
struct SearchConfig {
    prefix_lower: String,
    suffix_lower: String,
    threads: usize,
    batch_size: usize,
}

// ============================================================================
// U512 big-endian arithmetic helpers (for range precomputation at startup)
// ============================================================================

/// Multiply a 64-byte big-endian number by a small u32 constant, in-place.
/// Returns true if the result overflowed (didn't fit in 64 bytes).
fn u512_mul_small(a: &mut [u8; 64], m: u32) -> bool {
    let mut carry: u64 = 0;
    for i in (0..64).rev() {
        let prod = a[i] as u64 * m as u64 + carry;
        a[i] = prod as u8;
        carry = prod >> 8;
    }
    carry != 0
}

/// Add a single byte value to a 64-byte big-endian number, in-place.
fn u512_add_small(a: &mut [u8; 64], v: u8) {
    let mut carry = v as u16;
    for i in (0..64).rev() {
        let sum = a[i] as u16 + carry;
        a[i] = sum as u8;
        carry = sum >> 8;
        if carry == 0 {
            break;
        }
    }
}

/// Add two 64-byte big-endian numbers, returning the result (ignoring overflow).
fn u512_add(a: &[u8; 64], b: &[u8; 64]) -> [u8; 64] {
    let mut result = [0u8; 64];
    let mut carry: u16 = 0;
    for i in (0..64).rev() {
        let sum = a[i] as u16 + b[i] as u16 + carry;
        result[i] = sum as u8;
        carry = sum >> 8;
    }
    result
}

/// Compare two 64-byte big-endian numbers: returns Ordering.
fn u512_cmp(a: &[u8; 64], b: &[u8; 64]) -> std::cmp::Ordering {
    for i in 0..64 {
        match a[i].cmp(&b[i]) {
            std::cmp::Ordering::Equal => continue,
            other => return other,
        }
    }
    std::cmp::Ordering::Equal
}

// ============================================================================
// Range computation for prefix/suffix matching without base58 encoding
// ============================================================================

/// Map a lowercase ASCII character to its base58 digit value(s).
/// Returns the possible digit values for case-insensitive matching.
/// For digits (1-9): single value. For letters: up to 2 values (upper + lower).
fn base58_char_values(c: u8) -> Vec<u8> {
    let mut values = Vec::new();
    for (i, &ch) in BASE58_CHARS.iter().enumerate() {
        if ch.to_ascii_lowercase() == c {
            values.push(i as u8);
        }
    }
    values
}

/// Compute the numeric ranges [lo, hi) of 64-byte big-endian values whose
/// base58 encoding starts with the given case-insensitive prefix.
///
/// Returns a sorted, merged list of non-overlapping ranges.
fn compute_prefix_ranges(prefix_lower: &str) -> Vec<([u8; 64], [u8; 64])> {
    if prefix_lower.is_empty() {
        return vec![];
    }

    let prefix_bytes = prefix_lower.as_bytes();

    // For each position, get the possible base58 digit values
    let digit_options: Vec<Vec<u8>> = prefix_bytes
        .iter()
        .map(|&c| base58_char_values(c))
        .collect();

    // Generate all case combinations via Cartesian product
    let mut combos: Vec<Vec<u8>> = vec![vec![]];
    for opts in &digit_options {
        let mut new_combos = Vec::new();
        for combo in &combos {
            for &opt in opts {
                let mut new = combo.clone();
                new.push(opt);
                new_combos.push(new);
            }
        }
        combos = new_combos;
    }

    let p = prefix_bytes.len();
    let mut ranges: Vec<([u8; 64], [u8; 64])> = Vec::new();

    // 2^512 as a ceiling (all 0xFF bytes = max value, but actual max is 2^512-1)
    let max_val = [0xFFu8; 64];

    // Base58 encoding of 64 bytes produces 87 or 88 characters
    for addr_len in [87usize, 88] {
        if p > addr_len {
            continue;
        }
        let remaining = addr_len - p;

        // Precompute 58^remaining (shared across combos for this addr_len)
        let mut power = [0u8; 64];
        power[63] = 1;
        let mut power_overflow = false;
        for _ in 0..remaining {
            power_overflow |= u512_mul_small(&mut power, 58);
        }
        if power_overflow {
            continue; // 58^remaining doesn't fit; skip this addr_len
        }

        for combo in &combos {
            // Compute lo = d[0]*58^(L-1) + d[1]*58^(L-2) + ... + d[P-1]*58^(L-P)
            // Using Horner's method: val = ((d[0]*58 + d[1])*58 + d[2])*58 + ...
            // then multiply by 58^(L-P)
            let mut lo = [0u8; 64];
            let mut overflow = false;
            for &d in combo {
                overflow |= u512_mul_small(&mut lo, 58);
                u512_add_small(&mut lo, d);
            }
            // Multiply by 58^remaining
            for _ in 0..remaining {
                overflow |= u512_mul_small(&mut lo, 58);
            }

            // Skip if lo computation overflowed
            if overflow {
                continue;
            }

            // hi = lo + 58^remaining
            let hi = u512_add(&lo, &power);

            // Cap hi at max_val if it overflowed (wrapped to smaller value)
            let clamped_hi = if u512_cmp(&hi, &lo) == std::cmp::Ordering::Less {
                // Addition wrapped around
                max_val
            } else {
                hi
            };

            if u512_cmp(&lo, &clamped_hi) == std::cmp::Ordering::Less {
                ranges.push((lo, clamped_hi));
            }
        }
    }

    // Sort by lo
    ranges.sort_by(|a, b| u512_cmp(&a.0, &b.0));

    // Merge overlapping ranges
    let mut merged: Vec<([u8; 64], [u8; 64])> = Vec::new();
    for (lo, hi) in ranges {
        if let Some(last) = merged.last_mut() {
            // If current lo <= last hi, merge
            if u512_cmp(&lo, &last.1) != std::cmp::Ordering::Greater {
                if u512_cmp(&hi, &last.1) == std::cmp::Ordering::Greater {
                    last.1 = hi;
                }
                continue;
            }
        }
        merged.push((lo, hi));
    }

    merged
}

/// Compute suffix matching targets: returns (modulus, targets) where
/// modulus = 58^suffix_len and targets are the valid remainder values.
fn compute_suffix_targets(suffix_lower: &str) -> (u64, Vec<u64>) {
    if suffix_lower.is_empty() {
        return (0, vec![]);
    }

    let suffix_bytes = suffix_lower.as_bytes();
    let s = suffix_bytes.len();

    // modulus = 58^s
    let mut modulus: u64 = 1;
    for _ in 0..s {
        modulus = modulus
            .checked_mul(58)
            .expect("suffix too long for u64 modulus");
    }

    // Enumerate case combinations
    let digit_options: Vec<Vec<u8>> = suffix_bytes
        .iter()
        .map(|&c| base58_char_values(c))
        .collect();

    let mut combos: Vec<Vec<u8>> = vec![vec![]];
    for opts in &digit_options {
        let mut new_combos = Vec::new();
        for combo in &combos {
            for &opt in opts {
                let mut new = combo.clone();
                new.push(opt);
                new_combos.push(new);
            }
        }
        combos = new_combos;
    }

    let mut targets: Vec<u64> = Vec::new();
    for combo in &combos {
        // target = d[0]*58^(s-1) + d[1]*58^(s-2) + ... + d[s-1]
        let mut val: u64 = 0;
        for &d in combo {
            val = val * 58 + d as u64;
        }
        if !targets.contains(&val) {
            targets.push(val);
        }
    }
    targets.sort();

    (modulus, targets)
}

/// Compute 256^32 mod m (the shift factor for splitting the 64-byte mod into two 32-byte halves)
fn suffix_shift_mod(modulus: u64) -> u64 {
    if modulus == 0 {
        return 0;
    }
    let mut result: u64 = 1;
    for _ in 0..32 {
        result = ((result as u128 * 256) % modulus as u128) as u64;
    }
    result
}

/// Compute view_pub (as big-endian 32-byte number) mod suffix_modulus
fn suffix_view_offset(view_pub_bytes: &[u8; 32], modulus: u64) -> u64 {
    if modulus == 0 {
        return 0;
    }
    let mut result: u64 = 0;
    for &b in view_pub_bytes.iter() {
        result = (result * 256 + b as u64) % modulus;
    }
    result
}

#[inline]
fn cmp_combined_split_to_bound(
    spend_pub_bytes: &[u8; 32],
    view_pub_bytes: &[u8; 32],
    bound: &[u8; 64],
) -> std::cmp::Ordering {
    for i in 0..32 {
        match spend_pub_bytes[i].cmp(&bound[i]) {
            std::cmp::Ordering::Equal => {}
            ord => return ord,
        }
    }
    for i in 0..32 {
        match view_pub_bytes[i].cmp(&bound[32 + i]) {
            std::cmp::Ordering::Equal => {}
            ord => return ord,
        }
    }
    std::cmp::Ordering::Equal
}

#[inline]
fn prefix_match_ranges_split(
    spend_pub_bytes: &[u8; 32],
    view_pub_bytes: &[u8; 32],
    prefix_ranges: &[([u8; 64], [u8; 64])],
) -> bool {
    if prefix_ranges.is_empty() {
        return true;
    }

    let mut left = 0usize;
    let mut right = prefix_ranges.len();
    while left < right {
        let mid = left + ((right - left) >> 1);
        let (lo, hi) = &prefix_ranges[mid];
        match cmp_combined_split_to_bound(spend_pub_bytes, view_pub_bytes, lo) {
            std::cmp::Ordering::Less => {
                right = mid;
            }
            _ => {
                if cmp_combined_split_to_bound(spend_pub_bytes, view_pub_bytes, hi)
                    == std::cmp::Ordering::Less
                {
                    return true;
                }
                left = mid + 1;
            }
        }
    }

    false
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
        let use_legacy_base58_matcher =
            !config.prefix_lower.is_empty() && config.prefix_lower.starts_with('1');

        let prefix_ranges = if use_legacy_base58_matcher {
            Vec::new()
        } else {
            compute_prefix_ranges(&config.prefix_lower)
        };
        let (suffix_modulus, suffix_targets) = compute_suffix_targets(&config.suffix_lower);
        let suffix_shift = suffix_shift_mod(suffix_modulus);
        let prefix_ranges = Arc::new(prefix_ranges);
        let suffix_targets = Arc::new(suffix_targets);
        let prefix_bytes = Arc::<[u8]>::from(config.prefix_lower.as_bytes());
        let suffix_bytes = Arc::<[u8]>::from(config.suffix_lower.as_bytes());

        for _ in 0..config.threads {
            let match_tx = match_tx.clone();
            let counter = counter.clone();
            let prefix_ranges = prefix_ranges.clone();
            let suffix_targets = suffix_targets.clone();
            let prefix_bytes = prefix_bytes.clone();
            let suffix_bytes = suffix_bytes.clone();

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
                combined[32..].copy_from_slice(&view_pub_bytes);
                let suffix_view_mod = suffix_view_offset(&view_pub_bytes, suffix_modulus);

                loop {
                    let spend_pub = spend_pub_point.compress();
                    let spend_pub_bytes = spend_pub.as_bytes();
                    let mut matched_address: Option<String> = None;

                    if use_legacy_base58_matcher {
                        combined[..32].copy_from_slice(spend_pub_bytes);
                        let encoded_len = bs58::encode(&combined)
                            .onto(&mut address_bytes[..])
                            .expect("encoding to byte buffer cannot fail");

                        let address_view = &address_bytes[..encoded_len];
                        let prefix_ok = starts_with_ascii_lowered(address_view, &prefix_bytes);
                        let suffix_ok = ends_with_ascii_lowered(address_view, &suffix_bytes);

                        if prefix_ok && suffix_ok {
                            matched_address = Some(
                                std::str::from_utf8(address_view)
                                    .expect("base58 output must be valid ASCII")
                                    .to_owned(),
                            );
                        }
                    } else {
                        let prefix_ok = prefix_match_ranges_split(
                            spend_pub_bytes,
                            &view_pub_bytes,
                            &prefix_ranges,
                        );

                        let suffix_ok = if !prefix_ok || suffix_targets.is_empty() {
                            prefix_ok
                        } else {
                            let mut spend_mod: u64 = 0;
                            for &byte in spend_pub_bytes {
                                spend_mod = (spend_mod * 256 + byte as u64) % suffix_modulus;
                            }
                            let mod_val = ((spend_mod as u128 * suffix_shift as u128
                                + suffix_view_mod as u128)
                                % suffix_modulus as u128)
                                as u64;
                            suffix_targets.binary_search(&mod_val).is_ok()
                        };

                        if prefix_ok && suffix_ok {
                            combined[..32].copy_from_slice(spend_pub_bytes);
                            matched_address = Some(bs58::encode(&combined).into_string());
                        }
                    }

                    if let Some(address) = matched_address {
                        let wallet = VanityWallet {
                            address,
                            spend_private_key: hex::encode(spend_priv.as_bytes()),
                            spend_public_key: hex::encode(spend_pub_bytes),
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

// ============================================================================
// CUDA Backend - Full GPU key generation
// ============================================================================

#[allow(dead_code)]
struct CudaBackend {
    batch_size: usize,
}

impl CudaBackend {
    fn new(batch_size: usize) -> Self {
        Self { batch_size }
    }
}

// FFI declarations for the new CUDA worker API
#[cfg(all(feature = "cuda", target_os = "linux"))]
unsafe extern "C" {
    fn cuda_init_gen_table(table_data: *const u64, num_points: i32) -> i32;

    fn cuda_worker_create(
        max_batch: i32,
        prefix: *const i8,
        prefix_len: i32,
        suffix: *const i8,
        suffix_len: i32,
        mode: i32,
    ) -> *mut std::ffi::c_void;

    fn cuda_worker_set_ranges(
        handle: *mut std::ffi::c_void,
        prefix_ranges: *const u8,
        num_prefix_ranges: i32,
        suffix_modulus: u64,
        suffix_targets: *const u64,
        num_suffix_targets: i32,
        suffix_shift_mod: u64,
        suffix_view_offset: u64,
    ) -> i32;

    fn cuda_worker_submit_v2(
        handle: *mut std::ffi::c_void,
        start_x: *const u64,
        start_y: *const u64,
        start_z: *const u64,
        start_t: *const u64,
        view_pub: *const u8,
        count: i32,
    ) -> i32;

    fn cuda_worker_get_flags(handle: *mut std::ffi::c_void) -> *const u8;

    #[allow(dead_code)]
    fn cuda_worker_destroy(handle: *mut std::ffi::c_void);

    fn cuda_verify_compress(
        start_x: *const u64,
        start_y: *const u64,
        start_z: *const u64,
        start_t: *const u64,
        count: i32,
        out_compressed: *mut u8,
    ) -> i32;

    fn cuda_diag_compress(
        start_x: *const u64,
        start_y: *const u64,
        start_z: *const u64,
        start_t: *const u64,
        out: *mut u8,
    ) -> i32;
}

/// Extract the internal extended Edwards coordinates from a RistrettoPoint.
/// Returns [X[5], Y[5], Z[5], T[5]] = 20 u64 limbs in radix-2^51.
///
/// SAFETY: This depends on the internal memory layout of curve25519-dalek v4.
/// RistrettoPoint -> EdwardsPoint -> { X: FieldElement51, Y: .., Z: .., T: .. }
/// FieldElement51 = [u64; 5]
#[cfg(all(feature = "cuda", target_os = "linux"))]
fn extract_point_coords(point: &RistrettoPoint) -> [u64; 20] {
    // Verify our assumption about the struct size
    assert_eq!(
        std::mem::size_of::<RistrettoPoint>(),
        160,
        "RistrettoPoint size mismatch - curve25519-dalek internal layout may have changed"
    );

    let ptr = point as *const RistrettoPoint as *const u64;
    let mut coords = [0u64; 20];
    unsafe {
        std::ptr::copy_nonoverlapping(ptr, coords.as_mut_ptr(), 20);
    }
    coords
}

/// Build the generator table: [G, 2G, 4G, 8G, ..., 2^(TABLE_BITS-1)*G]
/// Each entry is 20 u64s (X[5], Y[5], Z[5], T[5])
#[cfg(all(feature = "cuda", target_os = "linux"))]
fn build_gen_table() -> Vec<u64> {
    let mut table = Vec::with_capacity(TABLE_BITS * 20);
    let mut power = RISTRETTO_BASEPOINT_POINT; // G

    for _ in 0..TABLE_BITS {
        let coords = extract_point_coords(&power);
        table.extend_from_slice(&coords);
        power = power + power; // double
    }

    table
}

/// Verify GPU Ristretto compression against CPU for N test points.
/// Runs after the generator table is uploaded.
/// Returns Ok(()) if all match, Err(description) otherwise.
#[cfg(all(feature = "cuda", target_os = "linux"))]
fn verify_gpu_crypto(test_count: usize) -> Result<(), String> {
    // Verify struct layout assumption
    if std::mem::size_of::<RistrettoPoint>() != 160 {
        return Err(format!(
            "RistrettoPoint size is {} bytes, expected 160 - internal layout changed",
            std::mem::size_of::<RistrettoPoint>()
        ));
    }

    // Use a known starting point: 42 * G
    let start_scalar = Scalar::from(42u64);
    let start_point: RistrettoPoint = &start_scalar * RISTRETTO_BASEPOINT_TABLE;
    let coords = extract_point_coords(&start_point);

    // Get GPU compressed results
    let mut gpu_compressed = vec![0u8; test_count * 32];
    let rc = unsafe {
        cuda_verify_compress(
            coords[0..5].as_ptr(),
            coords[5..10].as_ptr(),
            coords[10..15].as_ptr(),
            coords[15..20].as_ptr(),
            test_count as i32,
            gpu_compressed.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(format!("cuda_verify_compress failed with code {}", rc));
    }

    // Compare each against CPU
    for i in 0..test_count {
        let expected_scalar = start_scalar + Scalar::from(i as u64);
        let expected_point = &expected_scalar * RISTRETTO_BASEPOINT_TABLE;
        let expected_compressed = expected_point.compress();
        let expected_bytes = expected_compressed.as_bytes();

        let gpu_bytes = &gpu_compressed[i * 32..(i + 1) * 32];

        if gpu_bytes != expected_bytes {
            return Err(format!(
                "GPU/CPU mismatch at index {}: GPU={} CPU={}",
                i,
                hex::encode(gpu_bytes),
                hex::encode(expected_bytes)
            ));
        }
    }

    Ok(())
}

impl SearchBackend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda-gpu"
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
            // Build and upload generator table
            let gen_table = build_gen_table();
            let rc = unsafe { cuda_init_gen_table(gen_table.as_ptr(), TABLE_BITS as i32) };
            if rc != 0 {
                return Err(format!("cuda_init_gen_table failed with code {}", rc));
            }

            // Verify GPU crypto matches CPU crypto
            eprintln!("  verifying GPU crypto...");
            verify_gpu_crypto(256).map_err(|e| format!("GPU verification failed: {}", e))?;
            eprintln!("  GPU verification passed (256 keys matched CPU)");

            let batch_size = self.batch_size;
            // Use fewer GPU dispatch threads - one per GPU is ideal, but
            // 2-4 allows pipelining while one kernel runs
            let worker_count = config.threads.min(4).max(1);

            // Precompute ranges on CPU (once, shared by all workers)
            let prefix_ranges = compute_prefix_ranges(&config.prefix_lower);
            let (suffix_modulus, suffix_targets) = compute_suffix_targets(&config.suffix_lower);

            // Flatten prefix ranges into contiguous bytes: [lo0[64] hi0[64] lo1[64] hi1[64] ...]
            let prefix_ranges_flat: Vec<u8> = prefix_ranges
                .iter()
                .flat_map(|(lo, hi)| lo.iter().chain(hi.iter()).copied())
                .collect();

            eprintln!(
                "  range matching: {} prefix ranges, {} suffix targets (mod {})",
                prefix_ranges.len(),
                suffix_targets.len(),
                suffix_modulus
            );

            let prefix_ranges_flat = Arc::new(prefix_ranges_flat);
            let suffix_targets = Arc::new(suffix_targets);
            let num_prefix_ranges = prefix_ranges.len() as i32;

            for _ in 0..worker_count {
                let config = config.clone();
                let match_tx = match_tx.clone();
                let counter = counter.clone();
                let prefix_ranges_flat = prefix_ranges_flat.clone();
                let suffix_targets = suffix_targets.clone();

                thread::spawn(move || {
                    let mut rng = rand::thread_rng();

                    // Fixed view keypair per worker
                    let view_priv = random_scalar(&mut rng);
                    let view_pub = (&view_priv * RISTRETTO_BASEPOINT_TABLE).compress();
                    let view_pub_bytes = view_pub.to_bytes();

                    // Random starting scalar
                    let mut spend_priv = random_scalar(&mut rng);
                    let mut spend_pub_point: RistrettoPoint =
                        &spend_priv * RISTRETTO_BASEPOINT_TABLE;

                    // Precompute the batch stride: batch_size * G
                    let batch_stride_scalar = Scalar::from(batch_size as u64);
                    let batch_stride_point = &batch_stride_scalar * RISTRETTO_BASEPOINT_TABLE;

                    // Create CUDA worker with persistent memory
                    let prefix_c = CString::new(config.prefix_lower.as_str()).unwrap();
                    let suffix_c = CString::new(config.suffix_lower.as_str()).unwrap();
                    let handle = unsafe {
                        cuda_worker_create(
                            batch_size as i32,
                            prefix_c.as_ptr(),
                            config.prefix_lower.len() as i32,
                            suffix_c.as_ptr(),
                            config.suffix_lower.len() as i32,
                            1, // mode=1: full GPU keygen
                        )
                    };

                    if handle.is_null() {
                        eprintln!("cuda worker creation failed");
                        return;
                    }

                    // Upload precomputed ranges to GPU
                    let shift_mod = suffix_shift_mod(suffix_modulus);
                    let view_offset = suffix_view_offset(&view_pub_bytes, suffix_modulus);
                    let rc = unsafe {
                        cuda_worker_set_ranges(
                            handle,
                            if prefix_ranges_flat.is_empty() {
                                std::ptr::null()
                            } else {
                                prefix_ranges_flat.as_ptr()
                            },
                            num_prefix_ranges,
                            suffix_modulus,
                            if suffix_targets.is_empty() {
                                std::ptr::null()
                            } else {
                                suffix_targets.as_ptr()
                            },
                            suffix_targets.len() as i32,
                            shift_mod,
                            view_offset,
                        )
                    };
                    if rc != 0 {
                        eprintln!("cuda_worker_set_ranges failed with code {}", rc);
                        return;
                    }

                    // Get pointer to pinned host flag buffer (allocated by CUDA)
                    let flags_ptr = unsafe { cuda_worker_get_flags(handle) };

                    loop {
                        // Extract starting point coordinates
                        let coords = extract_point_coords(&spend_pub_point);

                        // Submit to GPU: GPU does point addition + Ristretto compression +
                        // range-based pattern matching for all batch_size candidates
                        let rc = unsafe {
                            cuda_worker_submit_v2(
                                handle,
                                coords[0..5].as_ptr(),   // X
                                coords[5..10].as_ptr(),  // Y
                                coords[10..15].as_ptr(), // Z
                                coords[15..20].as_ptr(), // T
                                view_pub_bytes.as_ptr(),
                                batch_size as i32,
                            )
                        };

                        if rc != 0 {
                            eprintln!("cuda worker submit failed with code {}", rc);
                            return;
                        }

                        // Check for matches in pinned host memory (very rare)
                        let flags = unsafe { std::slice::from_raw_parts(flags_ptr, batch_size) };
                        for i in 0..batch_size {
                            if flags[i] == 0 {
                                continue;
                            }

                            // Reconstruct the matching wallet on CPU
                            let match_priv = spend_priv + Scalar::from(i as u64);
                            let match_pub = (&match_priv * RISTRETTO_BASEPOINT_TABLE).compress();
                            let match_pub_bytes = match_pub.to_bytes();

                            let mut combined = [0u8; 64];
                            combined[..32].copy_from_slice(&match_pub_bytes);
                            combined[32..].copy_from_slice(&view_pub_bytes);
                            let address = bs58::encode(&combined).into_string();

                            let wallet = VanityWallet {
                                address,
                                spend_private_key: hex::encode(match_priv.as_bytes()),
                                spend_public_key: hex::encode(match_pub_bytes),
                                view_private_key: hex::encode(view_priv.as_bytes()),
                                view_public_key: hex::encode(view_pub_bytes),
                            };

                            if match_tx.send(wallet).is_err() {
                                return;
                            }
                        }

                        // Advance to next batch: one point addition instead of batch_size
                        spend_priv += batch_stride_scalar;
                        spend_pub_point = spend_pub_point + batch_stride_point;

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

    // CUDA range matching doesn't support prefixes starting with '1' (leading zero byte)
    if args.cuda && args.prefix.starts_with('1') {
        eprintln!("error: CUDA backend does not support prefixes starting with '1'");
        eprintln!(
            "       (leading '1' in base58 encodes a zero byte, which requires special handling)"
        );
        std::process::exit(1);
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
        if args.cuda {
            // Two submit threads tend to outperform four on desktop GPUs that also drive displays.
            2
        } else {
            thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        }
    });

    let batch_size = if args.cuda {
        // Clamp batch size to max supported by table (2^TABLE_BITS)
        // and round down to multiple of 8 (KEYS_PER_THREAD)
        let max_batch = 1 << TABLE_BITS;
        (args.batch_size.min(max_batch)) & !7
    } else {
        args.batch_size
    };

    let config = Arc::new(SearchConfig {
        prefix_lower: args.prefix.to_ascii_lowercase(),
        suffix_lower: args.suffix.to_ascii_lowercase(),
        threads: num_threads,
        batch_size,
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
        Box::new(CudaBackend::new(batch_size))
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
    if args.cuda {
        eprintln!("  batch size: {}", format_count(batch_size as u64));
    }
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
        let estimate = if rate > 0.0 {
            (expected / rate) as u64
        } else {
            0
        };
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_pattern_valid() {
        assert!(validate_pattern("abc").is_ok());
        assert!(validate_pattern("123").is_ok());
        assert!(validate_pattern("ABCdef").is_ok());
        assert!(validate_pattern("z").is_ok());
    }

    #[test]
    fn test_validate_pattern_invalid() {
        assert!(validate_pattern("0").is_err()); // zero not in base58
        assert!(validate_pattern("!").is_err()); // symbol not in base58
        assert!(validate_pattern(" ").is_err()); // space not in base58
                                                 // O, I, l are valid for case-insensitive search (match o, i, L)
        assert!(validate_pattern("O").is_ok());
        assert!(validate_pattern("I").is_ok());
        assert!(validate_pattern("l").is_ok());
    }

    #[test]
    fn test_starts_with_ascii_lowered() {
        assert!(starts_with_ascii_lowered(b"ABCdef", b"abc"));
        assert!(starts_with_ascii_lowered(b"abcdef", b"abc"));
        assert!(!starts_with_ascii_lowered(b"xbcdef", b"abc"));
        assert!(starts_with_ascii_lowered(b"anything", b""));
        assert!(!starts_with_ascii_lowered(b"ab", b"abc"));
    }

    #[test]
    fn test_ends_with_ascii_lowered() {
        assert!(ends_with_ascii_lowered(b"abcDEF", b"def"));
        assert!(ends_with_ascii_lowered(b"abcdef", b"def"));
        assert!(!ends_with_ascii_lowered(b"abcxyz", b"def"));
        assert!(ends_with_ascii_lowered(b"anything", b""));
        assert!(!ends_with_ascii_lowered(b"ef", b"def"));
    }

    #[test]
    fn test_cpu_keygen_produces_valid_addresses() {
        // Generate a keypair and verify the address is valid base58
        let mut rng = rand::thread_rng();
        let spend_priv = random_scalar(&mut rng);
        let view_priv = random_scalar(&mut rng);

        let spend_pub = (&spend_priv * RISTRETTO_BASEPOINT_TABLE).compress();
        let view_pub = (&view_priv * RISTRETTO_BASEPOINT_TABLE).compress();

        let mut combined = [0u8; 64];
        combined[..32].copy_from_slice(spend_pub.as_bytes());
        combined[32..].copy_from_slice(view_pub.as_bytes());

        let address = bs58::encode(&combined).into_string();

        // Address should be valid base58 and reasonable length (~87 chars for 64 bytes)
        assert!(
            address.len() >= 80 && address.len() <= 90,
            "address length {} outside expected range",
            address.len()
        );

        // Round-trip: decode and re-encode should match
        let decoded = bs58::decode(&address).into_vec().unwrap();
        assert_eq!(decoded, combined.to_vec());
    }

    #[test]
    fn test_incremental_keygen_consistency() {
        // Verify that scalar increment + point addition produces consistent keys
        let mut rng = rand::thread_rng();
        let start_priv = random_scalar(&mut rng);
        let mut point = &start_priv * RISTRETTO_BASEPOINT_TABLE;

        for i in 0u64..100 {
            let expected_priv = start_priv + Scalar::from(i);
            let expected_pub = (&expected_priv * RISTRETTO_BASEPOINT_TABLE).compress();
            let actual_pub = point.compress();

            assert_eq!(
                expected_pub.as_bytes(),
                actual_pub.as_bytes(),
                "mismatch at increment {}",
                i
            );

            point += RISTRETTO_BASEPOINT_POINT;
        }
    }

    #[test]
    fn test_wallet_keys_are_consistent() {
        // Generate a wallet and verify that the private key reproduces the public key
        let mut rng = rand::thread_rng();
        let spend_priv = random_scalar(&mut rng);
        let view_priv = random_scalar(&mut rng);

        let spend_pub = (&spend_priv * RISTRETTO_BASEPOINT_TABLE).compress();
        let view_pub = (&view_priv * RISTRETTO_BASEPOINT_TABLE).compress();

        let wallet = VanityWallet {
            address: "test".into(),
            spend_private_key: hex::encode(spend_priv.as_bytes()),
            spend_public_key: hex::encode(spend_pub.as_bytes()),
            view_private_key: hex::encode(view_priv.as_bytes()),
            view_public_key: hex::encode(view_pub.as_bytes()),
        };

        // Recover private keys from hex and verify they regenerate the public keys
        let recovered_spend_priv_bytes = hex::decode(&wallet.spend_private_key).unwrap();
        let recovered_view_priv_bytes = hex::decode(&wallet.view_private_key).unwrap();

        let mut spend_arr = [0u8; 32];
        let mut view_arr = [0u8; 32];
        spend_arr.copy_from_slice(&recovered_spend_priv_bytes);
        view_arr.copy_from_slice(&recovered_view_priv_bytes);

        let recovered_spend_priv = Scalar::from_canonical_bytes(spend_arr);
        let recovered_view_priv = Scalar::from_canonical_bytes(view_arr);

        // Scalars from random_scalar may not be canonical, so we check via
        // public key regeneration using the original bytes
        let regen_spend_pub = (&spend_priv * RISTRETTO_BASEPOINT_TABLE).compress();
        let regen_view_pub = (&view_priv * RISTRETTO_BASEPOINT_TABLE).compress();

        assert_eq!(
            hex::encode(regen_spend_pub.as_bytes()),
            wallet.spend_public_key
        );
        assert_eq!(
            hex::encode(regen_view_pub.as_bytes()),
            wallet.view_public_key
        );

        // If canonical, also verify round-trip
        if let Some(s) = recovered_spend_priv.into() {
            let rpub = (&s * RISTRETTO_BASEPOINT_TABLE).compress();
            assert_eq!(hex::encode(rpub.as_bytes()), wallet.spend_public_key);
        }
        if let Some(v) = recovered_view_priv.into() {
            let rpub = (&v * RISTRETTO_BASEPOINT_TABLE).compress();
            assert_eq!(hex::encode(rpub.as_bytes()), wallet.view_public_key);
        }
    }

    #[test]
    fn test_format_count() {
        assert_eq!(format_count(0), "0");
        assert_eq!(format_count(999), "999");
        assert_eq!(format_count(1000), "1,000");
        assert_eq!(format_count(1234567), "1,234,567");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(0), "0s");
        assert_eq!(format_duration(59), "59s");
        assert_eq!(format_duration(60), "1m 00s");
        assert_eq!(format_duration(3661), "1h 01m 01s");
        assert_eq!(format_duration(86400), "1d 00h 00m 00s");
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn test_ristretto_point_size() {
        // Guard against dalek internal layout changes
        assert_eq!(
            std::mem::size_of::<RistrettoPoint>(),
            160,
            "RistrettoPoint size changed - extract_point_coords needs update"
        );
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn test_extract_point_coords_deterministic() {
        let point = &Scalar::from(12345u64) * RISTRETTO_BASEPOINT_TABLE;
        let c1 = extract_point_coords(&point);
        let c2 = extract_point_coords(&point);
        assert_eq!(c1, c2);
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn test_gen_table_has_correct_length() {
        let table = build_gen_table();
        assert_eq!(table.len(), TABLE_BITS * 20);
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn test_gpu_ristretto_matches_cpu() {
        // Upload generator table
        let gen_table = build_gen_table();
        let rc = unsafe { cuda_init_gen_table(gen_table.as_ptr(), TABLE_BITS as i32) };
        assert_eq!(rc, 0, "cuda_init_gen_table failed");

        // Verify 256 keys match CPU
        verify_gpu_crypto(256).expect("GPU crypto verification failed");
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn test_gpu_different_starting_points() {
        // Upload generator table
        let gen_table = build_gen_table();
        let rc = unsafe { cuda_init_gen_table(gen_table.as_ptr(), TABLE_BITS as i32) };
        assert_eq!(rc, 0);

        // Test with several different starting scalars
        for start_val in [0u64, 1, 100, 999999, u64::MAX / 2] {
            let start_scalar = Scalar::from(start_val);
            let start_point: RistrettoPoint = &start_scalar * RISTRETTO_BASEPOINT_TABLE;
            let coords = extract_point_coords(&start_point);

            let test_count = 16;
            let mut gpu_compressed = vec![0u8; test_count * 32];
            let rc = unsafe {
                cuda_verify_compress(
                    coords[0..5].as_ptr(),
                    coords[5..10].as_ptr(),
                    coords[10..15].as_ptr(),
                    coords[15..20].as_ptr(),
                    test_count as i32,
                    gpu_compressed.as_mut_ptr(),
                )
            };
            assert_eq!(rc, 0, "cuda_verify_compress failed for start={}", start_val);

            for i in 0..test_count {
                let expected = (&(start_scalar + Scalar::from(i as u64))
                    * RISTRETTO_BASEPOINT_TABLE)
                    .compress();

                let gpu_bytes = &gpu_compressed[i * 32..(i + 1) * 32];
                assert_eq!(
                    gpu_bytes,
                    expected.as_bytes(),
                    "mismatch at start={} index={}",
                    start_val,
                    i
                );
            }
        }
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn test_gpu_vanity_match_agrees_with_cpu() {
        // Upload generator table
        let gen_table = build_gen_table();
        let rc = unsafe { cuda_init_gen_table(gen_table.as_ptr(), TABLE_BITS as i32) };
        assert_eq!(rc, 0);

        let start_scalar = Scalar::from(77u64);
        let start_point: RistrettoPoint = &start_scalar * RISTRETTO_BASEPOINT_TABLE;
        let coords = extract_point_coords(&start_point);

        let view_priv = Scalar::from(99u64);
        let view_pub = (&view_priv * RISTRETTO_BASEPOINT_TABLE).compress();
        let view_pub_bytes = view_pub.to_bytes();

        // Use a 1-char prefix that matches frequently
        let prefix = "b";
        let prefix_ranges = compute_prefix_ranges(prefix);
        let prefix_ranges_flat: Vec<u8> = prefix_ranges
            .iter()
            .flat_map(|(lo, hi)| lo.iter().chain(hi.iter()).copied())
            .collect();

        let prefix_c = CString::new(prefix).unwrap();
        let suffix_c = CString::new("").unwrap();

        let batch_size = 4096;
        let handle = unsafe {
            cuda_worker_create(
                batch_size as i32,
                prefix_c.as_ptr(),
                prefix.len() as i32,
                suffix_c.as_ptr(),
                0,
                1, // full GPU mode
            )
        };
        assert!(!handle.is_null());

        // Set ranges for range-based matching
        let rc = unsafe {
            cuda_worker_set_ranges(
                handle,
                if prefix_ranges_flat.is_empty() {
                    std::ptr::null()
                } else {
                    prefix_ranges_flat.as_ptr()
                },
                prefix_ranges.len() as i32,
                0, // no suffix
                std::ptr::null(),
                0,
                0,
                0,
            )
        };
        assert_eq!(rc, 0, "cuda_worker_set_ranges failed");

        let rc = unsafe {
            cuda_worker_submit_v2(
                handle,
                coords[0..5].as_ptr(),
                coords[5..10].as_ptr(),
                coords[10..15].as_ptr(),
                coords[15..20].as_ptr(),
                view_pub_bytes.as_ptr(),
                batch_size as i32,
            )
        };
        assert_eq!(rc, 0, "cuda_worker_submit_v2 failed");

        let flags =
            unsafe { std::slice::from_raw_parts(cuda_worker_get_flags(handle), batch_size) };

        // Verify every GPU match/non-match against CPU
        let mut match_count = 0;
        for i in 0..batch_size {
            let priv_i = start_scalar + Scalar::from(i as u64);
            let pub_i = (&priv_i * RISTRETTO_BASEPOINT_TABLE).compress();

            let mut combined = [0u8; 64];
            combined[..32].copy_from_slice(pub_i.as_bytes());
            combined[32..].copy_from_slice(&view_pub_bytes);

            let address = bs58::encode(&combined).into_string();
            let cpu_match = address.to_ascii_lowercase().starts_with(prefix);

            let gpu_match = flags[i] != 0;

            assert_eq!(
                gpu_match, cpu_match,
                "GPU/CPU match disagree at index {}: gpu={} cpu={} addr={}",
                i, gpu_match, cpu_match, address
            );

            if gpu_match {
                match_count += 1;
            }
        }

        // With prefix "b" and 4096 candidates, we should find some matches
        assert!(
            match_count > 0,
            "no matches found in 4096 candidates with prefix 'b'"
        );

        unsafe {
            cuda_worker_destroy(handle);
        }
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn test_diag_gpu_coords_and_compress() {
        // Use a known scalar
        let s = Scalar::from(42u64);
        let point = &s * RISTRETTO_BASEPOINT_TABLE;
        let coords = extract_point_coords(&point);

        let mut diag = [0u8; 676];
        let rc = unsafe {
            cuda_diag_compress(
                coords[0..5].as_ptr(),
                coords[5..10].as_ptr(),
                coords[10..15].as_ptr(),
                coords[15..20].as_ptr(),
                diag.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0, "cuda_diag_compress failed");

        let labels = [
            "u1",
            "u2",
            "inv",
            "den1",
            "den2",
            "z_inv",
            "ix0",
            "iy0",
            "ench_den",
            "t_zinv",
            "x(rot)",
            "y(rot)",
            "den_inv(rot)",
            "x_zinv",
            "y_before_neg",
            "Z-y",
            "s_before_abs",
            "s_final",
            "compressed",
        ];
        for (i, label) in labels.iter().enumerate() {
            let bytes = &diag[i * 32..(i + 1) * 32];
            eprintln!("GPU {:>15}: {}", label, hex::encode(bytes));
        }
        let rotate = diag[608];
        let neg_y = diag[609];
        let was_sq = diag[610];
        eprintln!("rotate={} neg_y={} was_sq={}", rotate, neg_y, was_sq);

        // Now compute u1 = (Z+Y)*(Z-Y) on CPU using the same limbs and same arithmetic
        let x_limbs = &coords[0..5];
        let y_limbs = &coords[5..10];
        let z_limbs = &coords[10..15];
        let t_limbs = &coords[15..20];

        fn fe_add_limbs(f: &[u64], g: &[u64]) -> [u64; 5] {
            [
                f[0] + g[0],
                f[1] + g[1],
                f[2] + g[2],
                f[3] + g[3],
                f[4] + g[4],
            ]
        }
        fn fe_sub_limbs(f: &[u64], g: &[u64]) -> [u64; 5] {
            let mask: u64 = (1u64 << 51) - 1;
            [
                (f[0] + 2 * ((1u64 << 51) - 19)) - g[0],
                (f[1] + 2 * mask) - g[1],
                (f[2] + 2 * mask) - g[2],
                (f[3] + 2 * mask) - g[3],
                (f[4] + 2 * mask) - g[4],
            ]
        }
        fn fe_mul_limbs(f: &[u64], g: &[u64]) -> [u64; 5] {
            let (f0, f1, f2, f3, f4) = (f[0], f[1], f[2], f[3], f[4]);
            let (g0, g1, g2, g3, g4) = (g[0], g[1], g[2], g[3], g[4]);
            let (g1_19, g2_19, g3_19, g4_19) = (g1 * 19, g2 * 19, g3 * 19, g4 * 19);
            let h0: u128 = f0 as u128 * g0 as u128
                + f1 as u128 * g4_19 as u128
                + f2 as u128 * g3_19 as u128
                + f3 as u128 * g2_19 as u128
                + f4 as u128 * g1_19 as u128;
            let h1: u128 = f0 as u128 * g1 as u128
                + f1 as u128 * g0 as u128
                + f2 as u128 * g4_19 as u128
                + f3 as u128 * g3_19 as u128
                + f4 as u128 * g2_19 as u128;
            let h2: u128 = f0 as u128 * g2 as u128
                + f1 as u128 * g1 as u128
                + f2 as u128 * g0 as u128
                + f3 as u128 * g4_19 as u128
                + f4 as u128 * g3_19 as u128;
            let h3: u128 = f0 as u128 * g3 as u128
                + f1 as u128 * g2 as u128
                + f2 as u128 * g1 as u128
                + f3 as u128 * g0 as u128
                + f4 as u128 * g4_19 as u128;
            let h4: u128 = f0 as u128 * g4 as u128
                + f1 as u128 * g3 as u128
                + f2 as u128 * g2 as u128
                + f3 as u128 * g1 as u128
                + f4 as u128 * g0 as u128;
            let mask: u64 = (1u64 << 51) - 1;
            let c = (h0 >> 51) as u64;
            let v0 = h0 as u64 & mask;
            let h1 = h1 + c as u128;
            let c = (h1 >> 51) as u64;
            let v1 = h1 as u64 & mask;
            let h2 = h2 + c as u128;
            let c = (h2 >> 51) as u64;
            let v2 = h2 as u64 & mask;
            let h3 = h3 + c as u128;
            let c = (h3 >> 51) as u64;
            let v3 = h3 as u64 & mask;
            let h4 = h4 + c as u128;
            let c = (h4 >> 51) as u64;
            let v4 = h4 as u64 & mask;
            let mut r = [v0 + c * 19, v1, v2, v3, v4];
            let c = r[0] >> 51;
            r[0] &= mask;
            r[1] += c;
            r
        }
        fn limbs_to_bytes_v2(v: &[u64; 5]) -> [u8; 32] {
            // reduce first
            let mut h = *v;
            let mask: u64 = (1u64 << 51) - 1;
            let mut c: u64;
            c = h[0] >> 51;
            h[0] &= mask;
            h[1] += c;
            c = h[1] >> 51;
            h[1] &= mask;
            h[2] += c;
            c = h[2] >> 51;
            h[2] &= mask;
            h[3] += c;
            c = h[3] >> 51;
            h[3] &= mask;
            h[4] += c;
            c = h[4] >> 51;
            h[4] &= mask;
            h[0] += c * 19;
            c = h[0] >> 51;
            h[0] &= mask;
            h[1] += c;
            let mut q = (h[0] + 19) >> 51;
            q = (h[1] + q) >> 51;
            q = (h[2] + q) >> 51;
            q = (h[3] + q) >> 51;
            q = (h[4] + q) >> 51;
            h[0] += q * 19;
            c = h[0] >> 51;
            h[0] &= mask;
            h[1] += c;
            c = h[1] >> 51;
            h[1] &= mask;
            h[2] += c;
            c = h[2] >> 51;
            h[2] &= mask;
            h[3] += c;
            c = h[3] >> 51;
            h[3] &= mask;
            h[4] += c;
            h[4] &= mask;
            let lo0 = h[0] | (h[1] << 51);
            let lo1 = (h[1] >> 13) | (h[2] << 38);
            let lo2 = (h[2] >> 26) | (h[3] << 25);
            let lo3 = (h[3] >> 39) | (h[4] << 12);
            let mut out = [0u8; 32];
            out[0..8].copy_from_slice(&lo0.to_le_bytes());
            out[8..16].copy_from_slice(&lo1.to_le_bytes());
            out[16..24].copy_from_slice(&lo2.to_le_bytes());
            out[24..32].copy_from_slice(&lo3.to_le_bytes());
            out
        }

        let t0 = fe_add_limbs(z_limbs, y_limbs);
        let t1 = fe_sub_limbs(z_limbs, y_limbs);
        let cpu_u1 = fe_mul_limbs(&t0, &t1);
        let cpu_u1_bytes = limbs_to_bytes_v2(&cpu_u1);
        eprintln!("CPU             u1: {}", hex::encode(cpu_u1_bytes));

        let cpu_u2 = fe_mul_limbs(x_limbs, y_limbs);
        let cpu_u2_bytes = limbs_to_bytes_v2(&cpu_u2);
        eprintln!("CPU             u2: {}", hex::encode(cpu_u2_bytes));

        fn fe_sq_limbs(f: &[u64]) -> [u64; 5] {
            let (f0, f1, f2, f3, f4) = (f[0], f[1], f[2], f[3], f[4]);
            let f0_2 = f0 * 2;
            let f1_2 = f1 * 2;

            let h0: u128 = f0 as u128 * f0 as u128
                + (f1 * 38) as u128 * f4 as u128
                + (f2 * 38) as u128 * f3 as u128;
            let h1: u128 = f0_2 as u128 * f1 as u128
                + (f2 * 38) as u128 * f4 as u128
                + (f3 * 19) as u128 * f3 as u128;
            let h2: u128 = f0_2 as u128 * f2 as u128
                + f1 as u128 * f1 as u128
                + (f3 * 38) as u128 * f4 as u128;
            let h3: u128 = f0_2 as u128 * f3 as u128
                + f1_2 as u128 * f2 as u128
                + (f4 * 19) as u128 * f4 as u128;
            let h4: u128 =
                f0_2 as u128 * f4 as u128 + f1_2 as u128 * f3 as u128 + f2 as u128 * f2 as u128;

            let mask: u64 = (1u64 << 51) - 1;
            let c = (h0 >> 51) as u64;
            let v0 = h0 as u64 & mask;
            let h1 = h1 + c as u128;
            let c = (h1 >> 51) as u64;
            let v1 = h1 as u64 & mask;
            let h2 = h2 + c as u128;
            let c = (h2 >> 51) as u64;
            let v2 = h2 as u64 & mask;
            let h3 = h3 + c as u128;
            let c = (h3 >> 51) as u64;
            let v3 = h3 as u64 & mask;
            let h4 = h4 + c as u128;
            let c = (h4 >> 51) as u64;
            let v4 = h4 as u64 & mask;
            let mut r = [v0 + c * 19, v1, v2, v3, v4];
            let c = r[0] >> 51;
            r[0] &= mask;
            r[1] += c;
            r
        }

        // u2_sq = u2^2
        let cpu_u2_sq = fe_sq_limbs(&cpu_u2);
        let cpu_u2_sq_bytes = limbs_to_bytes_v2(&cpu_u2_sq);
        eprintln!("CPU        u2_sq: {}", hex::encode(cpu_u2_sq_bytes));

        // u1_u2sq = u1 * u2^2
        let cpu_u1_u2sq = fe_mul_limbs(&cpu_u1, &cpu_u2_sq);
        let cpu_u1_u2sq_bytes = limbs_to_bytes_v2(&cpu_u1_u2sq);
        eprintln!("CPU      u1_u2sq: {}", hex::encode(cpu_u1_u2sq_bytes));

        // Now compute sqrt_ratio_m1(1, u1_u2sq) step by step
        // v = u1_u2sq, u = 1
        // v3 = v^2 * v
        let v = cpu_u1_u2sq;
        let v_sq = fe_sq_limbs(&v);
        let v3 = fe_mul_limbs(&v_sq, &v);
        let v3_bytes = limbs_to_bytes_v2(&v3);
        eprintln!("CPU           v3: {}", hex::encode(v3_bytes));

        // v7 = v3^2 * v
        let v3_sq = fe_sq_limbs(&v3);
        let v7 = fe_mul_limbs(&v3_sq, &v);
        let v7_bytes = limbs_to_bytes_v2(&v7);
        eprintln!("CPU           v7: {}", hex::encode(v7_bytes));

        // uv7 = 1 * v7 = v7
        // pow22523(uv7)
        // This is too complex to inline. Let me just check if inv matches.
        let gpu_inv = &diag[2 * 32..3 * 32];
        eprintln!("GPU          inv: {}", hex::encode(gpu_inv));

        // Check first divergence
        let gpu_u1 = &diag[0..32];
        let gpu_u2 = &diag[1 * 32..2 * 32];
        assert_eq!(gpu_u1, &cpu_u1_bytes, "u1 mismatch");
        assert_eq!(gpu_u2, &cpu_u2_bytes, "u2 mismatch");

        // Implement full sqrt_ratio_m1 on CPU to find where GPU diverges
        fn fe_sq_n_limbs(f: &[u64; 5], n: usize) -> [u64; 5] {
            let mut h = fe_sq_limbs(f);
            for _ in 1..n {
                h = fe_sq_limbs(&h);
            }
            h
        }
        fn fe_neg_limbs(f: &[u64; 5]) -> [u64; 5] {
            let zero = [0u64; 5];
            fe_sub_limbs(&zero, f)
        }
        fn fe_pow22523_limbs(f: &[u64; 5]) -> [u64; 5] {
            let mut t0 = fe_sq_limbs(f); // f^2
            let mut t1 = fe_sq_n_limbs(&t0, 2); // f^8
            t1 = fe_mul_limbs(f, &t1); // f^9
            t0 = fe_mul_limbs(&t0, &t1); // f^11
            t0 = fe_sq_limbs(&t0); // f^22
            t0 = fe_mul_limbs(&t1, &t0); // f^31
            t1 = fe_sq_n_limbs(&t0, 5); // f^(2^10-32)
            t0 = fe_mul_limbs(&t1, &t0); // f^(2^10-1)
            t1 = fe_sq_n_limbs(&t0, 10); // f^(2^20-2^10)
            t1 = fe_mul_limbs(&t1, &t0); // f^(2^20-1)
            let mut t2 = fe_sq_n_limbs(&t1, 20); // f^(2^40-2^20)
            t1 = fe_mul_limbs(&t2, &t1); // f^(2^40-1)
            t1 = fe_sq_n_limbs(&t1, 10); // f^(2^50-2^10)
            t0 = fe_mul_limbs(&t1, &t0); // f^(2^50-1)
            t1 = fe_sq_n_limbs(&t0, 50); // f^(2^100-2^50)
            t1 = fe_mul_limbs(&t1, &t0); // f^(2^100-1)
            t2 = fe_sq_n_limbs(&t1, 100); // f^(2^200-2^100)
            t1 = fe_mul_limbs(&t2, &t1); // f^(2^200-1)
            t1 = fe_sq_n_limbs(&t1, 50); // f^(2^250-2^50)
            t0 = fe_mul_limbs(&t1, &t0); // f^(2^250-1)
            t0 = fe_sq_n_limbs(&t0, 2); // f^(2^252-4)
            fe_mul_limbs(&t0, f) // f^(2^252-3)
        }
        fn fe_tobytes_limbs(v: &[u64; 5]) -> [u8; 32] {
            limbs_to_bytes_v2(v)
        }
        fn fe_equal_limbs(f: &[u64; 5], g: &[u64; 5]) -> bool {
            fe_tobytes_limbs(f) == fe_tobytes_limbs(g)
        }
        fn fe_isneg_limbs(f: &[u64; 5]) -> bool {
            fe_tobytes_limbs(f)[0] & 1 == 1
        }
        fn fe_cmov_limbs(f: &mut [u64; 5], g: &[u64; 5], b: bool) {
            if b {
                *f = *g;
            }
        }
        fn fe_abs_limbs(f: &[u64; 5]) -> [u64; 5] {
            if fe_isneg_limbs(f) {
                fe_neg_limbs(f)
            } else {
                *f
            }
        }

        let sqrt_m1: [u64; 5] = [
            1718705420411056,
            234908883556509,
            2233514472574048,
            2117202627021982,
            765476049583133,
        ];

        fn sqrt_ratio_m1_cpu(u: &[u64; 5], v: &[u64; 5], sqrt_m1: &[u64; 5]) -> ([u64; 5], bool) {
            let v_sq = fe_sq_limbs(v);
            let v3 = fe_mul_limbs(&v_sq, v);
            let v3_sq = fe_sq_limbs(&v3);
            let v7 = fe_mul_limbs(&v3_sq, v);
            let uv7 = fe_mul_limbs(u, &v7);
            let mut r = fe_pow22523_limbs(&uv7);
            r = fe_mul_limbs(&r, &v3);
            r = fe_mul_limbs(&r, u);

            let r_sq = fe_sq_limbs(&r);
            let check = fe_mul_limbs(v, &r_sq);

            let neg_u = fe_neg_limbs(u);
            let neg_u_i = fe_mul_limbs(&neg_u, sqrt_m1);

            let correct_sign = fe_equal_limbs(&check, u);
            let flipped_sign = fe_equal_limbs(&check, &neg_u);
            let flipped_sign_i = fe_equal_limbs(&check, &neg_u_i);

            let r_prime = fe_mul_limbs(&r, sqrt_m1);
            fe_cmov_limbs(&mut r, &r_prime, flipped_sign | flipped_sign_i);
            r = fe_abs_limbs(&r);

            (r, correct_sign | flipped_sign)
        }

        let one_limbs: [u64; 5] = [1, 0, 0, 0, 0];
        let (cpu_inv, _) = sqrt_ratio_m1_cpu(&one_limbs, &cpu_u1_u2sq, &sqrt_m1);
        let cpu_inv_bytes = limbs_to_bytes_v2(&cpu_inv);
        eprintln!("CPU          inv: {}", hex::encode(cpu_inv_bytes));
        eprintln!("GPU          inv: {}", hex::encode(&diag[2 * 32..3 * 32]));
        if cpu_inv_bytes == diag[2 * 32..3 * 32] {
            eprintln!("*** inv MATCHES - bug is after sqrt_ratio_m1 ***");
        } else {
            eprintln!("*** inv MISMATCH - bug is in sqrt_ratio_m1/pow22523 ***");
        }

        // Verify Edwards invariant: T*Z == X*Y (if fields in expected order)
        let xy_limbs = fe_mul_limbs(x_limbs, y_limbs);
        let tz_limbs = fe_mul_limbs(
            &coords[15..20].try_into().unwrap_or([0; 5]),
            &coords[10..15].try_into().unwrap_or([0; 5]),
        );
        let xy_bytes = limbs_to_bytes_v2(&xy_limbs);
        let tz_bytes = limbs_to_bytes_v2(&tz_limbs);
        eprintln!("X*Y: {}", hex::encode(xy_bytes));
        eprintln!("T*Z: {}", hex::encode(tz_bytes));
        if xy_bytes != tz_bytes {
            eprintln!(
                "*** FIELDS REORDERED! T*Z != X*Y - extract_point_coords has wrong field order ***"
            );

            // Try all 24 permutations of the 4 field elements
            let fields = [
                (&coords[0..5], "f0"),
                (&coords[5..10], "f1"),
                (&coords[10..15], "f2"),
                (&coords[15..20], "f3"),
            ];
            for ix in 0..4 {
                for iy in 0..4 {
                    if iy == ix {
                        continue;
                    }
                    for iz in 0..4 {
                        if iz == ix || iz == iy {
                            continue;
                        }
                        let it = 6 - ix - iy - iz; // remaining index
                        let xy = fe_mul_limbs(fields[ix].0, fields[iy].0);
                        let tz = fe_mul_limbs(fields[it].0, fields[iz].0);
                        if limbs_to_bytes_v2(&xy) == limbs_to_bytes_v2(&tz) {
                            eprintln!(
                                "Found correct ordering: X={} Y={} Z={} T={}",
                                fields[ix].1, fields[iy].1, fields[iz].1, fields[it].1
                            );
                        }
                    }
                }
            }
        } else {
            eprintln!("Edwards invariant T*Z == X*Y holds - field order is correct");
        }

        // Now continue the ristretto_encode on CPU
        let cpu_den1 = fe_mul_limbs(&cpu_inv, &cpu_u1);
        let cpu_den2 = fe_mul_limbs(&cpu_inv, &cpu_u2);
        let cpu_z_inv_tmp = fe_mul_limbs(&cpu_den1, &cpu_den2);
        let t_limbs_arr: [u64; 5] = [coords[15], coords[16], coords[17], coords[18], coords[19]];
        let cpu_z_inv = fe_mul_limbs(&cpu_z_inv_tmp, &t_limbs_arr);

        let x_arr: [u64; 5] = [coords[0], coords[1], coords[2], coords[3], coords[4]];
        let y_arr: [u64; 5] = [coords[5], coords[6], coords[7], coords[8], coords[9]];
        let cpu_ix0 = fe_mul_limbs(&x_arr, &sqrt_m1);
        let cpu_iy0 = fe_mul_limbs(&y_arr, &sqrt_m1);
        let invsqrt_a_minus_d: [u64; 5] = [
            278908739862762,
            821645201101625,
            8113234426968,
            1777959178193151,
            2118520810568447,
        ];
        let cpu_ench_den = fe_mul_limbs(&cpu_den1, &invsqrt_a_minus_d);

        let cpu_t_zinv = fe_mul_limbs(&t_limbs_arr, &cpu_z_inv);
        let cpu_rotate = fe_isneg_limbs(&cpu_t_zinv);

        let mut cpu_x = x_arr;
        fe_cmov_limbs(&mut cpu_x, &cpu_iy0, cpu_rotate);
        let mut cpu_y = y_arr;
        fe_cmov_limbs(&mut cpu_y, &cpu_ix0, cpu_rotate);
        let mut cpu_den_inv = cpu_den2;
        fe_cmov_limbs(&mut cpu_den_inv, &cpu_ench_den, cpu_rotate);

        let cpu_x_zinv = fe_mul_limbs(&cpu_x, &cpu_z_inv);
        let cpu_neg_y = fe_isneg_limbs(&cpu_x_zinv);

        let mut cpu_y_final = cpu_y;
        if cpu_neg_y {
            cpu_y_final = fe_neg_limbs(&cpu_y);
        }

        let z_arr: [u64; 5] = [coords[10], coords[11], coords[12], coords[13], coords[14]];
        let cpu_t0 = fe_sub_limbs(&z_arr, &cpu_y_final);
        let cpu_s = fe_mul_limbs(&cpu_den_inv, &cpu_t0);
        let cpu_s_final = fe_abs_limbs(&cpu_s);

        // Print all CPU intermediates and compare
        let vals = [
            ("den1", &cpu_den1),
            ("den2", &cpu_den2),
            ("z_inv", &cpu_z_inv),
            ("ix0", &cpu_ix0),
            ("iy0", &cpu_iy0),
            ("ench_den", &cpu_ench_den),
            ("t_zinv", &cpu_t_zinv),
            ("x(rot)", &cpu_x),
            ("y(rot)", &cpu_y),
            ("den_inv(rot)", &cpu_den_inv),
            ("x_zinv", &cpu_x_zinv),
            ("s_final", &cpu_s_final),
        ];
        let gpu_labels_and_offsets = [
            ("den1", 3),
            ("den2", 4),
            ("z_inv", 5),
            ("ix0", 6),
            ("iy0", 7),
            ("ench_den", 8),
            ("t_zinv", 9),
            ("x(rot)", 10),
            ("y(rot)", 11),
            ("den_inv(rot)", 12),
            ("x_zinv", 13),
            ("s_final", 17),
        ];
        eprintln!("CPU rotate={} neg_y={}", cpu_rotate, cpu_neg_y);
        let mut first_mismatch = None;
        for (i, ((name, cpu_val), (_, gpu_offset))) in
            vals.iter().zip(gpu_labels_and_offsets.iter()).enumerate()
        {
            let cpu_bytes = limbs_to_bytes_v2(cpu_val);
            let gpu_bytes = &diag[gpu_offset * 32..(gpu_offset + 1) * 32];
            let matched = cpu_bytes == gpu_bytes;
            eprintln!(
                "CPU {:>15}: {} {}",
                name,
                hex::encode(&cpu_bytes),
                if matched { "✓" } else { "MISMATCH" }
            );
            if !matched && first_mismatch.is_none() {
                first_mismatch = Some(i);
            }
        }
        if let Some(idx) = first_mismatch {
            eprintln!("*** First mismatch at step: {} ***", vals[idx].0);
        }

        // Compare fe_sq vs fe_mul on GPU
        let gpu_u2_sq_via_sq = &diag[612..644];
        let gpu_u2_sq_via_mul = &diag[644..676];
        eprintln!("GPU u2^2 (fe_sq): {}", hex::encode(gpu_u2_sq_via_sq));
        eprintln!("GPU u2^2 (fe_mul): {}", hex::encode(gpu_u2_sq_via_mul));
        eprintln!("CPU u2^2 (fe_sq): {}", hex::encode(cpu_u2_sq_bytes));
        if gpu_u2_sq_via_sq != gpu_u2_sq_via_mul {
            eprintln!("*** fe_sq and fe_mul DISAGREE on GPU! fe_sq is buggy ***");
        }
        if gpu_u2_sq_via_sq != &cpu_u2_sq_bytes {
            eprintln!("*** GPU fe_sq disagrees with CPU fe_sq ***");
        }
        if gpu_u2_sq_via_mul != &cpu_u2_sq_bytes {
            eprintln!("*** GPU fe_mul(u2,u2) disagrees with CPU fe_sq(u2) ***");
        }

        // Compare compressed output
        let gpu_compressed = &diag[18 * 32..19 * 32];
        let cpu_compressed = point.compress();
        eprintln!(
            "CPU    compressed: {}",
            hex::encode(cpu_compressed.as_bytes())
        );

        // Try to decompress GPU result
        use curve25519_dalek::ristretto::CompressedRistretto;
        let gpu_cr = CompressedRistretto::from_slice(gpu_compressed).unwrap();
        match gpu_cr.decompress() {
            Some(gpu_point) => {
                eprintln!("GPU compressed IS a valid ristretto point");
                if gpu_point == point {
                    eprintln!(
                        "GPU point == original point (same group element, different encoding?!)"
                    );
                } else {
                    eprintln!("GPU point != original point (WRONG point!)");
                    // Check what point it is
                    let gpu_recompress = gpu_point.compress();
                    eprintln!(
                        "GPU point recompressed: {}",
                        hex::encode(gpu_recompress.as_bytes())
                    );
                }
            }
            None => {
                eprintln!("GPU compressed is NOT a valid ristretto point!");
            }
        }

        // Also check: does the identity compress correctly?
        let identity_scalar = Scalar::ZERO;
        let identity_point = &identity_scalar * RISTRETTO_BASEPOINT_TABLE;
        let identity_compressed = identity_point.compress();
        eprintln!(
            "Identity compressed: {}",
            hex::encode(identity_compressed.as_bytes())
        );

        // Also try the basepoint
        let bp_compressed = RISTRETTO_BASEPOINT_POINT.compress();
        eprintln!(
            "Basepoint compressed: {}",
            hex::encode(bp_compressed.as_bytes())
        );

        // Extract basepoint coords and run our algorithm
        let bp_coords = extract_point_coords(&RISTRETTO_BASEPOINT_POINT);
        let bp_x: [u64; 5] = [
            bp_coords[0],
            bp_coords[1],
            bp_coords[2],
            bp_coords[3],
            bp_coords[4],
        ];
        let bp_y: [u64; 5] = [
            bp_coords[5],
            bp_coords[6],
            bp_coords[7],
            bp_coords[8],
            bp_coords[9],
        ];
        let bp_z: [u64; 5] = [
            bp_coords[10],
            bp_coords[11],
            bp_coords[12],
            bp_coords[13],
            bp_coords[14],
        ];
        // Check if Z is 1 (the basepoint might be in affine form)
        eprintln!(
            "Basepoint Z bytes: {}",
            hex::encode(limbs_to_bytes_v2(&bp_z))
        );
        // Check if the basepoint is in affine form: Z should be [1,0,0,0,0]
        eprintln!("Basepoint Z limbs: {:?}", bp_z);
        eprintln!("Basepoint X limbs: {:?}", bp_x);
        eprintln!("Basepoint Y limbs: {:?}", bp_y);

        assert_eq!(
            gpu_compressed,
            cpu_compressed.as_bytes(),
            "Ristretto compression mismatch"
        );
    }

    /// CPU-only test: verify coordinate layout and ristretto encode correctness
    /// This doesn't require CUDA and directly tests our field arithmetic against dalek
    #[test]
    fn test_cpu_ristretto_encode_matches_dalek() {
        use curve25519_dalek::constants::ED25519_BASEPOINT_TABLE;
        use curve25519_dalek::ristretto::CompressedRistretto;

        // ===== Helper functions =====
        fn fe_add(f: &[u64; 5], g: &[u64; 5]) -> [u64; 5] {
            [
                f[0] + g[0],
                f[1] + g[1],
                f[2] + g[2],
                f[3] + g[3],
                f[4] + g[4],
            ]
        }
        fn fe_sub(f: &[u64; 5], g: &[u64; 5]) -> [u64; 5] {
            let mask: u64 = (1u64 << 51) - 1;
            [
                (f[0] + 2 * ((1u64 << 51) - 19)) - g[0],
                (f[1] + 2 * mask) - g[1],
                (f[2] + 2 * mask) - g[2],
                (f[3] + 2 * mask) - g[3],
                (f[4] + 2 * mask) - g[4],
            ]
        }
        fn fe_mul(f: &[u64; 5], g: &[u64; 5]) -> [u64; 5] {
            let (f0, f1, f2, f3, f4) = (f[0], f[1], f[2], f[3], f[4]);
            let (g0, g1, g2, g3, g4) = (g[0], g[1], g[2], g[3], g[4]);
            let (g1_19, g2_19, g3_19, g4_19) = (g1 * 19, g2 * 19, g3 * 19, g4 * 19);
            let h0: u128 = f0 as u128 * g0 as u128
                + f1 as u128 * g4_19 as u128
                + f2 as u128 * g3_19 as u128
                + f3 as u128 * g2_19 as u128
                + f4 as u128 * g1_19 as u128;
            let h1: u128 = f0 as u128 * g1 as u128
                + f1 as u128 * g0 as u128
                + f2 as u128 * g4_19 as u128
                + f3 as u128 * g3_19 as u128
                + f4 as u128 * g2_19 as u128;
            let h2: u128 = f0 as u128 * g2 as u128
                + f1 as u128 * g1 as u128
                + f2 as u128 * g0 as u128
                + f3 as u128 * g4_19 as u128
                + f4 as u128 * g3_19 as u128;
            let h3: u128 = f0 as u128 * g3 as u128
                + f1 as u128 * g2 as u128
                + f2 as u128 * g1 as u128
                + f3 as u128 * g0 as u128
                + f4 as u128 * g4_19 as u128;
            let h4: u128 = f0 as u128 * g4 as u128
                + f1 as u128 * g3 as u128
                + f2 as u128 * g2 as u128
                + f3 as u128 * g1 as u128
                + f4 as u128 * g0 as u128;
            let mask: u64 = (1u64 << 51) - 1;
            let c = (h0 >> 51) as u64;
            let v0 = h0 as u64 & mask;
            let h1 = h1 + c as u128;
            let c = (h1 >> 51) as u64;
            let v1 = h1 as u64 & mask;
            let h2 = h2 + c as u128;
            let c = (h2 >> 51) as u64;
            let v2 = h2 as u64 & mask;
            let h3 = h3 + c as u128;
            let c = (h3 >> 51) as u64;
            let v3 = h3 as u64 & mask;
            let h4 = h4 + c as u128;
            let c = (h4 >> 51) as u64;
            let v4 = h4 as u64 & mask;
            let mut r = [v0 + c * 19, v1, v2, v3, v4];
            let c2 = r[0] >> 51;
            r[0] &= mask;
            r[1] += c2;
            r
        }
        fn fe_sq(f: &[u64; 5]) -> [u64; 5] {
            let (f0, f1, f2, f3, f4) = (f[0], f[1], f[2], f[3], f[4]);
            let f0_2 = f0 * 2;
            let f1_2 = f1 * 2;
            let h0: u128 = f0 as u128 * f0 as u128
                + (f1 * 38) as u128 * f4 as u128
                + (f2 * 38) as u128 * f3 as u128;
            let h1: u128 = f0_2 as u128 * f1 as u128
                + (f2 * 38) as u128 * f4 as u128
                + (f3 * 19) as u128 * f3 as u128;
            let h2: u128 = f0_2 as u128 * f2 as u128
                + f1 as u128 * f1 as u128
                + (f3 * 38) as u128 * f4 as u128;
            let h3: u128 = f0_2 as u128 * f3 as u128
                + f1_2 as u128 * f2 as u128
                + (f4 * 19) as u128 * f4 as u128;
            let h4: u128 =
                f0_2 as u128 * f4 as u128 + f1_2 as u128 * f3 as u128 + f2 as u128 * f2 as u128;
            let mask: u64 = (1u64 << 51) - 1;
            let c = (h0 >> 51) as u64;
            let v0 = h0 as u64 & mask;
            let h1 = h1 + c as u128;
            let c = (h1 >> 51) as u64;
            let v1 = h1 as u64 & mask;
            let h2 = h2 + c as u128;
            let c = (h2 >> 51) as u64;
            let v2 = h2 as u64 & mask;
            let h3 = h3 + c as u128;
            let c = (h3 >> 51) as u64;
            let v3 = h3 as u64 & mask;
            let h4 = h4 + c as u128;
            let c = (h4 >> 51) as u64;
            let v4 = h4 as u64 & mask;
            let mut r = [v0 + c * 19, v1, v2, v3, v4];
            let c2 = r[0] >> 51;
            r[0] &= mask;
            r[1] += c2;
            r
        }
        fn fe_tobytes(v: &[u64; 5]) -> [u8; 32] {
            let mut h = *v;
            let mask: u64 = (1u64 << 51) - 1;
            // carry
            let mut c: u64;
            c = h[0] >> 51;
            h[0] &= mask;
            h[1] += c;
            c = h[1] >> 51;
            h[1] &= mask;
            h[2] += c;
            c = h[2] >> 51;
            h[2] &= mask;
            h[3] += c;
            c = h[3] >> 51;
            h[3] &= mask;
            h[4] += c;
            c = h[4] >> 51;
            h[4] &= mask;
            h[0] += c * 19;
            c = h[0] >> 51;
            h[0] &= mask;
            h[1] += c;
            // reduce
            let mut q = (h[0] + 19) >> 51;
            q = (h[1] + q) >> 51;
            q = (h[2] + q) >> 51;
            q = (h[3] + q) >> 51;
            q = (h[4] + q) >> 51;
            h[0] += q * 19;
            c = h[0] >> 51;
            h[0] &= mask;
            h[1] += c;
            c = h[1] >> 51;
            h[1] &= mask;
            h[2] += c;
            c = h[2] >> 51;
            h[2] &= mask;
            h[3] += c;
            c = h[3] >> 51;
            h[3] &= mask;
            h[4] += c;
            h[4] &= mask;
            // pack
            let lo0 = h[0] | (h[1] << 51);
            let lo1 = (h[1] >> 13) | (h[2] << 38);
            let lo2 = (h[2] >> 26) | (h[3] << 25);
            let lo3 = (h[3] >> 39) | (h[4] << 12);
            let mut out = [0u8; 32];
            out[0..8].copy_from_slice(&lo0.to_le_bytes());
            out[8..16].copy_from_slice(&lo1.to_le_bytes());
            out[16..24].copy_from_slice(&lo2.to_le_bytes());
            out[24..32].copy_from_slice(&lo3.to_le_bytes());
            out
        }
        fn fe_frombytes(bytes: &[u8; 32]) -> [u64; 5] {
            let lo0 = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
            let lo1 = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
            let lo2 = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
            let lo3 = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
            let mask: u64 = (1u64 << 51) - 1;
            [
                lo0 & mask,
                ((lo0 >> 51) | (lo1 << 13)) & mask,
                ((lo1 >> 38) | (lo2 << 26)) & mask,
                ((lo2 >> 25) | (lo3 << 39)) & mask,
                (lo3 >> 12) & mask,
            ]
        }
        fn fe_neg(f: &[u64; 5]) -> [u64; 5] {
            fe_sub(&[0; 5], f)
        }
        fn fe_isneg(f: &[u64; 5]) -> bool {
            fe_tobytes(f)[0] & 1 == 1
        }
        fn fe_equal(f: &[u64; 5], g: &[u64; 5]) -> bool {
            fe_tobytes(f) == fe_tobytes(g)
        }
        fn fe_abs(f: &[u64; 5]) -> [u64; 5] {
            if fe_isneg(f) {
                fe_neg(f)
            } else {
                *f
            }
        }
        fn fe_sq_n(f: &[u64; 5], n: usize) -> [u64; 5] {
            let mut h = fe_sq(f);
            for _ in 1..n {
                h = fe_sq(&h);
            }
            h
        }
        fn fe_pow22523(f: &[u64; 5]) -> [u64; 5] {
            let mut t0 = fe_sq(f);
            let mut t1 = fe_sq_n(&t0, 2);
            t1 = fe_mul(f, &t1);
            t0 = fe_mul(&t0, &t1);
            t0 = fe_sq(&t0);
            t0 = fe_mul(&t1, &t0);
            t1 = fe_sq_n(&t0, 5);
            t0 = fe_mul(&t1, &t0);
            t1 = fe_sq_n(&t0, 10);
            t1 = fe_mul(&t1, &t0);
            let mut t2 = fe_sq_n(&t1, 20);
            t1 = fe_mul(&t2, &t1);
            t1 = fe_sq_n(&t1, 10);
            t0 = fe_mul(&t1, &t0);
            t1 = fe_sq_n(&t0, 50);
            t1 = fe_mul(&t1, &t0);
            t2 = fe_sq_n(&t1, 100);
            t1 = fe_mul(&t2, &t1);
            t1 = fe_sq_n(&t1, 50);
            t0 = fe_mul(&t1, &t0);
            t0 = fe_sq_n(&t0, 2);
            fe_mul(&t0, f)
        }

        let sqrt_m1: [u64; 5] = [
            1718705420411056,
            234908883556509,
            2233514472574048,
            2117202627021982,
            765476049583133,
        ];
        let invsqrt_a_minus_d: [u64; 5] = [
            278908739862762,
            821645201101625,
            8113234426968,
            1777959178193151,
            2118520810568447,
        ];

        // ===== Step 0: Verify field arithmetic basics =====
        // SQRT_M1^2 should equal -1 (= p-1)
        let sqrt_m1_sq = fe_sq(&sqrt_m1);
        let sqrt_m1_sq_bytes = fe_tobytes(&sqrt_m1_sq);
        // p-1 in LE bytes
        let p_minus_1: [u8; 32] = [
            0xec, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0x7f,
        ];
        assert_eq!(
            sqrt_m1_sq_bytes, p_minus_1,
            "SQRT_M1^2 != -1 -- field arithmetic is broken!"
        );
        eprintln!("SQRT_M1^2 == -1 (p-1): PASS");

        // ===== Step 1: Verify coordinate layout =====
        let scalar = Scalar::from(42u64);
        let ristretto_point: RistrettoPoint = &scalar * RISTRETTO_BASEPOINT_TABLE;
        let edwards_point = &scalar * ED25519_BASEPOINT_TABLE;

        let compressed_edwards = edwards_point.compress();
        let mut y_bytes = *compressed_edwards.as_bytes();
        let _x_sign = y_bytes[31] >> 7;
        y_bytes[31] &= 0x7F; // clear sign bit

        // Extract coords from RistrettoPoint
        assert_eq!(std::mem::size_of::<RistrettoPoint>(), 160);
        let coords: [u64; 20] = unsafe { std::mem::transmute(ristretto_point) };
        let fields: [[u64; 5]; 4] = [
            [coords[0], coords[1], coords[2], coords[3], coords[4]],
            [coords[5], coords[6], coords[7], coords[8], coords[9]],
            [coords[10], coords[11], coords[12], coords[13], coords[14]],
            [coords[15], coords[16], coords[17], coords[18], coords[19]],
        ];

        // Test: which field * which field == y_bytes when serialized?
        // If layout is [X, Y, Z, T], then y = Y/Z, so y*Z = Y, meaning
        // fe_frombytes(y_bytes) * fields[2] should equal fields[1]
        let y_from_edwards = fe_frombytes(&y_bytes);

        let field_names = ["f0(X?)", "f1(Y?)", "f2(Z?)", "f3(T?)"];
        let mut found_y = false;
        let mut y_idx = 1usize;
        let mut z_idx = 2usize;

        for zi in 0..4 {
            for yi in 0..4 {
                if zi == yi {
                    continue;
                }
                let product = fe_mul(&y_from_edwards, &fields[zi]);
                if fe_equal(&product, &fields[yi]) {
                    eprintln!(
                        "Layout check: y_edwards * {} == {} => Y={}, Z={}",
                        field_names[zi], field_names[yi], field_names[yi], field_names[zi]
                    );
                    y_idx = yi;
                    z_idx = zi;
                    found_y = true;
                }
            }
        }
        assert!(
            found_y,
            "Could not find Y,Z fields matching CompressedEdwardsY!"
        );

        // Now determine X and T from Edwards invariant T*Z = X*Y
        let mut x_idx = usize::MAX;
        let mut t_idx = usize::MAX;
        for xi in 0..4 {
            if xi == y_idx || xi == z_idx {
                continue;
            }
            for ti in 0..4 {
                if ti == y_idx || ti == z_idx || ti == xi {
                    continue;
                }
                let xy = fe_mul(&fields[xi], &fields[y_idx]);
                let tz = fe_mul(&fields[ti], &fields[z_idx]);
                if fe_equal(&xy, &tz) {
                    // Also verify x sign matches
                    // x = X/Z, x is negative (odd) if x_sign == 1
                    // We check: is X*Z^{-1} negative? We can check X*Z_inv but
                    // we don't have inversion. Instead check: for x_sign to match,
                    // if we had x, isneg(x) should equal x_sign.
                    eprintln!(
                        "Edwards invariant: {}*{} == {}*{} => X={}, T={}",
                        field_names[xi],
                        field_names[y_idx],
                        field_names[ti],
                        field_names[z_idx],
                        field_names[xi],
                        field_names[ti]
                    );
                    x_idx = xi;
                    t_idx = ti;
                }
            }
        }
        assert!(
            x_idx != usize::MAX,
            "Could not determine X,T from Edwards invariant"
        );

        eprintln!(
            "Determined layout: X=fields[{}], Y=fields[{}], Z=fields[{}], T=fields[{}]",
            x_idx, y_idx, z_idx, t_idx
        );

        let assumed_correct = x_idx == 0 && y_idx == 1 && z_idx == 2 && t_idx == 3;
        if !assumed_correct {
            eprintln!("*** LAYOUT DIFFERS FROM ASSUMED [X,Y,Z,T] = [f0,f1,f2,f3]! ***");
        } else {
            eprintln!("Layout matches assumed [X,Y,Z,T] = [f0,f1,f2,f3]");
        }

        let x_limbs = fields[x_idx];
        let y_limbs = fields[y_idx];
        let z_limbs = fields[z_idx];
        let t_limbs = fields[t_idx];

        // ===== Step 2: Run full ristretto encode using CORRECT field ordering =====
        // u1 = (Z+Y)*(Z-Y)
        let t0 = fe_add(&z_limbs, &y_limbs);
        let t1 = fe_sub(&z_limbs, &y_limbs);
        let u1 = fe_mul(&t0, &t1);

        // u2 = X*Y
        let u2 = fe_mul(&x_limbs, &y_limbs);

        // inv = sqrt_ratio(1, u1*u2^2)
        let u2_sq = fe_sq(&u2);
        let u1_u2sq = fe_mul(&u1, &u2_sq);
        let one = [1u64, 0, 0, 0, 0];

        // sqrt_ratio_m1(u=1, v=u1_u2sq)
        let v = u1_u2sq;
        let v_sq = fe_sq(&v);
        let v3 = fe_mul(&v_sq, &v);
        let v3_sq = fe_sq(&v3);
        let v7 = fe_mul(&v3_sq, &v);
        let uv7 = fe_mul(&one, &v7); // u=1, so uv7 = v7
        let pow = fe_pow22523(&uv7);
        let mut r = fe_mul(&pow, &v3);
        r = fe_mul(&r, &one); // u=1

        let r_sq = fe_sq(&r);
        let check = fe_mul(&v, &r_sq);
        let neg_u = fe_neg(&one);
        let neg_u_i = fe_mul(&neg_u, &sqrt_m1);
        let correct_sign = fe_equal(&check, &one);
        let flipped_sign = fe_equal(&check, &neg_u);
        let flipped_sign_i = fe_equal(&check, &neg_u_i);

        let r_prime = fe_mul(&r, &sqrt_m1);
        if flipped_sign || flipped_sign_i {
            r = r_prime;
        }
        let inv = fe_abs(&r);
        let was_sq = correct_sign || flipped_sign;

        eprintln!(
            "sqrt_ratio: correct={} flipped={} flipped_i={} was_sq={}",
            correct_sign, flipped_sign, flipped_sign_i, was_sq
        );

        // den1 = inv * u1, den2 = inv * u2
        let den1 = fe_mul(&inv, &u1);
        let den2 = fe_mul(&inv, &u2);

        // z_inv = den1 * den2 * T
        let z_inv = fe_mul(&fe_mul(&den1, &den2), &t_limbs);

        // ix0 = X * SQRT_M1, iy0 = Y * SQRT_M1
        let ix0 = fe_mul(&x_limbs, &sqrt_m1);
        let iy0 = fe_mul(&y_limbs, &sqrt_m1);

        // ench_den = den1 * INVSQRT_A_MINUS_D
        let ench_den = fe_mul(&den1, &invsqrt_a_minus_d);

        // rotate = isneg(T * z_inv)
        let t_zinv = fe_mul(&t_limbs, &z_inv);
        let rotate = fe_isneg(&t_zinv);

        // conditional rotation
        let mut x = x_limbs;
        if rotate {
            x = iy0;
        }
        let mut y = y_limbs;
        if rotate {
            y = ix0;
        }
        let mut den_inv = den2;
        if rotate {
            den_inv = ench_den;
        }

        // neg_y = isneg(x * z_inv)
        let x_zinv = fe_mul(&x, &z_inv);
        let neg_y = fe_isneg(&x_zinv);

        let mut y_final = y;
        if neg_y {
            y_final = fe_neg(&y);
        }

        // s = |den_inv * (Z - y_final)|
        let z_minus_y = fe_sub(&z_limbs, &y_final);
        let s = fe_mul(&den_inv, &z_minus_y);
        let s_final = fe_abs(&s);

        let our_compressed = fe_tobytes(&s_final);
        let dalek_compressed = ristretto_point.compress();

        eprintln!("rotate={} neg_y={}", rotate, neg_y);
        eprintln!("Our  compressed: {}", hex::encode(our_compressed));
        eprintln!(
            "Dalek compressed: {}",
            hex::encode(dalek_compressed.as_bytes())
        );

        if our_compressed != *dalek_compressed.as_bytes() {
            // Check if valid
            let our_cr = CompressedRistretto::from_slice(&our_compressed).unwrap();
            match our_cr.decompress() {
                Some(_) => eprintln!("Our result IS a valid ristretto point (but wrong one)"),
                None => eprintln!("Our result is NOT a valid ristretto point"),
            }

            // Print intermediates for debugging
            eprintln!("u1:      {}", hex::encode(fe_tobytes(&u1)));
            eprintln!("u2:      {}", hex::encode(fe_tobytes(&u2)));
            eprintln!("inv:     {}", hex::encode(fe_tobytes(&inv)));
            eprintln!("den1:    {}", hex::encode(fe_tobytes(&den1)));
            eprintln!("den2:    {}", hex::encode(fe_tobytes(&den2)));
            eprintln!("z_inv:   {}", hex::encode(fe_tobytes(&z_inv)));
            eprintln!("t_zinv:  {}", hex::encode(fe_tobytes(&t_zinv)));
            eprintln!("x(rot):  {}", hex::encode(fe_tobytes(&x)));
            eprintln!("y(rot):  {}", hex::encode(fe_tobytes(&y)));
            eprintln!("den_inv: {}", hex::encode(fe_tobytes(&den_inv)));
            eprintln!("x_zinv:  {}", hex::encode(fe_tobytes(&x_zinv)));
            eprintln!("y_final: {}", hex::encode(fe_tobytes(&y_final)));
            eprintln!("Z-y:     {}", hex::encode(fe_tobytes(&z_minus_y)));
            eprintln!("s:       {}", hex::encode(fe_tobytes(&s)));
            eprintln!("s_final: {}", hex::encode(fe_tobytes(&s_final)));
        }

        assert_eq!(
            our_compressed,
            *dalek_compressed.as_bytes(),
            "CPU ristretto encode doesn't match dalek"
        );
    }

    // ========================================================================
    // U512 arithmetic tests
    // ========================================================================

    #[test]
    fn test_u512_mul_small() {
        let mut a = [0u8; 64];
        a[63] = 58;
        u512_mul_small(&mut a, 58);
        // 58 * 58 = 3364 = 0x0D24
        assert_eq!(a[62], 0x0D);
        assert_eq!(a[63], 0x24);
    }

    #[test]
    fn test_u512_add_small() {
        let mut a = [0u8; 64];
        a[63] = 255;
        u512_add_small(&mut a, 1);
        assert_eq!(a[63], 0);
        assert_eq!(a[62], 1);
    }

    #[test]
    fn test_u512_add() {
        let mut a = [0u8; 64];
        let mut b = [0u8; 64];
        a[63] = 200;
        b[63] = 100;
        let result = u512_add(&a, &b);
        // 200 + 100 = 300 = 0x012C
        assert_eq!(result[62], 1);
        assert_eq!(result[63], 0x2C);
    }

    #[test]
    fn test_u512_cmp() {
        let a = [0u8; 64];
        let b = [0u8; 64];
        assert_eq!(u512_cmp(&a, &b), std::cmp::Ordering::Equal);

        let mut c = [0u8; 64];
        c[0] = 1;
        assert_eq!(u512_cmp(&c, &a), std::cmp::Ordering::Greater);
        assert_eq!(u512_cmp(&a, &c), std::cmp::Ordering::Less);
    }

    // ========================================================================
    // Range computation tests
    // ========================================================================

    #[test]
    fn test_base58_char_values() {
        // '1' is digit 0
        assert_eq!(base58_char_values(b'1'), vec![0]);
        // 'a' -> both 'A' (index 9) and 'a' (index 33)
        let vals = base58_char_values(b'a');
        assert!(vals.contains(&9)); // 'A' is at index 9
        assert!(vals.contains(&33)); // 'a' is at index 33
                                     // '2' is digit 1
        assert_eq!(base58_char_values(b'2'), vec![1]);
    }

    #[test]
    fn test_compute_prefix_ranges_basic() {
        // A single-char prefix should produce some ranges
        let ranges = compute_prefix_ranges("b");
        assert!(
            !ranges.is_empty(),
            "expected non-empty ranges for prefix 'b'"
        );
        // Ranges should be sorted and non-overlapping
        for i in 1..ranges.len() {
            assert!(
                u512_cmp(&ranges[i - 1].1, &ranges[i].0) != std::cmp::Ordering::Greater,
                "ranges not sorted/non-overlapping at index {}",
                i
            );
        }
    }

    #[test]
    fn test_prefix_ranges_match_bs58() {
        // Generate random keys and verify that range-based matching agrees with bs58
        let mut rng = rand::thread_rng();
        let prefix = "te";
        let ranges = compute_prefix_ranges(prefix);

        for _ in 0..1000 {
            let spend_priv = random_scalar(&mut rng);
            let view_priv = random_scalar(&mut rng);
            let spend_pub = (&spend_priv * RISTRETTO_BASEPOINT_TABLE).compress();
            let view_pub = (&view_priv * RISTRETTO_BASEPOINT_TABLE).compress();

            let mut combined = [0u8; 64];
            combined[..32].copy_from_slice(spend_pub.as_bytes());
            combined[32..].copy_from_slice(view_pub.as_bytes());

            let address = bs58::encode(&combined).into_string();
            let bs58_match = address.to_ascii_lowercase().starts_with(prefix);

            // Range-based check
            let range_match = ranges.iter().any(|(lo, hi)| {
                u512_cmp(&combined, lo) != std::cmp::Ordering::Less
                    && u512_cmp(&combined, hi) == std::cmp::Ordering::Less
            });
            let range_match_split =
                prefix_match_ranges_split(spend_pub.as_bytes(), view_pub.as_bytes(), &ranges);

            assert_eq!(
                range_match, bs58_match,
                "range vs bs58 disagree for addr={} (range={}, bs58={})",
                address, range_match, bs58_match
            );
            assert_eq!(
                range_match_split, range_match,
                "split range matcher disagrees for addr={} (split={}, classic={})",
                address, range_match_split, range_match
            );
        }
    }

    #[test]
    fn test_compute_suffix_targets_basic() {
        let (modulus, targets) = compute_suffix_targets("ab");
        assert_eq!(modulus, 58 * 58);
        assert!(!targets.is_empty());
    }

    #[test]
    fn test_suffix_targets_match_bs58() {
        let mut rng = rand::thread_rng();
        let suffix = "z";
        let (modulus, targets) = compute_suffix_targets(suffix);

        for _ in 0..1000 {
            let spend_priv = random_scalar(&mut rng);
            let view_priv = random_scalar(&mut rng);
            let spend_pub = (&spend_priv * RISTRETTO_BASEPOINT_TABLE).compress();
            let view_pub = (&view_priv * RISTRETTO_BASEPOINT_TABLE).compress();

            let mut combined = [0u8; 64];
            combined[..32].copy_from_slice(spend_pub.as_bytes());
            combined[32..].copy_from_slice(view_pub.as_bytes());

            let address = bs58::encode(&combined).into_string();
            let bs58_match = address.to_ascii_lowercase().ends_with(suffix);

            // Modular arithmetic check
            let mut mod_val: u64 = 0;
            for &byte in combined.iter() {
                mod_val = (mod_val * 256 + byte as u64) % modulus;
            }
            let mod_match = targets.contains(&mod_val);

            assert_eq!(
                mod_match, bs58_match,
                "suffix mod vs bs58 disagree for addr={} (mod={}, bs58={})",
                address, mod_match, bs58_match
            );
        }
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn test_gpu_range_match_agrees_with_cpu() {
        // Upload generator table
        let gen_table = build_gen_table();
        let rc = unsafe { cuda_init_gen_table(gen_table.as_ptr(), TABLE_BITS as i32) };
        assert_eq!(rc, 0);

        let start_scalar = Scalar::from(77u64);
        let start_point: RistrettoPoint = &start_scalar * RISTRETTO_BASEPOINT_TABLE;
        let coords = extract_point_coords(&start_point);

        let view_priv = Scalar::from(99u64);
        let view_pub = (&view_priv * RISTRETTO_BASEPOINT_TABLE).compress();
        let view_pub_bytes = view_pub.to_bytes();

        let prefix = "b";
        let prefix_ranges = compute_prefix_ranges(prefix);
        let prefix_ranges_flat: Vec<u8> = prefix_ranges
            .iter()
            .flat_map(|(lo, hi)| lo.iter().chain(hi.iter()).copied())
            .collect();

        let prefix_c = CString::new(prefix).unwrap();
        let suffix_c = CString::new("").unwrap();

        let batch_size = 4096;
        let handle = unsafe {
            cuda_worker_create(
                batch_size as i32,
                prefix_c.as_ptr(),
                prefix.len() as i32,
                suffix_c.as_ptr(),
                0,
                1, // full GPU mode
            )
        };
        assert!(!handle.is_null());

        // Set ranges
        let rc = unsafe {
            cuda_worker_set_ranges(
                handle,
                if prefix_ranges_flat.is_empty() {
                    std::ptr::null()
                } else {
                    prefix_ranges_flat.as_ptr()
                },
                prefix_ranges.len() as i32,
                0, // no suffix
                std::ptr::null(),
                0,
                0,
                0,
            )
        };
        assert_eq!(rc, 0, "cuda_worker_set_ranges failed");

        let rc = unsafe {
            cuda_worker_submit_v2(
                handle,
                coords[0..5].as_ptr(),
                coords[5..10].as_ptr(),
                coords[10..15].as_ptr(),
                coords[15..20].as_ptr(),
                view_pub_bytes.as_ptr(),
                batch_size as i32,
            )
        };
        assert_eq!(rc, 0, "cuda_worker_submit_v2 failed");

        let flags =
            unsafe { std::slice::from_raw_parts(cuda_worker_get_flags(handle), batch_size) };

        // Verify every GPU match/non-match against CPU
        let mut match_count = 0;
        for i in 0..batch_size {
            let priv_i = start_scalar + Scalar::from(i as u64);
            let pub_i = (&priv_i * RISTRETTO_BASEPOINT_TABLE).compress();

            let mut combined = [0u8; 64];
            combined[..32].copy_from_slice(pub_i.as_bytes());
            combined[32..].copy_from_slice(&view_pub_bytes);

            let address = bs58::encode(&combined).into_string();
            let cpu_match = address.to_ascii_lowercase().starts_with(prefix);

            let gpu_match = flags[i] != 0;

            assert_eq!(
                gpu_match, cpu_match,
                "GPU/CPU range match disagree at index {}: gpu={} cpu={} addr={}",
                i, gpu_match, cpu_match, address
            );

            if gpu_match {
                match_count += 1;
            }
        }

        assert!(
            match_count > 0,
            "no matches found in 4096 candidates with prefix 'b'"
        );

        unsafe {
            cuda_worker_destroy(handle);
        }
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn test_gpu_suffix_range_match() {
        // Upload generator table
        let gen_table = build_gen_table();
        let rc = unsafe { cuda_init_gen_table(gen_table.as_ptr(), TABLE_BITS as i32) };
        assert_eq!(rc, 0);

        let start_scalar = Scalar::from(42u64);
        let start_point: RistrettoPoint = &start_scalar * RISTRETTO_BASEPOINT_TABLE;
        let coords = extract_point_coords(&start_point);

        let view_priv = Scalar::from(123u64);
        let view_pub = (&view_priv * RISTRETTO_BASEPOINT_TABLE).compress();
        let view_pub_bytes = view_pub.to_bytes();

        // Test multiple suffixes to ensure at least one produces matches.
        // Due to gcd(2^256, 58) = 2, single-char suffixes may be unreachable
        // for certain view keys (combined mod 58 parity constraint).
        let suffixes = ["a", "b", "2"];
        let batch_size = 4096;
        let mut total_match_count = 0;

        for suffix in &suffixes {
            let (suffix_modulus, suffix_targets) = compute_suffix_targets(suffix);

            let prefix_c = CString::new("").unwrap();
            let suffix_c = CString::new(*suffix).unwrap();

            let handle = unsafe {
                cuda_worker_create(
                    batch_size as i32,
                    prefix_c.as_ptr(),
                    0,
                    suffix_c.as_ptr(),
                    suffix.len() as i32,
                    1,
                )
            };
            assert!(!handle.is_null());

            let shift_mod = suffix_shift_mod(suffix_modulus);
            let view_offset = suffix_view_offset(&view_pub_bytes, suffix_modulus);
            let rc = unsafe {
                cuda_worker_set_ranges(
                    handle,
                    std::ptr::null(),
                    0, // no prefix
                    suffix_modulus,
                    suffix_targets.as_ptr(),
                    suffix_targets.len() as i32,
                    shift_mod,
                    view_offset,
                )
            };
            assert_eq!(rc, 0, "cuda_worker_set_ranges failed");

            let rc = unsafe {
                cuda_worker_submit_v2(
                    handle,
                    coords[0..5].as_ptr(),
                    coords[5..10].as_ptr(),
                    coords[10..15].as_ptr(),
                    coords[15..20].as_ptr(),
                    view_pub_bytes.as_ptr(),
                    batch_size as i32,
                )
            };
            assert_eq!(rc, 0, "cuda_worker_submit_v2 failed");

            let flags =
                unsafe { std::slice::from_raw_parts(cuda_worker_get_flags(handle), batch_size) };

            let mut match_count = 0;
            for i in 0..batch_size {
                let priv_i = start_scalar + Scalar::from(i as u64);
                let pub_i = (&priv_i * RISTRETTO_BASEPOINT_TABLE).compress();

                let mut combined = [0u8; 64];
                combined[..32].copy_from_slice(pub_i.as_bytes());
                combined[32..].copy_from_slice(&view_pub_bytes);

                let address = bs58::encode(&combined).into_string();
                let cpu_match = address.to_ascii_lowercase().ends_with(*suffix);
                let gpu_match = flags[i] != 0;

                assert_eq!(
                    gpu_match, cpu_match,
                    "GPU/CPU suffix match disagree for suffix '{}' at index {}: gpu={} cpu={} addr={}",
                    suffix, i, gpu_match, cpu_match, address
                );

                if gpu_match {
                    match_count += 1;
                }
            }

            total_match_count += match_count;
            unsafe {
                cuda_worker_destroy(handle);
            }
        }

        // Across all tested suffixes, we should find some matches
        assert!(
            total_match_count > 0,
            "no suffix matches found across all tested suffixes"
        );
    }
}
