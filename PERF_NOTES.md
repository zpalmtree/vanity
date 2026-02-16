# Performance Notes (2026-02-15)

## Scope
This run focused on speed across:
- CPU backend hot loop
- CUDA vanity kernel prefix/suffix matching
- CUDA default worker tuning

## Baseline (before changes)
All measurements used:
- pattern: `--prefix zzzzzzzz`
- duration: `timeout 16s`
- GPU batch: `8388608` unless noted
- machine: RTX 3080 desktop (display-attached, non-exclusive)

### Throughput snapshots
- `--cuda -t 1`: `57,312,597 keys/s`
- `--cuda -t 2`: `58,994,488 keys/s`
- `--cuda -t 4`: `59,532,421 keys/s`
- CPU `-t 1`: `132,437 keys/s`
- CPU `-t 8`: `1,005,189 keys/s`

### Batch-size sweep (before)
- `batch=2,097,152`: `58,657,465 keys/s`
- `batch=4,194,304`: `55,113,890 keys/s`
- `batch=8,388,608`: `54,583,836 keys/s`
- `batch=16,777,216`: `56,762,673 keys/s`

## Changes implemented

### 1) CPU backend fast matcher
- Replaced per-key base58 encode/match with:
  - precomputed prefix ranges
  - binary-search range lookup on split bytes (`spend_pub || view_pub`)
  - modular suffix match with precomputed shift/view offset
- Kept legacy base58 loop only for prefixes starting with `'1'`.

### 2) CUDA kernel matcher rewrite
- Removed per-key `combined[64]` construction.
- Added split-byte comparator for `spend_pub + view_pub` vs range bounds.
- Replaced linear prefix range scan with binary search over sorted ranges.
- Replaced linear suffix target scan with binary search.
- Fixed suffix modular combine multiply to use 128-bit intermediate.

### 3) CUDA default worker tuning
- Default worker threads for CUDA mode changed to `2` (when `-t` not provided).
- Manual override via `-t` is unchanged.

## Validation
- `cargo test --features cuda -- --test-threads=1`
- Result: `28 passed, 0 failed`

## Compile telemetry (release, CUDA)
`vanity_kernel` ptxas changes:
- stack frame: `568 -> 496 bytes`
- spill stores: `648 -> 532 bytes`
- spill loads: `648 -> 532 bytes`

## Post-change benchmarks

### Throughput snapshots
- `--cuda -t 1`: `66,322,139 keys/s`
- `--cuda -t 2`: `67,987,334 keys/s`
- `--cuda -t 4`: `65,851,167 keys/s`
- CPU `-t 1`: `255,862 keys/s`
- CPU `-t 8`: `1,940,204 keys/s`

### Batch-size sweep (after, `-t 2`)
- `batch=2,097,152`: `60,934,166 keys/s`
- `batch=4,194,304`: `64,571,342 keys/s`
- `batch=8,388,608`: `76,709,206 keys/s`
- `batch=16,777,216`: `76,174,753 keys/s`

### Repeat runs (after, `batch=8,388,608`)
- `-t 1`: `71,689,188`, `74,363,532`, `73,679,729`
- `-t 2`: `78,742,778`, `70,527,020`, `75,007,801`
- `-t 4`: `63,799,573`, `71,820,992`, `73,071,742`

Observed medians:
- `-t 1`: `73,679,729`
- `-t 2`: `75,007,801`
- `-t 4`: `71,820,992`

## Notes on noise
Desktop GPU processes (Xorg/browser/Discord) were active during runs, so single-run throughput is noisy. Median-over-repeats was used to pick default guidance.

## 2026-02-15 strict interleaved A/B (baseline commit vs candidate)

### Why this run
Earlier short-window snapshots mixed peak and last-sample reporting and were noisy. This pass used a stricter methodology to reduce ordering and thermal bias.

### Method
- Baseline: commit `e66cf8f`
- Candidate: commit `cf94dfd`
- Interleaved pair order: alternating baseline-first / candidate-first
- Warmup: 12s per variant per mode
- Measured window: 30s
- Cooldown: 20s between runs
- Pairs: 4 per mode
- Fixed settings: `--cuda -t 2 --batch-size 8388608`
- Metrics captured per run: last / median / mean / max (from progress samples)
- Raw CSV: `/tmp/vanity_ab_strict.csv`

### Aggregate (measured pairs only)
- Prefix baseline avg median: `59,355,107/s`
- Prefix candidate avg median: `71,600,234/s`
- Prefix delta: `+12,245,127/s` (`+20.63%`)
- Suffix baseline avg median: `66,281,124/s`
- Suffix candidate avg median: `70,166,250/s`
- Suffix delta: `+3,885,126/s` (`+5.86%`)

### Pairwise deltas (median, candidate - baseline)
- Prefix pair 1: `+15.76%`
- Prefix pair 2: `+22.60%`
- Prefix pair 3: `+36.36%`
- Prefix pair 4: `+10.74%`
- Suffix pair 1: `+9.66%`
- Suffix pair 2: `+1.84%`
- Suffix pair 3: `+12.68%`
- Suffix pair 4: `+0.11%`

## 2026-02-15 additional CUDA optimization attempts

### Attempt A (rejected): per-thread bitmask output instead of per-key flags
- Idea: write one mask byte per `KEYS_PER_THREAD` to reduce D→H copy and host scan cost.
- Result (`/tmp/vanity_ab_mask.csv`, interleaved A/B, 3 pairs, 20s rounds):
  - Prefix avg median regressed (`70.2M -> 63.2M`, about `-10%`).
  - Suffix avg median improved (`63.1M -> 68.9M`, about `+9%`).
- Decision: rejected due prefix regression (not acceptable for mixed workloads).

### Attempt B (rejected): vanity kernel launch at 128 threads/block
- Idea: lower block size from `256` to `128` for scheduling/occupancy tradeoff.
- Result (`/tmp/vanity_ab_threads128_quick.csv`, interleaved A/B, 2 pairs, 15s rounds):
  - Prefix avg median regressed (`68.4M -> 64.1M`).
  - Suffix avg median regressed (`67.5M -> 58.2M`).
- Decision: rejected.

### Attempt C (accepted): remove redundant `cudaMemsetAsync` before kernels
- Change:
  - Removed pre-launch clear in `cuda_worker_submit_v2` (full GPU path).
  - Removed pre-launch clear in `cuda_worker_submit` (legacy path).
  - Rationale: both kernels write every output slot each launch.
- Result (`/tmp/vanity_ab_no_memset.csv`, interleaved A/B, 3 pairs, 20s rounds):
  - Prefix avg median: `69,383,952 -> 76,487,549` (`+10.24%`)
  - Suffix avg median: `70,583,312 -> 76,742,800` (`+8.73%`)
  - Pairwise medians:
    - Prefix: `+11.80%`, `-3.49%`, `+23.59%`
    - Suffix: `+3.07%`, `+5.67%`, `+18.83%`
- Decision: accepted.
