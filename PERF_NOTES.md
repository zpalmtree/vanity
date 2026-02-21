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

## 2026-02-16 further CUDA optimization attempts

### Attempt D (rejected): template-specialized kernels for prefix-only / suffix-only / both
- Idea: compile separate `vanity_kernel` variants with dead paths removed per mode.
- Result (quick screening, 24s windows):
  - Prefix regressed (`~62.1M -> ~59.2M` median).
  - Suffix was near-flat/slight up (`~58.6M -> ~59.0M` median).
  - ptxas showed much worse prefix-only compile metrics vs generic path (`~496B/532 spills` to `~1184B/1260 spills`).
- Decision: rejected due register-pressure/spill regression.

### Attempt E (rejected): use constant memory directly for generator table
- Idea: stop loading `d_gen_table` into shared memory each block; read table from constant memory directly.
- Result (quick screening, 24s windows):
  - Prefix regressed (`~80.6M -> ~72.3M` median).
  - Suffix regressed (`~73.9M -> ~71.1M` median).
- Decision: rejected.

### Attempt F (accepted): chunked suffix modulus + cached `view_pub` upload
- Changes:
  - Suffix path in `vanity_kernel` now accumulates `spend_mod` in 8-byte chunks:
    - `4` modular steps using precomputed `256^8 mod m` instead of `32` byte-steps.
  - Added `suffix_chunk_mul` plumbing from Rust setup into CUDA worker state.
  - Cached worker `view_pub` on host side and only upload to device when it changes (avoids redundant tiny H→D copy each submit).
- Strict interleaved A/B (`/tmp/vanity_ab_suffix_chunk.csv`):
  - Baseline: commit `096adb9`
  - Candidate: chunked suffix + cached `view_pub` upload
  - Warmup: 12s per variant per mode
  - Measured window: 24s
  - Cooldown: 15s
  - Pairs: 3 per mode
  - Interleaved order: alternating baseline-first / candidate-first
  - Fixed settings: `--cuda -t 2 --batch-size 8388608`
- Aggregate (measured-pair avg medians):
  - Prefix: `67,522,484 -> 71,980,387` (`+4,457,903`, `+6.60%`)
  - Suffix: `66,701,708 -> 70,026,123` (`+3,324,415`, `+4.98%`)
- Pairwise deltas (median, candidate - baseline):
  - Prefix: `+3.96%`, `+12.33%`, `+3.94%`
  - Suffix: `+3.73%`, `+2.60%`, `+8.69%`
- Decision: accepted.

## 2026-02-16 profiling harness + expanded rewrite attempts

### Benchmark harness update (accepted)
- Added `scripts/bench_cuda_ab.sh` for strict interleaved CUDA A/B runs.
- Harness features:
  - alternating baseline/candidate order per pair
  - warmup + measured windows with cooldown gaps
  - per-run median/mean/min/max capture from progress samples
  - raw CSV + per-run log file paths
  - mode filter (`--modes prefix,suffix`) for isolated validation

### Attempt G (rejected): full rewrite bundle
- Candidate bundle included:
  - split kernels for prefix-only / suffix-only / both
  - packed 64-bit prefix comparator
  - reciprocal-based byte-step suffix reduction
  - startup CUDA init/autotune plumbing
  - ptxas knob surface (`CUDA_LAUNCH_MIN_BLOCKS`, `CUDA_MAXRREGCOUNT`)
- Strict interleaved A/B (`/tmp/vanity_ab_do_them_all.csv`):
  - baseline: commit `0e1feb5`
  - candidate: full rewrite bundle
  - warmup `12s`, measure `24s`, cooldown `15s`, `3` pairs/mode
  - fixed settings: `--cuda -t 2 --batch-size 8388608`
- Aggregate (avg medians):
  - Prefix: `59,282,461 -> 54,040,291` (`-5,242,169`, `-8.84%`)
  - Suffix: `60,927,988 -> 58,668,803` (`-2,259,185`, `-3.71%`)
- Pairwise deltas:
  - Prefix: `-6.82%`, `-3.06%`, `-16.47%`
  - Suffix: `-3.65%`, `-4.64%`, `-2.86%`
- Decision: rejected.

### Attempt H (rejected): prefix-only specialized kernel with generic fallback
- Change:
  - kept baseline generic `vanity_kernel` for suffix/both
  - added dedicated prefix-only kernel + dispatch split
- Interleaved A/B (`/tmp/vanity_ab_prefix_only_specialized_clean.csv`, 2 pairs/mode):
  - Prefix avg medians: `63,094,584 -> 77,034,185` (`+22.10%`)
  - Suffix avg medians: `57,719,110 -> 56,398,068` (`-2.29%`)
- Suffix-only confirmation (`/tmp/vanity_suffix_confirm.csv`, 3 pairs):
  - Baseline avg median: `53,979,448`
  - Candidate avg median: `52,118,468`
  - Delta: `-1,860,980` (`-3.45%`)
  - High variance observed across pairs (candidate had both large losses and one large win), so this path is not stable enough to ship.
- Decision: rejected for now; keep baseline kernel behavior.

### Attempt I (accepted): prefix-only specialization as default (prefix-first policy)
- Change:
  - default dispatch now uses dedicated prefix-only kernel when `num_prefix_ranges > 0 && num_suffix_targets == 0`.
  - suffix-only and prefix+suffix continue using the generic `vanity_kernel`.
- Strict interleaved A/B (`/tmp/vanity_ab_prefix_default_focus.csv`, 2 pairs/mode):
  - baseline: commit `0e1feb5`
  - candidate: prefix-only specialization default dispatch
  - warmup `12s`, measure `24s`, cooldown `15s`
  - fixed settings: `--cuda -t 2 --batch-size 8388608`
- Aggregate (avg medians):
  - Prefix: `73,977,773 -> 84,727,040` (`+10,749,267`, `+14.53%`)
  - Suffix: `63,353,813 -> 63,260,534` (`-93,279`, `-0.15%`)
- Decision: accepted and made default under prefix-first optimization policy.

## Optimization policy update (2026-02-16)
- We prioritize prefix throughput over suffix throughput for CUDA tuning decisions.
- Acceptance rule for default path:
  - significant prefix gain required
  - suffix impact must remain small/near-flat under strict interleaved A/B
- Benchmark source of truth remains strict interleaved runs captured by `scripts/bench_cuda_ab.sh`.

## 2026-02-21 checksum regression follow-up (kernel benchmark + recovery)

### Context
- Checksum rollout commit: `c7c4088`
- Parent baseline (pre-checksum): `4c062fd`
- Initial strict A/B (`/tmp/vanity_ab_checksum_regression.csv`, 2 pairs, warmup 8s, measure 16s):
  - Prefix: `83,826,040 -> 87,220,975` (`+4.05%`)
  - Suffix: `77,580,556 -> 56,301,992` (`-27.43%`)

### Attempt J (accepted): lightweight checksum absorb + chunked suffix modulo
- Changes:
  - Reworked `address_checksum4` absorb path to start from precomputed SHA3 base lanes, avoiding per-key 136-byte block setup.
  - Restored chunked suffix modulo reduction for checksummed addresses:
    - 8-byte chunk reduction for `spend_pub` and `view_pub`
    - 4-byte tail combine for `checksum4`
  - Added `suffix_chunk_mul` / `suffix_tail_mul` precompute on host and passed to CUDA worker setup.
- Compile telemetry (release):
  - `vanity_kernel` stack/spills improved from checksum baseline (`1176B`, `1544/1724`) to (`568B`, `572/748`).

### Clean validation vs checksum baseline (`c7c4088`)
- Suffix-only strict interleaved A/B (`/tmp/vanity_ab_opt2_suffix_clean.csv`, 3 pairs, warmup 10s, measure 20s):
  - Baseline avg median: `60,978,599`
  - Candidate avg median: `62,132,150`
  - Delta: `+1,153,551` (`+1.89%`)
- Prefix-only check (`/tmp/vanity_ab_opt2_prefix_clean.csv`, 2 pairs, warmup 8s, measure 16s):
  - Baseline avg median: `92,241,660`
  - Candidate avg median: `91,535,618`
  - Delta: `-706,042` (`-0.77%`)

### Position vs pre-checksum parent (`4c062fd`)
- Interleaved A/B (`/tmp/vanity_ab_parent_vs_opt2.csv`, 2 pairs, warmup 8s, measure 16s):
  - Prefix: `94,540,444 -> 93,841,348` (`-0.74%`)
  - Suffix: `82,386,436 -> 62,787,102` (`-23.79%`)
- Net: this recovers part of the checksum suffix regression (from `-27.43%` to `-23.79%` in this setup), but not all of it.

### Notes
- One intermediate suffix run showed a severe outlier due accidental external GPU miner activity and was discarded.
- Final accepted numbers above are from clean runs with miner stopped and interleaved ordering.
