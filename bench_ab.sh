#!/usr/bin/env bash
# Interleaved A/B benchmark for blocknet-vanity
# Usage: ./bench_ab.sh <baseline_binary> <candidate_binary> [threads]
#
# Runs 4 pairs of alternating baseline/candidate measurements with warmup
# and cooldown between each run. Outputs CSV to stdout.

set -euo pipefail

BASELINE="${1:?Usage: $0 <baseline_binary> <candidate_binary> [threads]}"
CANDIDATE="${2:?Usage: $0 <baseline_binary> <candidate_binary> [threads]}"
THREADS="${3:-1}"

WARMUP=12
DURATION=30
COOLDOWN=20
PAIRS=4
PREFIX="zzzzzzzz"

# Extract the last throughput rate from stderr output
extract_rate() {
    # Matches "123456/s" or "123456.7/s" from the status line
    grep -oE '[0-9]+\.?[0-9]*/s' | tail -1 | sed 's|/s||'
}

run_one() {
    local binary="$1"
    local label="$2"
    # Warmup
    "$binary" --prefix "$PREFIX" -t "$THREADS" --duration "$WARMUP" 2>/dev/null
    sleep 2
    # Measured run — capture stderr
    local rate
    rate=$("$binary" --prefix "$PREFIX" -t "$THREADS" --duration "$DURATION" 2>&1 >/dev/null | extract_rate)
    echo "$rate"
}

echo "pair,variant,rate"

for pair in $(seq 1 "$PAIRS"); do
    if (( pair % 2 == 1 )); then
        # Odd pair: baseline first
        first_bin="$BASELINE";  first_label="baseline"
        second_bin="$CANDIDATE"; second_label="candidate"
    else
        # Even pair: candidate first
        first_bin="$CANDIDATE"; first_label="candidate"
        second_bin="$BASELINE";  second_label="baseline"
    fi

    rate=$(run_one "$first_bin" "$first_label")
    echo "${pair},${first_label},${rate}"
    sleep "$COOLDOWN"

    rate=$(run_one "$second_bin" "$second_label")
    echo "${pair},${second_label},${rate}"
    sleep "$COOLDOWN"
done
