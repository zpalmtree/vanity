#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Interleaved CUDA A/B benchmark harness for blocknet-vanity.

Usage:
  scripts/bench_cuda_ab.sh \
    --baseline-bin <path> \
    --candidate-bin <path> \
    [--modes <csv>] \
    [--out <csv>] \
    [--pairs <n>] \
    [--warmup-secs <n>] \
    [--measure-secs <n>] \
    [--cooldown-secs <n>] \
    [--threads <n>] \
    [--batch-size <n>] \
    [--prefix <pattern>] \
    [--suffix <pattern>]

Defaults:
  --modes prefix,suffix
  --out /tmp/vanity_ab_cuda.csv
  --pairs 3
  --warmup-secs 12
  --measure-secs 24
  --cooldown-secs 15
  --threads 2
  --batch-size 8388608
  --prefix zzzzzzzz
  --suffix zzzzzzzz
USAGE
}

baseline_bin=""
candidate_bin=""
out_csv="/tmp/vanity_ab_cuda.csv"
pairs=3
warmup_secs=12
measure_secs=24
cooldown_secs=15
threads=2
batch_size=8388608
prefix_pattern="zzzzzzzz"
suffix_pattern="zzzzzzzz"
modes_csv="prefix,suffix"

while (($#)); do
    case "$1" in
        --baseline-bin)
            baseline_bin="${2:-}"
            shift 2
            ;;
        --candidate-bin)
            candidate_bin="${2:-}"
            shift 2
            ;;
        --out)
            out_csv="${2:-}"
            shift 2
            ;;
        --modes)
            modes_csv="${2:-}"
            shift 2
            ;;
        --pairs)
            pairs="${2:-}"
            shift 2
            ;;
        --warmup-secs)
            warmup_secs="${2:-}"
            shift 2
            ;;
        --measure-secs)
            measure_secs="${2:-}"
            shift 2
            ;;
        --cooldown-secs)
            cooldown_secs="${2:-}"
            shift 2
            ;;
        --threads)
            threads="${2:-}"
            shift 2
            ;;
        --batch-size)
            batch_size="${2:-}"
            shift 2
            ;;
        --prefix)
            prefix_pattern="${2:-}"
            shift 2
            ;;
        --suffix)
            suffix_pattern="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown arg: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$baseline_bin" || -z "$candidate_bin" ]]; then
    echo "error: --baseline-bin and --candidate-bin are required" >&2
    usage >&2
    exit 1
fi

if [[ ! -x "$baseline_bin" ]]; then
    echo "error: baseline binary is not executable: $baseline_bin" >&2
    exit 1
fi
if [[ ! -x "$candidate_bin" ]]; then
    echo "error: candidate binary is not executable: $candidate_bin" >&2
    exit 1
fi

for n in "$pairs" "$warmup_secs" "$measure_secs" "$cooldown_secs" "$threads" "$batch_size"; do
    if ! [[ "$n" =~ ^[0-9]+$ ]]; then
        echo "error: numeric args must be non-negative integers" >&2
        exit 1
    fi
done
if ((pairs < 1)); then
    echo "error: --pairs must be >= 1" >&2
    exit 1
fi
if ((measure_secs < 1)); then
    echo "error: --measure-secs must be >= 1" >&2
    exit 1
fi
if ((threads < 1)); then
    echo "error: --threads must be >= 1" >&2
    exit 1
fi
if ((batch_size < 8)); then
    echo "error: --batch-size must be >= 8" >&2
    exit 1
fi

IFS=',' read -r -a modes_raw <<< "$modes_csv"
modes=()
for mode in "${modes_raw[@]}"; do
    mode="$(echo "$mode" | tr -d '[:space:]')"
    case "$mode" in
        prefix|suffix)
            modes+=("$mode")
            ;;
        "")
            ;;
        *)
            echo "error: unsupported mode '$mode' in --modes (use prefix,suffix)" >&2
            exit 1
            ;;
    esac
done
if ((${#modes[@]} == 0)); then
    echo "error: --modes must include at least one of prefix or suffix" >&2
    exit 1
fi

mkdir -p "$(dirname "$out_csv")"
printf "variant,mode,pair,slot,median_rate,mean_rate,max_rate,min_rate,last_rate,samples,log_file\n" > "$out_csv"

extract_rates() {
    local log_file="$1"
    tr '\r' '\n' < "$log_file" \
        | sed -E 's/\x1b\[[0-9;]*[A-Za-z]//g' \
        | awk -v w="$warmup_secs" -v m="$measure_secs" '
            match($0, /([0-9,]+) keys \| ([0-9]+)\/s \| [0-9]+ found \| ([0-9]+)s/, a) {
                sec = a[3] + 0;
                rate = a[2] + 0;
                if (sec > w && sec <= (w + m)) {
                    print rate;
                    used = 1;
                }
                all_rates[++n_all] = rate;
                all_secs[n_all] = sec;
            }
            END {
                if (!used) {
                    for (i = 1; i <= n_all; i++) {
                        if (all_secs[i] > w) {
                            print all_rates[i];
                        }
                    }
                }
            }
        '
}

calc_stats() {
    local rates="$1"
    local samples
    samples="$(printf "%s\n" "$rates" | sed '/^$/d' | wc -l | tr -d ' ')"
    if [[ -z "$samples" || "$samples" == "0" ]]; then
        echo "0,0,0,0,0,0"
        return
    fi

    local sorted
    sorted="$(printf "%s\n" "$rates" | sed '/^$/d' | sort -n)"
    local median
    median="$(printf "%s\n" "$sorted" | awk '
        { a[NR] = $1 }
        END {
            if (NR == 0) { print 0; exit }
            if (NR % 2 == 1) {
                print a[(NR + 1) / 2]
            } else {
                printf "%.0f\n", (a[NR / 2] + a[NR / 2 + 1]) / 2
            }
        }
    ')"
    local mean
    mean="$(printf "%s\n" "$rates" | sed '/^$/d' | awk '
        { sum += $1; n++ }
        END {
            if (n == 0) print 0;
            else printf "%.0f\n", sum / n;
        }
    ')"
    local max_rate
    max_rate="$(printf "%s\n" "$rates" | sed '/^$/d' | awk '
        NR == 1 || $1 > max { max = $1 }
        END { if (NR == 0) print 0; else print max }
    ')"
    local min_rate
    min_rate="$(printf "%s\n" "$rates" | sed '/^$/d' | awk '
        NR == 1 || $1 < min { min = $1 }
        END { if (NR == 0) print 0; else print min }
    ')"
    local last_rate
    last_rate="$(printf "%s\n" "$rates" | sed '/^$/d' | tail -n 1)"

    echo "${samples},${median},${mean},${max_rate},${min_rate},${last_rate}"
}

total_runs=$((pairs * 2 * ${#modes[@]}))
run_index=0

run_one() {
    local variant="$1"
    local mode="$2"
    local pair="$3"
    local slot="$4"
    local bin="$5"

    local pattern_args=()
    if [[ "$mode" == "prefix" ]]; then
        pattern_args=(--prefix "$prefix_pattern")
    else
        pattern_args=(--suffix "$suffix_pattern")
    fi

    local round_timeout=$((warmup_secs + measure_secs + 8))
    local log_file
    log_file="$(mktemp "/tmp/vanity_ab_${mode}_${variant}_p${pair}_${slot}_XXXX.log")"

    echo "[run] mode=${mode} pair=${pair} slot=${slot} variant=${variant}"
    timeout "${round_timeout}s" "$bin" --cuda -t "$threads" --batch-size "$batch_size" "${pattern_args[@]}" > "$log_file" 2>&1 || true

    local rates
    rates="$(extract_rates "$log_file")"
    local stats
    stats="$(calc_stats "$rates")"

    local samples median mean max_rate min_rate last_rate
    IFS=',' read -r samples median mean max_rate min_rate last_rate <<< "$stats"

    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$variant" "$mode" "$pair" "$slot" "$median" "$mean" "$max_rate" "$min_rate" "$last_rate" "$samples" "$log_file" \
        >> "$out_csv"

    echo "  median=${median} mean=${mean} max=${max_rate} min=${min_rate} samples=${samples}"

    run_index=$((run_index + 1))
    if ((cooldown_secs > 0 && run_index < total_runs)); then
        echo "  cooldown ${cooldown_secs}s"
        sleep "$cooldown_secs"
    fi
}

for mode in "${modes[@]}"; do
    for pair in $(seq 1 "$pairs"); do
        if ((pair % 2 == 1)); then
            run_one baseline "$mode" "$pair" first "$baseline_bin"
            run_one candidate "$mode" "$pair" second "$candidate_bin"
        else
            run_one candidate "$mode" "$pair" first "$candidate_bin"
            run_one baseline "$mode" "$pair" second "$baseline_bin"
        fi
    done
done

echo
echo "[summary] csv=$out_csv"
awk -F',' '
NR > 1 {
    key = $2 "|" $1;
    sum[key] += $5;
    n[key] += 1;
}
END {
    for (k in sum) {
        split(k, p, "|");
        mode = p[1];
        variant = p[2];
        avg = (n[k] > 0) ? sum[k] / n[k] : 0;
        printf "  mode=%s variant=%s avg_median=%.0f samples=%d\n", mode, variant, avg, n[k];
    }
}
' "$out_csv" | sort
