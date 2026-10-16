#!/bin/bash

set -euo pipefail
export LC_ALL=C

binary=${1:-./matrix_cuda}
output_dir=${2:-results/sergey-korolev}
repeats=${3:-5}
seed=42
sizes=(200 400 800 1200 1600 2000)
tiles=(4 8 16 32)

if [[ ! -x "$binary" ]]; then
    printf 'CUDA binary not found or not executable: %s\n' "$binary" >&2
    exit 1
fi
if ! [[ "$repeats" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Repeat count must be a positive integer.\n' >&2
    exit 1
fi

mkdir -p "$output_dir/reference"
printf 'size,processes,tile_size,repeat,time_seconds,end_to_end_seconds,matrix_bytes,scalar_operations,matches_baseline\n' \
    > "$output_dir/raw.csv"
printf 'size,processes,tile_size,repeats,median_seconds,end_to_end_median_seconds,min_seconds,max_seconds,stddev_seconds,speedup,matches_baseline\n' \
    > "$output_dir/summary.csv"

temporary=$(mktemp -d "${TMPDIR:-/tmp}/lab4-cuda.XXXXXX")
trap 'rm -rf "$temporary"' EXIT
a_path="$temporary/A.txt"
b_path="$temporary/B.txt"
c_path="$temporary/C.txt"
run_output="$temporary/run.out"
baseline_path="$temporary/C_baseline.txt"
device=""

write_matrix() {
    local path=$1
    local size=$2
    local state=$3
    awk -v n="$size" -v state="$state" '
        BEGIN {
            printf "%d\n", n
            for (i = 0; i < n; ++i) {
                for (j = 0; j < n; ++j) {
                    state = (state * 48271) % 2147483647
                    value = state % 19 - 9
                    printf "%s%d", j == 0 ? "" : " ", value
                }
                printf "\n"
            }
        }
    ' > "$path"
}

field() {
    local name=$1
    local path=$2
    awk -F= -v name="$name" '$1 == name { print $2; exit }' "$path"
}

stats() {
    printf '%s\n' "$@" | sort -n | awk '
        { values[NR] = $1; sum += $1; sum_squared += $1 * $1 }
        END {
            count = NR
            if (count % 2) median = values[(count + 1) / 2]
            else median = (values[count / 2] + values[count / 2 + 1]) / 2
            mean = sum / count
            variance = sum_squared / count - mean * mean
            if (variance < 0) variance = 0
            printf "%.9g %.9g %.9g %.9g\n", median, values[1], values[count], sqrt(variance)
        }
    '
}

for size in "${sizes[@]}"; do
    write_matrix "$a_path" "$size" "$((seed + size * 1009))"
    write_matrix "$b_path" "$size" "$((seed + size * 1009 + 1))"
    if [[ -e "$baseline_path" ]]; then rm -f "$baseline_path"; fi
    baseline_median=""

    for tile in "${tiles[@]}"; do
        times=()
        wall_times=()
        for ((repeat = 0; repeat <= repeats; ++repeat)); do
            if ! "$binary" "$a_path" "$b_path" "$c_path" "$tile" > "$run_output" 2>&1; then
                cat "$run_output" >&2
                exit 1
            fi

            device=$(field device "$run_output")
            elapsed=$(field time_seconds "$run_output")
            end_to_end=$(field end_to_end_seconds "$run_output")
            expected_bytes=$((24 * size * size))
            expected_operations=$((2 * size * size * size))
            [[ "$(field size "$run_output")" == "$size" ]] || { printf 'Wrong matrix size reported.\n' >&2; exit 1; }
            [[ "$(field tile_size "$run_output")" == "$tile" ]] || { printf 'Wrong tile size reported.\n' >&2; exit 1; }
            [[ "$(field matrix_bytes "$run_output")" == "$expected_bytes" ]] || { printf 'Wrong matrix volume reported.\n' >&2; exit 1; }
            [[ "$(field scalar_operations "$run_output")" == "$expected_operations" ]] || { printf 'Wrong operation count reported.\n' >&2; exit 1; }

            if (( tile == 4 && repeat == 0 )); then cp "$c_path" "$baseline_path"; fi
            if ! cmp -s "$c_path" "$baseline_path"; then
                printf 'Output differs from tile 4 at n=%s, tile=%s, repeat=%s.\n' "$size" "$tile" "$repeat" >&2
                exit 1
            fi

            if (( repeat > 0 )); then
                times+=("$elapsed")
                wall_times+=("$end_to_end")
                printf '%s,,%s,%s,%s,%s,%s,%s,True\n' \
                    "$size" "$tile" "$repeat" "$elapsed" "$end_to_end" "$expected_bytes" "$expected_operations" \
                    >> "$output_dir/raw.csv"
            fi
        done

        read -r median minimum maximum deviation <<< "$(stats "${times[@]}")"
        read -r wall_median _ _ _ <<< "$(stats "${wall_times[@]}")"
        if (( tile == 4 )); then baseline_median=$median; fi
        speedup=$(awk -v base="$baseline_median" -v value="$median" 'BEGIN { printf "%.9g", base / value }')
        printf '%s,,%s,%s,%s,%s,%s,%s,%s,%s,True\n' \
            "$size" "$tile" "$repeats" "$median" "$wall_median" "$minimum" "$maximum" "$deviation" "$speedup" \
            >> "$output_dir/summary.csv"
        printf 'n=%s, tile=%s: median=%ss, matches_baseline=True\n' "$size" "$tile" "$median" \
            | tee -a "$output_dir/benchmark.log"
    done

    gzip -c "$baseline_path" > "$output_dir/reference/C_${size}.txt.gz"
done

size_json=$(IFS=,; printf '%s' "${sizes[*]}")
tile_json=$(IFS=,; printf '%s' "${tiles[*]}")
binary_sha256=$(sha256sum "$binary" | awk '{ print $1 }')
cuda_version=$(nvcc --version | awk -F 'release ' '/release / { split($2, parts, ","); print parts[1]; exit }')
cpu_count=$(getconf _NPROCESSORS_ONLN)
created_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
host=$(hostname)
os=$(uname -sr)
architecture=$(uname -m)

cat > "$output_dir/environment.json" <<EOF
{
  "created_at_utc": "$created_at",
  "program": "cuda",
  "host": "$host",
  "os": "$os",
  "architecture": "$architecture",
  "logical_cpu_count": $cpu_count,
  "compiler": "nvcc",
  "compiler_version": "CUDA $cuda_version",
  "compiler_flags": "-O2 -arch=sm_20",
  "binary": "$(basename "$binary")",
  "binary_sha256": "$binary_sha256",
  "gpu_device": "$device",
  "sizes": [$size_json],
  "configurations": [$tile_json],
  "repeats": $repeats,
  "warmups_per_configuration": 1,
  "seed": $seed,
  "input_generator": "Park-Miller LCG, integer matrix values in [-9, 9]",
  "reference_outputs": "results/reference/C_<size>.txt.gz; local NumPy verification required",
  "parallel_consistency": "Each run output compared byte-for-byte with the 4 x 4 baseline",
  "timing_scope": "CUDA events for kernel; wall time includes allocation, copies and file output",
  "slurm_job_id": "${SLURM_JOB_ID:-}"
}
EOF

printf 'Results saved to %s\n' "$output_dir"
