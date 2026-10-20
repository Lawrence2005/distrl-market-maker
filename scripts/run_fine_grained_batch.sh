#!/usr/bin/env bash
# scripts/run_fine_grained_batch.sh
#
# Collects the fine-grained market-data batch with scripts/collect_fine_grained.py:
# every regime x policy cell, N_EPISODES episodes each, split into CHUNK-sized
# jobs run MAX_CONCURRENT at a time.
#
# Chunking keeps each job's memory bounded (the recorder holds a job's raw
# events until it finishes) and makes the batch resumable: a chunk whose
# output already has a meta.json is skipped, so re-running the script after an
# interruption only does the missing chunks. Chunks are ordered chunk-major, so
# at any moment every cell has covered the same seeds.
#
# Chunk i of a cell covers seeds BASE_SEED + i*CHUNK ... + CHUNK-1 and is written to
#   data/processed/fine_grained/<regime>_<policy>_seed<first seed>/
# (the `seed` column is the unique episode id; `episode` restarts per chunk).
# With the defaults chunks 0-2 are seeds 90000-90059 (discovery) and chunks 3-4
# are 90060-90099 (confirmation).
#
# This machine is memory-constrained (~9.7GB, with the editor using ~4GB) and
# each job uses ~0.9-1.2GB, so the default cap is 3 concurrent jobs.
#
# Run unattended:
#   nohup setsid scripts/run_fine_grained_batch.sh > logs/fine_grained_batch.log 2>&1 &
# Progress: ls data/processed/fine_grained/*/meta.json | wc -l   (60 = done)
#   per-job logs: logs/fine_grained/<regime>_<policy>_seed<seed>.log
#
# Overrides (environment): REGIMES, POLICIES, BASE_SEED, N_EPISODES, CHUNK,
# MAX_CONCURRENT, OUT_DIR, DRY_RUN=1 to list the jobs without running them.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REGIMES="${REGIMES:-low_vol normal high_vol}"
POLICIES="${POLICIES:-glft random as fixedspread}"
BASE_SEED="${BASE_SEED:-90000}"
N_EPISODES="${N_EPISODES:-100}"
CHUNK="${CHUNK:-20}"
MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
OUT_DIR="${OUT_DIR:-data/processed/fine_grained}"
LOG_DIR="logs/fine_grained"

mkdir -p "$LOG_DIR"
# shellcheck disable=SC1091
source venv/bin/activate

jobs_file="$(mktemp)"
trap 'rm -f "$jobs_file"' EXIT

for ((start = 0; start < N_EPISODES; start += CHUNK)); do
    n=$(( N_EPISODES - start < CHUNK ? N_EPISODES - start : CHUNK ))
    for regime in $REGIMES; do
        for policy in $POLICIES; do
            seed=$(( BASE_SEED + start ))
            if [[ -f "$OUT_DIR/${regime}_${policy}_seed${seed}/meta.json" ]]; then
                continue
            fi
            echo "$regime $policy $seed $n" >> "$jobs_file"
        done
    done
done

total=$(wc -l < "$jobs_file")
echo "=== fine-grained batch @ $(date -Iseconds): $total chunk jobs, max_concurrent=$MAX_CONCURRENT, out=$OUT_DIR ==="

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    cat "$jobs_file"
    exit 0
fi

run_one() {
    local regime="$1" policy="$2" seed="$3" n="$4"
    local tag="${regime}_${policy}_seed${seed}"
    echo "[$(date +%T)] start  $tag ($n episodes)"
    python scripts/collect_fine_grained.py \
        --regime "$regime" --policy "$policy" --seed "$seed" --n_episodes "$n" \
        --out_dir "$OUT_DIR" > "$LOG_DIR/${tag}.log" 2>&1
    local rc=$?
    echo "[$(date +%T)] done   $tag rc=$rc"
}
export -f run_one
export OUT_DIR LOG_DIR

xargs -P "$MAX_CONCURRENT" -L 1 bash -c 'run_one "$@"' _ < "$jobs_file"

done_n=$(ls "$OUT_DIR"/*/meta.json 2>/dev/null | wc -l)
echo "=== finished @ $(date -Iseconds): $done_n output dirs with meta.json ==="
