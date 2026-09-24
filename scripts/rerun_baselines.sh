#!/bin/bash
# Regenerates logs/{baseline}_analytical_asymmetric_{regime}_seed42/train_history.json
# for all 3 baselines x 3 regimes against the current (bug-fixed) simulator —
# the existing files predate the envs/lob_env.py reward/mark-to-market/
# sign-flip fixes (Sep 14-16 vs the Sep 19 fix) and would otherwise be
# silently picked up as stale by scripts/run_analysis.py's
# run_model_comparison() (via evaluation.metrics.summary_table(), which has
# no freshness check on logs/*/train_history.json).
set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

MAX_CONCURRENT=2
LOGDIR=/home/lawre/distrl-logs
mkdir -p "$LOGDIR"

BASELINES="fixedspread as glft"
REGIMES="low_vol normal high_vol"

for baseline in $BASELINES; do
  for regime in $REGIMES; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_CONCURRENT" ]; do
      wait -n
    done
    echo "Launching: $baseline / $regime"
    python scripts/run_baseline.py --baseline "$baseline" --regime "$regime" \
      --seed 42 --n_episodes 50 \
      > "$LOGDIR/baseline_${baseline}_${regime}.out" 2>&1 &
  done
done

wait
echo "All baseline reruns complete."
