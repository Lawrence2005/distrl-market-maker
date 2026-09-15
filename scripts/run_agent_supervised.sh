#!/usr/bin/env bash
# Runs one training job under training/train.py, auto-restarting with
# training.resume=true if the process ever dies (OOM kill, an unhandled
# exception, the host/WSL VM restarting mid-run, ...) instead of the run
# silently going nowhere for days.
#
# Usage:
#   scripts/run_agent_supervised.sh <log_file> <hydra overrides...>
#
# Example (replaces the bare `nohup python training/train.py ... &` pattern):
#   for agent in dqn qrdqn iqn ppo sarsa; do
#     nohup scripts/run_agent_supervised.sh \
#       ~/distrl-logs/w06_${agent}_low_vol_seed42.out \
#       agent=$agent encoder=handcrafted reward=asymmetric \
#       env.regime=low_vol env.use_abides=true training.n_episodes=1000 seed=42 \
#       > /dev/null 2>&1 &
#   done
#
# Progress can be checked at any time with scripts/status.sh, or by
# `cat logs/<run_tag>/status.json`, without tailing these log files.

set -uo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <log_file> <hydra overrides...>" >&2
    exit 1
fi

LOG_FILE="$1"; shift
OVERRIDES=("$@")

# This mkdir is the fix for the failure mode that started this: if the log
# file's directory doesn't exist, the shell redirect in a plain
# `nohup ... > file &` fails silently and the job never runs at all.
mkdir -p "$(dirname "$LOG_FILE")"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Absolute path, not just `python` on PATH — this script also runs as a
# systemd ExecStart (see scripts/systemd/distrl-train@.service), which
# doesn't have the venv activated.
PYTHON="$REPO_ROOT/venv/bin/python"

MAX_RETRIES=20
RETRY_DELAY=30

attempt=0
while (( attempt < MAX_RETRIES )); do
    attempt=$((attempt + 1))
    {
        echo "=== attempt ${attempt}/${MAX_RETRIES} @ $(date -Iseconds) ==="
        "$PYTHON" training/train.py "${OVERRIDES[@]}" training.resume=true
    } >> "$LOG_FILE" 2>&1
    exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        echo "=== completed successfully @ $(date -Iseconds) ===" >> "$LOG_FILE"
        exit 0
    fi

    echo "=== exited with code ${exit_code}, retrying in ${RETRY_DELAY}s (${attempt}/${MAX_RETRIES}) ===" >> "$LOG_FILE"
    sleep "$RETRY_DELAY"
done

echo "=== gave up after ${MAX_RETRIES} attempts ===" >> "$LOG_FILE"
exit 1
