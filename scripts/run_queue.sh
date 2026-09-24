#!/usr/bin/env bash
# scripts/run_queue.sh
#
# Launches a queue of training jobs (each matching scripts/jobs/<name>.args)
# one at a time as slots free up, keeping at most MAX_CONCURRENT running
# concurrently via scripts/run_job.sh — this machine is memory-constrained
# (~9.7GB), and each real-ABIDES training job uses ~1-1.2GB, so 2 concurrent
# jobs is the safe cap established this session (leaves headroom for eval/
# pytest work run alongside).
#
# Runs to completion unattended (intended for a multi-day campaign) — start
# it with nohup so it survives the launching shell exiting:
#   nohup scripts/run_queue.sh job1 job2 job3 ... > /path/to/queue.log 2>&1 &
#
# Progress: /path/to/queue.log gets one line per launch/completion. Check
# any individual job's own progress the normal way (logs/<run_tag>/status.json
# or scripts/status.sh) — this script only manages concurrency, it doesn't
# duplicate per-job progress reporting.

set -uo pipefail

MAX_CONCURRENT=2
POLL_INTERVAL=120

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <job-name> [<job-name> ...]  (names matching scripts/jobs/<name>.args)" >&2
    exit 1
fi

running_count() {
    pgrep -f "run_agent_supervised\.sh" | wc -l
}

echo "=== queue started @ $(date -Iseconds), $# jobs, max_concurrent=$MAX_CONCURRENT ==="

for job in "$@"; do
    if [[ ! -f "scripts/jobs/${job}.args" ]]; then
        echo "=== SKIP: no args file for '$job' ==="
        continue
    fi
    while (( $(running_count) >= MAX_CONCURRENT )); do
        sleep "$POLL_INTERVAL"
    done
    echo "=== launching $job @ $(date -Iseconds) (running before launch: $(running_count)) ==="
    nohup scripts/run_job.sh "$job" > /dev/null 2>&1 &
    disown
    sleep 15   # stagger so two launches never race the same memory check
done

echo "=== all jobs launched @ $(date -Iseconds), waiting for the last ones to finish ==="
while (( $(running_count) > 0 )); do
    sleep "$POLL_INTERVAL"
done

echo "=== QUEUE COMPLETE @ $(date -Iseconds) ==="
