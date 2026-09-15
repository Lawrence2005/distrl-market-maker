#!/usr/bin/env bash
# Looks up scripts/jobs/<name>.args and launches it through
# run_agent_supervised.sh. This indirection exists so a systemd unit only
# needs to know a short job name (%i), not the full hydra override list —
# see scripts/systemd/distrl-train@.service.
#
# Job args file format (one per line):
#   line 1:      log file path
#   remaining:   hydra overrides, one per line
#
# Usage: scripts/run_job.sh <name>   (name matches scripts/jobs/<name>.args)

set -uo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <job-name>" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="$1"
ARGS_FILE="$REPO_ROOT/scripts/jobs/$NAME.args"

if [[ ! -f "$ARGS_FILE" ]]; then
    echo "run_job.sh: no such job args file: $ARGS_FILE" >&2
    exit 1
fi

mapfile -t lines < "$ARGS_FILE"
LOG_FILE="${lines[0]}"
OVERRIDES=("${lines[@]:1}")

exec "$REPO_ROOT/scripts/run_agent_supervised.sh" "$LOG_FILE" "${OVERRIDES[@]}"
