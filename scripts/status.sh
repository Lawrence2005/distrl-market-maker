#!/usr/bin/env bash
# Prints a one-line progress summary for every run under logs/, reading
# each run's logs/<run_tag>/status.json (written by training/train.py).
#
# Usage: scripts/status.sh

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

printf "%-55s %-9s %13s %7s %9s %8s\n" "RUN" "STATE" "EPISODE" "PCT" "SHARPE" "ETA"

shopt -s nullglob
for f in logs/*/status.json; do
    python3 -c "
import json
d = json.load(open('$f'))
eta = d.get('eta_sec')
eta_str = f'{eta/3600:.1f}h' if eta else '-'
sharpe = d.get('sharpe')
sharpe_str = f'{sharpe:+.3f}' if sharpe is not None else '-'
print(f\"{d['run_tag']:<55} {d['state']:<9} {d['episode']:>5}/{d['n_episodes']:<6} {d['pct_complete']:>6.1f}% {sharpe_str:>9} {eta_str:>8}\")
"
done
