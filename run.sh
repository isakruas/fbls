#!/usr/bin/env bash
# Post-search stages of the survey, in order; each one processes only what the previous stage
# produced and skips what it already did. The GPU search runs separately:
#   nohup .venv/bin/fbls-survey search --targets targets/targets_download.csv --workers 4 > search.log 2>&1 &
#   nohup .venv/bin/fbls --input data/curves/3756996/sector-*.npz --pmax 100 --output public/2026-11-05/figures/3756996_search.json --save-table public/2026-11-05/figures/3756996_table.npz

set -euo pipefail
cd "$(dirname "$0")"
TARGETS=${1:-targets/targets_download.csv}
WORKERS=${WORKERS:-4}
.venv/bin/fbls-survey vet --targets "$TARGETS" --workers "$WORKERS"
.venv/bin/fbls-survey pixel --targets "$TARGETS" --workers "$WORKERS"
.venv/bin/fbls-survey fpp --targets "$TARGETS" --workers "$WORKERS"
.venv/bin/fbls-survey refine --targets "$TARGETS"
.venv/bin/fbls-survey publish --targets "$TARGETS"
