#!/usr/bin/env bash
# GitHub release of one deposit folder: tag v<day>, the four files of paper/deposit/<day>/ as the
# assets, the ReadMe as the notes. Zenodo's GitHub integration turns the release into a record version.
#   paper/release.sh [day]      (default: today, UTC)
set -euo pipefail
cd "$(dirname "$0")/.."
DAY=${1:-$(date -u +%F)}
FOLDER=paper/deposit/$DAY
[ -d "$FOLDER" ] || { echo "no $FOLDER; run paper/build.py first" >&2; exit 1; }
gh release create "v$DAY" "$FOLDER"/* --title "Catalog $DAY" --notes-file "$FOLDER/ReadMe.txt"
