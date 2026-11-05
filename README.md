# tess-gpu-search

CUDA float64 transit search for TESS light curves, with the survey layer that
vets candidates and writes ExoFOP CTOI rows. Linux, NVIDIA GPU, Python 3.10+.

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -e '.[targets,survey,dev]'
```

## Use

```bash
tess-search --input curve.npz --pmax 100 --output detections.json      # one star, JSON schema 3
tess-survey --targets targets/targets_download.csv                      # search, cross-match, vetting, CTOI rows in public/
python targets/select_targets.py fetch-snapshots                        # catalog and MAST snapshots
python targets/select_targets.py fetch-tic && python targets/select_targets.py select
python targets/download.py --pilot && python targets/download.py --workers 4
```

Input NPZ columns: `time_bjd`, `flux`, optional `flux_err`, `sector`, `centroid`.
Output: `candidates` (model-preferred first), `signal_candidates` (by SNR),
`single_events`, and the per-period table (`--save-table`).

## Method in brief

Each transit duration gets its own detrending window and period grid
(duration families). A fused Triton kernel scans every period, phase and shape
with the box first and trapezoids only where the box is significant; ranking in
float32, exact winner in float64. Peaks are accepted by isolated rescoring
without the observations of accepted peaks, two sampled events and a veto on
single dominant events; accepted peaks are masked and the search repeats up to
three times. A planet/binary/noise evidence ranks the strongest peaks, and an
unfolded scan reports single events. Full description: `docs/method.md`;
flowchart: `docs/pipeline.mmd`; agent rules: `AGENTS.md`.

Author: Isak Paulo de Andrade Ruas, NROLABS.
