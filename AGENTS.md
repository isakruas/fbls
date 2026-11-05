# Agent guide for tess-gpu-search

Read this before changing anything in this repository.

## What this repository is

A standalone CUDA float64 transit-search package (`tess_search/`) with a CLI,
its tests, and the tools that build and download the star samples it is
evaluated and run on (`targets/`). Everything that ships lives here
(github.com/isakruas/tess-gpu-search); nothing depends on files outside the
repository, and no path in the code points at a specific machine.

## Rules

- Code, comments, docstrings, commit messages and documentation in English;
  technical, no references to other methods inside code comments.
- Commits without any AI attribution lines.
- Heavy computation on the GPU in float64. No CPU fallbacks, no float32
  storage of scientific arrays. Float32 is allowed only for ranking inside a
  kernel when the winner is recomputed exactly in float64 (see `period_kernel.py`).
- No dead code, no stray files. Research scripts and experiments belong to the
  outer workspace, not here. Run `python -m pyflakes tess_search tests targets`
  and `python -m unittest discover -s tests` before committing.
- Never tune the detector on the control group (see Samples). Development
  uses the development group only.
- Catalog information (TOI, CTOI, planets, TCE, EB) is used after a blind
  search to evaluate it, or to exclude stars when building samples. It never
  enters the detector.

## Layout

| Path | Content |
|---|---|
| `tess_search/core.py` | Preparation (normalization, binning, duration families, edge trim), noise model, period grid, fused signal search, single-event scan, evidence evaluation |
| `tess_search/kernels.py`, `period_kernel.py`, `local_kernel.py` | Triton kernels: fold, interval sums, tile scan with shape gate, evidence bank |
| `tess_search/api.py` | `SearchConfig`, multi-pass orchestration, peak selection (isolated rescoring, event rules), result and JSON schema 3 |
| `tess_search/cli.py` | `tess-search` command: NPZ input, JSON/NPZ output |
| `tests/` | `test_contract.py` (no GPU), `test_pipeline_gpu.py`, `test_evidence_kernels.py` |
| `targets/` | `select_targets.py` (blind sample), `download.py` (MAST/AWS fetch, NPZ conversion), `control_sample.py` (control and development groups), outputs CSV/JSON/MD |
| `tess_survey/evaluate.py` | Blind recovery of catalogued planets over a labeled group (`data/evaluation/<name>/recovery.json`); development group for tuning, control group once with the frozen detector |
| `tess_survey/` | Survey layer after the detector: `catalog.py` (cross-match with TOI/CTOI/planet/TCE/EB snapshots), `vetting.py` (odd/even, secondary, shape, density, centroid, crowding), `pixels.py` (TPF difference-image source test), `fpp.py` (TRICERATOPS, optional), `ctoi.py` (ExoFOP CTOI row), `run.py` (`tess-survey`: per-star JSON reports under `data/reports/`, dated `public/candidates_*.csv` and `public/ctoi_*.csv`) |
| `docs/pipeline.mmd` | Mermaid flowchart of the detector |
| `data/` | Not versioned: `curves/<tic>/sector-N.npz` + `.json`, `mast/lc/*.fits`, `snapshots/` (MAST manifests and catalog snapshots from `fetch-snapshots`) |

## Samples in `data/`

| Group | File | Stars | Use |
|---|---|---|---|
| Development | `targets/development_targets.csv` | 268 (228 confirmed-planet hosts, 40 FP-only hosts) | Debugging and tuning; may be rerun freely |
| Control | `targets/control_targets.csv` | 219 (179 hosts stratified by period bin 50/50/50/29, 40 FP-only hosts) | One run with the frozen detector; never used for tuning |
| Blind | `targets/targets_download.csv` | 1,000 stars without catalogued signals (priorities A 500 / B 350 / C 150) | The actual blind search |

The two labeled groups are disjoint and were drawn with seeded shuffles from
the labeled cache of the neighbouring `tess` project (`targets/*_sources.json`).

## Detector in one paragraph

Each duration gets a detrending window `clamp(3 D, 0.4 d, 3.5 d)`; equal windows
form a family with its own residual, noise scales (`rho_D`) and period grid
(0.5 d to `min(100 d, baseline)`). The fused scan evaluates the box at every
phase bin and the trapezoidal shapes only where the box reaches 0.6 x threshold;
ranking is float32 with the exact float64 winner. Peaks are accepted greedily
with isolated rescoring (statistic without observations of accepted peaks,
compared with threshold x sqrt(unshared fraction)), >= 2 sampled events and a
veto when the statistic without the strongest event drops below half the
threshold. Up to three passes mask accepted peaks. Planet/binary/noise evidence
is evaluated for the strongest peaks with the other peaks masked. A separate
single-event scan reports isolated events.

## Measured state (2026-11-05)

Development group, contiguous sector windows, blind 0.5-100 d: 152 of 170
detectable planets recovered in the top 20 by SNR (89%); 95-98% below 10 d,
80% between 10 and 100 d. Known open problem: strong long-period planets in
multi-sector windows are missed (TOI-216, TOI-4504, TOI-186, TOI-3563,
TOI-1339, TOI-2525; statistic 23-91 at the catalog ephemeris). The
planet/binary evidence ranks only about half of the false positives below the
planets. Timing: ~5 s per one-sector star, ~1 min per 1,000-day baseline star
for the shortest-duration family.

## Survey flow

1. `tess-survey --targets targets/targets_download.csv` runs the detector on
   every star (all sectors, 0.5-100 d), measures each accepted peak with the
   other peaks masked, cross-matches the catalog snapshots, runs the pixel test
   on signals that pass vetting and TRICERATOPS on new ones, assigns a tier and
   writes `data/reports/<tic>.json`.
2. The same run appends the day's vetted signals to `public/candidates_YYYY-MM-DD.csv`
   and the new candidates (not catalogued, tier strong or candidate) to
   `public/ctoi_YYYY-MM-DD.csv` in the ExoFOP CTOI column layout. A star is
   processed once, so a candidate is published once.
3. No learned classifier is used; tiers come from vetting flags, the detector's
   planet/binary evidence and FPP/NFPP.

## Workflows

```bash
PY=.venv/bin/python               # created with: python3 -m venv .venv; torch==2.7.1 from the cu128 index; pip install -e '.[targets,dev]'
$PY -m unittest discover -s tests
$PY -m pyflakes tess_search tests targets
$PY -m tess_search --input data/curves/<tic>/sector-*.npz --pmax 100 --output out.json
$PY targets/select_targets.py fetch-snapshots && $PY targets/select_targets.py fetch-tic && $PY targets/select_targets.py select
$PY targets/download.py --pilot && $PY targets/download.py --workers 4
$PY -m tess_survey.run --targets targets/targets_download.csv        # or: .venv/bin/tess-survey
$PY -m tess_survey.evaluate --group targets/development_targets.csv --name development --workers 4
```
