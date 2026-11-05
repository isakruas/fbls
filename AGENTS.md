# Agent guide for tess-gpu-search

## Rules

- English only in code, comments, docs and commit messages. No AI attribution in commits.
- GPU float64 for scientific computation; no CPU fallbacks. Float32 only inside a kernel
  for ranking, with the winner recomputed in float64.
- Self-contained: nothing points outside the repository, no absolute machine paths in any
  file (including untracked outputs under `data/`). Run `grep -r "/home/" --exclude-dir=.venv --exclude-dir=.git .`
  before committing.
- No dead code, no stray files. Before committing:
  `.venv/bin/python -m pyflakes tess_search tess_survey tests targets` and
  `.venv/bin/python -m unittest discover -s tests`.
- Catalog information (TOI, CTOI, planets, TCE, EB) is used after a search, never inside the detector.
- The control group (`targets/control_targets.csv`) is never used for tuning; it is evaluated once
  with the frozen detector. Tuning uses `targets/development_targets.csv`.

## Layout

| Path | Content |
|---|---|
| `tess_search/` | Detector: preparation and duration families (`core.py`), Triton kernels, multi-pass peak selection and JSON schema (`api.py`), `tess-search` CLI |
| `tess_survey/` | After the detector: catalog cross-match, vetting, TPF pixel test, TRICERATOPS FPP, ExoFOP CTOI rows, `tess-survey` runner, `evaluate.py` (recovery over a labeled group) |
| `targets/` | Sample tools: `select_targets.py` (blind sample and snapshots), `download.py`, `control_sample.py`; CSV/JSON outputs |
| `tests/` | `test_contract.py` (no GPU), `test_pipeline_gpu.py`, `test_evidence_kernels.py` |
| `docs/pipeline.mmd` | Detector flowchart (Mermaid) |
| `data/` (ignored) | `curves/<tic>/sector-N.npz` + `.json`, `mast/`, `snapshots/`, `reports/`, `evaluation/` |
| `public/` | Dated `candidates_*.csv` and `ctoi_*.csv` written by `tess-survey` |

## Workflows

```bash
python3 -m venv .venv
.venv/bin/python -m pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -e '.[targets,survey,dev]'
.venv/bin/python targets/select_targets.py fetch-snapshots && .venv/bin/python targets/select_targets.py fetch-tic
.venv/bin/python targets/select_targets.py select && .venv/bin/python targets/download.py --pilot && .venv/bin/python targets/download.py --workers 4
.venv/bin/tess-survey --targets targets/targets_download.csv
.venv/bin/python -m tess_survey.evaluate --group targets/control_targets.csv --name control --workers 4
```
