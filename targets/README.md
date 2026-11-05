# Target selection and download

Tools to build reproducible star samples and to download SPOC 120 s light curves
into `data/` in the layout the search CLI reads (`curves/<tic>/sector-N.npz` + `.json`,
originals in `mast/lc/`). Nothing here depends on files outside this repository:
`fetch-snapshots` downloads the official inputs into `data/snapshots/`.

```bash
python targets/select_targets.py fetch-snapshots     # MAST sector manifests; TOI, ExoFOP TOI, CTOI, planets, EB and TCE snapshots
python targets/select_targets.py fetch-tic           # TIC rows (Tmag, Teff, radius, contamination, ...) for the candidate pool, cached
python targets/select_targets.py select --pilot 10   # targets_download.csv, download_manifest.csv, excluded_targets.csv, report
python targets/download.py --pilot                   # pilot batch, then: python targets/download.py --workers 4
python targets/control_sample.py --source <labeled cache> --per-bin 50 --copy          # control group (confirmed hosts + FP-only hosts)
python targets/control_sample.py --source <labeled cache> --tics dev.json --label development --copy
```

`--source` for `control_sample.py` is a directory with `catalogs/toi.json`,
`catalogs/known.json` and `curves/<tic>/sector-N.npz` of labeled stars (the groups
already copied into `data/` are listed in `control_targets.csv` and
`development_targets.csv`, with provenance in `*_sources.json`).

Selection rules, exclusion counts and catalog limitations are written to
`selection_report.md` and `catalog_sources.json`. Absence from the catalogs is
not evidence that a star was never searched. `tic_cache.jsonl` and `data/` are
not versioned.
