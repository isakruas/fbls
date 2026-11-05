# Target selection and download

Tools to build a reproducible sample of TESS 120 s targets without catalogued
signals and to download their SPOC light curves for the blind search.

```bash
python targets/select_targets.py --source /path/to/snapshots fetch-tic   # TIC rows for the candidate pool (cached)
python targets/select_targets.py --source /path/to/snapshots select      # targets_download.csv, download_manifest.csv, report
python targets/download.py --pilot                                       # pilot batch, then the full manifest
```

```bash
python targets/control_sample.py --source /path/to/snapshots --per-bin 50 --exclude dev_tics.json --copy
```

`control_sample.py` draws confirmed-planet hosts (TOI CP/KP and published transiting
planets) from an existing cache, stratified by the longest planet period, excluding
development stars, and copies their files into `data/` as the control group
(`control_targets.csv`, `control_sources.json`).

`--source` points at local snapshots of the official products: `manifests/sector-N.sh`
(MAST bulk download scripts, one per sector), `catalogs/toi.json`, `catalogs/known.json`
(CTOIs, published planets, TESS EBs, SPOC TCEs), `catalogs/exofop_toi.csv`, and the
`curves/` cache of stars already used. Downloads go to `data/` (ignored by git) as
FITS plus NPZ/JSON in the same layout the search CLI reads.

Outputs written next to the scripts: `targets_download.csv`, `download_manifest.csv`,
`excluded_targets.csv`, `catalog_sources.json`, `selection_report.md`. The TIC cache
`tic_cache.jsonl` is not versioned.
