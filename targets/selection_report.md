# Target selection report (2026-11-06T00:14:23.363447+00:00)

Stars with SPOC 120 s light curves in 107 sector manifests: 592556

| Step | Stars |
|---|---:|
| available priority A | 3577 |
| available priority B | 6661 |
| candidates after criteria | 10238 |
| excluded: CTOI | 1986 |
| excluded: EB | 4048 |
| excluded: TCE | 56590 |
| excluded: TIC disposition ARTIFACT | 8 |
| excluded: TIC disposition DUPLICATE | 178 |
| excluded: TOI | 6698 |
| excluded: Tmag outside 8-12 | 93427 |
| excluded: contamination ratio > 0.2 | 11164 |
| excluded: fewer than 2 sectors | 278678 |
| excluded: fewer than 8 sectors within one season | 126485 |
| excluded: no TIC row | 240 |
| excluded: planet | 2816 |

| Priority | Available | Selected |
|---|---:|---:|
| A | 3577 | 500 |
| B | 6661 | 500 |

Selected: 1000 stars, 8000 sector products (largest group with gaps of at most 13 sectors, at most 8 sectors), ~15.6 GB; pilot batch: 10 stars.

Limitations:
- TOI/CTOI/planet/TCE/EB catalogs are the local snapshots listed in catalog_snapshot; a star absent from them may still have been searched by SPOC, QLP or published work.
- The TESS EB catalog snapshot covers sectors 1-26 (FFI based); later eclipsing binaries are not excluded by it.
- SPOC TCE files cover the DV runs listed on the MAST bulk page at snapshot time.
- Baseline, observed days and point counts are estimates from sector start dates; exact coverage is measured after download.
- Contamination uses the TIC contratio field; a missing value is not treated as low contamination but is not excluded either.
