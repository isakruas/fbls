# Target selection report (2026-11-06T00:57:00.310462+00:00)

Stars with SPOC 120 s light curves in 107 sector manifests: 592556

| Step | Stars |
|---|---:|
| available priority A | 24792 |
| available priority B | 32420 |
| candidates after criteria | 57212 |
| excluded: CTOI | 1986 |
| excluded: EB | 4048 |
| excluded: TCE | 56590 |
| excluded: TIC disposition ARTIFACT | 8 |
| excluded: TIC disposition DUPLICATE | 178 |
| excluded: TOI | 6698 |
| excluded: Tmag outside 8-12 | 93427 |
| excluded: contamination ratio > 0.2 | 11164 |
| excluded: fewer than 2 sectors | 278678 |
| excluded: no TIC row | 240 |
| excluded: planet | 2816 |
| excluded: search window not 2-3 consecutive sectors within one season | 79511 |

| Priority | Available | Selected |
|---|---:|---:|
| A | 24792 | 500 |
| B | 32420 | 500 |

Selected: 1000 stars, 2181 sector products (largest group with gaps of at most 13 sectors, at most 8 sectors), ~4.3 GB; pilot batch: 10 stars.

Limitations:
- TOI/CTOI/planet/TCE/EB catalogs are the local snapshots listed in catalog_snapshot; a star absent from them may still have been searched by SPOC, QLP or published work.
- The TESS EB catalog snapshot covers sectors 1-26 (FFI based); later eclipsing binaries are not excluded by it.
- SPOC TCE files cover the DV runs listed on the MAST bulk page at snapshot time.
- Baseline, observed days and point counts are estimates from sector start dates; exact coverage is measured after download.
- Contamination uses the TIC contratio field; a missing value is not treated as low contamination but is not excluded either.
