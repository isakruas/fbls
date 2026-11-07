# FBLS: software specification

FBLS stands for Family Box Least Squares: the box least squares statistic is
evaluated per family of transit durations, each with its own detrending
window and period grid.

## Purpose

FBLS searches TESS light curves for periodic box-shaped transits with a
multi-pass box least squares statistic evaluated on the GPU. The crate is a
single binary: preparation on the host, template scans as CUDA kernels.

## Inputs

NPZ files with columns `time_bjd`, `flux`, optional `flux_err`, `sector`
(scalar or vector), optional `centroid`/`centr1`,`centr2`. Several files are
concatenated; each (file, sector) pair is a normalization group.

## Pipeline

1. **Preparation** (host, f64). Per-group median normalization, 10-minute
   inverse-variance binning within groups, error inflation by the robust
   scatter of standardized adjacent differences, one biweight-detrended curve
   per duration family (window = clamp(3 D, 0.4, 3.5) days), positive clipping
   at 4 sigma, edge trimming of half a window. Each family then gets a
   diagonal noise model and a correlated-noise factor per duration.
2. **Signal scan** (GPU). For each family and each octave band of its period
   grid: fold the four precision fields into `nb` phase bins per period,
   build doubled prefix sums, and rank every (phase, duration, shape) staircase
   template by the one-sided generalized least-squares amplitude statistic.
   Tiles of 128 phase bins keep one winner; the best tile per period is
   reported when its score reaches the threshold.
3. **Peak selection** (host). Greedy acceptance with event-overlap rescoring,
   masking of accepted ephemerides, up to three passes.
4. **Model comparison** (GPU). For the strongest peaks, with the other
   accepted ephemerides masked, the joint planet / binary / noise evidence at
   the peak period: statistics folded over two periods keep cycle parity;
   each family integrates a half-normal eclipse depth and sums discrete
   scenarios (secondary eclipses at quarter phases, alternating primaries at
   twice the period); phase, duration and shape are marginalized. Reports
   `planet_log_odds`, the two Bayes factors against noise and the binary
   diagnostics (secondary ratio and phase, odd/even ratio, binary period).
5. **Single events** (GPU). Template scan at one trial period longer than the
   baseline.
6. **Output**. JSON schema 3 with candidates (model-preferred first), signal
   candidates (by amplitude), single events and statistics.

## Survey layer

The survey runs in stages over a star list; each stage consumes the previous
stage's files under `data/survey/` and skips work already done. Catalog
information is used only after the search.

1. **fetch**. SPOC 2-minute light-curve products of a TIC are found through
   the MAST CAOM filtered service (target name, TESS, timeseries, provenance
   SPOC, 120 s exposure) and downloaded from the public mirror with the MAST
   download endpoint as fallback. Each product is verified by reading it,
   linked as `data/curves/<tic>/sector-N.fits` and described by a provenance
   sidecar (`sector-N.json`: crowding, stellar parameters, target pixel from
   the aperture WCS, SHA-256, point counts). Existing NPZ sector files are
   read unchanged.
2. **FITS input**. SPOC light curves are read directly (TIME + BJDREF,
   PDCSAP_FLUX, PDCSAP_FLUX_ERR, MOM_CENTR1/2, QUALITY with the default
   bitmask 17087). `search` and `batch` accept `.fits` and `.npz` inputs
   interchangeably.
3. **search**. The detector on the search window of sectors (the largest
   group of sectors less than 13 apart, reduced to the 8 spanning the shortest
   time) with 19 durations up to 27.7 h and periods to 100 d; writes the
   peaks, single events and the detrended residuals of the families the peaks
   use (`<tic>.residuals.json`, no binning, on the original cadences).
4. **vet**. Per peak: catalog cross-match (TOI table, CTOIs, published
   planets, eclipsing binaries, SPOC TCEs: same period or harmonic 1, 2, 1/2,
   3, 1/3 within 1% and the same phase while the catalog ephemeris is
   predictive), duplicate detection against stronger peaks, and the vetting
   metrics on the folded residuals with the other signals masked: depth and
   SNR against the out-of-transit noise at the transit time scale, odd/even
   depths, secondary eclipse search at every phase, trapezoid fit and
   circular-orbit geometry with Monte Carlo errors, duration versus the TIC
   stellar density, centroid shift versus the target position, crowding, and
   the flux tests of the Kepler Robovetter as adapted for TESS (MES, SHP,
   ModShift uniqueness MS1-MS6 with Fred and the FA1/FA2 thresholds, chases,
   rubble, gaps, CHI, DMM, max SES / MES, SWEET sinusoid on the detrended and
   on the raw flux, trapezoid-versus-line AIC, asymmetry, V shape). Flags are
   `fail` or `warn`; the verdict fails when any flag fails. Single events are
   matched to catalog transit times and carried with their detector metrics.
5. **pixel**. For passing periodic signals, target pixel files of up to three
   sectors with transits are fetched on demand; the mean out-of-transit minus
   in-transit image over the covered transits is fitted with a Gaussian plus
   constant, and the source offset from the target (same stamp WCS),
   combined over sectors, fails above 4 sigma and half a pixel.
6. **publish**. Tiers (candidate, unconfirmed, inconclusive, rejected,
   duplicate) from the vetting verdict and the model preference;
   `public/candidates.csv` and `public/ctoi.csv` over every
   searched star; `public/<day>/` with the stars first published that day
   and, for each CTOI row, the panel data as CSV (`light_curve.csv`,
   `folded.csv` with the fitted trapezoid and parity, `transits.csv`,
   `event.json`).
7. **refine**. For each CTOI star: every sector fetched, searched, vetted and
   pixel-tested on the full baseline; `public/<day>/refine/<tic>/ctoi.json`
   records whether each survey candidate is recovered and its refined CTOI
   row.

## Numerical conventions

- Fold bins: `floor(t * nb / P) mod nb`, with t relative to the first observation.
- Template width `k = max(1, floor(D * nb / P + 0.5))`; templates need
  `k < nb / 2`, between 3 and n - 3 observations, and positive numerator.
- Ranking uses the f32 merit `a^2 / (rho^2 (c norm - hp^2))`; the winner is
  recomputed in f64; tiles with near ties or abnormal merit ranges are
  rescanned in f64.
- Shapes: ingress fractions (0, 0.2, 0.45), four nested layers; non-box
  shapes are evaluated only in tiles where a box reaches 0.6 x threshold.
- Durations: 0.5 h x 1.25^i, i = 0..12, rounded half-to-even at 1e-6 h.
- Robust statistics use the lower middle element as the median of an even
  count.
- Mixed precision: fold and prefix sums in f64; template ranking in f32 with
  an exact f64 recomputation of each tile winner; evidence in f64.

## Performance design

- Prefix sums are stored once in f64 and once as (hi, lo) f32 pairs. The
  pre-ranking of every template reads interval sums from the pairs, which
  gives f32-accurate differences without f64 arithmetic (f64 throughput on
  consumer GPUs is 1/64 of f32); the tile winner and any near tie are
  recomputed from the f64 sums.
- Tile winners are reduced with warp shuffles; the shape gate uses a warp
  vote.
- Device buffers are allocated once per band and reused across period
  batches; the batch size follows a fold budget capped at half of the free
  device memory.
- Peak selection rescoring runs on the GPU for all stale rows at once (one
  block per row), with a lazy max-heap on the host; a star whose deep eclipse
  qualifies millions of periods selects in under a second.
- The diagonal noise model makes the baseline projection equal to the
  precision weight, so three fields (v, w, count) are folded, scanned and
  read instead of four.
- The histogram fold uses shared-memory atomics when the bins fit
  (nb <= 2048), several periods per block when the histograms fit together,
  and global f64 atomics with four periods per block otherwise; prefix sums
  use warp-shuffle scans and derive the second copy by addition. The fold is
  bound by the per-observation-per-period instruction count (phase, three
  atomics); integer fixed-point atomics and multi-period blocks were measured
  and do not change it, so the remaining lever is fewer observations.
- `--prescan F` (configuration `prescan_factor`, default 2; 1 disables it)
  scans every period first on a phase grid coarser by F and rescans on the
  full grid only the periods whose coarse maximum reaches `prescan_fraction`
  x threshold. A template misaligned by half a coarse bin keeps at least
  1 - 1 / (2 x phase_oversample / F) of its amplitude, so F = 2 with fraction
  0.7 keeps every template able to reach the threshold up to 5% noise
  fluctuations. It saves scan and prefix time, not the fold.
- `--device N` selects the GPU. `fbls batch` runs several worker threads per
  device, each with its own context and stream, so the kernels of one star
  overlap the host stages (I/O, preparation, selection, report) of the
  others; one process per node with all its GPUs is the intended cluster
  deployment.
- The phase bin of an observation is floor(t x nb / P) mod nb computed with a
  corrected f64 reciprocal instead of an integer division.

## GPU resource mapping

| Kernel        | Grid                    | Block | Memory                     |
|---------------|-------------------------|-------|----------------------------|
| fold          | (ceil(n/256), periods/4)| 256   | global f64 atomics, four periods per block (nb > 2048) |
| fold_shared   | (periods)               | 256   | shared-memory histogram (nb <= 2048) |
| cumulative    | (periods)               | 256   | warp-shuffle scans, f64 and (hi, lo) f32 outputs |
| scan_tiles    | (tiles, periods)        | 128   | shared arrays for the block reduction, template sums in registers |
| merge_tiles   | ceil(periods/256)       | 256   | one thread per period      |
| scan_all      | (tiles, 1)              | 128   | append records above threshold through an atomic counter |
| fold_two, prefix_fields, comparison_screen, joint_tiles | (tiles, 1) | 64 | evidence at one period: two-period fold, screen, pilot floors, full integration |

Batches of periods are sized to a fold budget (`max_fold_mib`), halving on
overflow, so the template bank never changes with the batch size.

## Reference data layout

```
reference/stars.json                 selected stars: TIC, TOI, ExoFOP period, epoch, depth, duration, pmax;
                                     their light curves are the ones under data/curves/<tic>/
```

## Validation

`fbls validate --stars reference/stars.json` searches every reference star and
checks that the catalog ephemeris is recovered: a candidate within 1e-3
relative of the ExoFOP period or of its 1/2, 2x or 3x alias, with depth and
duration within a factor of two. Reference stars are confirmed or known
planets from the ExoFOP TOI catalog. `tools/benchmark.py` compares recovery
and run time with astropy BoxLeastSquares, transitleastsquares and cuvarbase on
the same inputs.

## Measured performance

RTX 3050 (8 GB, 20 SMs), default configuration, period range 0.3-20 d (measured with 0.5-20 d):

| Star (sectors, observations) | Signal | Periods tested | Total time |
|---|---|---|---|
| TIC 301055848 (2, 6.6k) | 1.11 d, 400 ppm | 39k | 1.0 s |
| TIC 392476080 (2, 4.2k) | 0.67 d, 6700 ppm (every period qualifies) | 543k | 14.7 s |
| TIC 153976959 (7, 21k) | 0.69 d, 150 ppm | 1.2M | 26 s |
| TIC 402026209 (8, 23k) | 1.34 d, 27000 ppm (2M qualifying periods) | 2.1M | ~90 s |

Scan throughput: 8 x 10^9 templates/s on ordinary stars, 2.8 x 10^9 when the
shape gate opens everywhere (deep eclipses). The multi-pass search repeats the
scan once per accepted signal.
