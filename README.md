# tess-gpu-search

Author: Isak Paulo de Andrade Ruas, NROLABS.

Standalone CUDA eclipse-model search with a Python API and an NPZ/JSON CLI.
Scientific array operations use float64 on the GPU. Runtime dependencies are
NumPy, CUDA-enabled PyTorch 2.7.1 and Triton 3.3.1. The package requires Linux
and an NVIDIA GPU. Python 3.10 or newer is required.

## Install and execute

Install CUDA-enabled PyTorch for the host before installing this package:

```bash
python -m pip install /path/to/release
tess-search --input curve.npz --output detections.json --save-table periods.npz
```

The module entry point is equivalent:

```bash
python -m tess_search --input curve.npz --pmin 0.5 --pmax 10 --threshold 7.1
```

With runtime dependencies already installed, the source package also runs
directly from this directory using `python -m tess_search`. JSON is written to
standard output unless `--output` specifies a file. Diagnostics use standard
error. Exit codes are 0 for success, 1 for runtime failure and 2 for invalid input.
Use `--help` for available parameters.

## Input contract

NPZ files must contain one-dimensional arrays of equal length:

| Column | Meaning |
|---|---|
| `time_bjd` | Absolute observation times, in BJD days |
| `flux` | Mission flux with a positive sector median |
| `flux_err` | Optional positive per-observation errors, in flux units |
| `sector` | Optional integer scalar or observation-vector normalization labels |

Raw preparation performs median normalization within each sector, weighted
time binning, robust detrending and positive-outlier clipping. Negative dips
are retained. Nonfinite observations and nonpositive errors are removed.

Multiple files can be passed after `--input`. Each file/sector pair is
normalized independently. Error arrays must be present in all files or absent
from all files. Time sorting is performed on the GPU after concatenation.

For prepared input, use `--prepared`: `flux` must then be a detrended fractional
residual with baseline zero, and `flux_err` is required in those same fractional
units. Prepared input is sorted and filtered; raw binning, detrending and clipping
are not applied.

## Python integration

```python
from tess_search import SearchConfig, search_transits

config = SearchConfig(
    period_min=0.5,
    period_max=10.0,
    durations_hours=(0.5, 1.0, 2.0, 4.0),
    threshold=7.1,
    max_candidates=10,
)
result = search_transits(time_bjd, flux, flux_err, sector, config=config)
peaks = result.candidates()
signal_peaks = result.candidates(ranking="signal")
gpu_table = result.table
json_mapping = result.to_dict()
```

Input arrays may be NumPy arrays, numeric sequences or CUDA tensors. The output
table remains on the GPU. Conversion to Python occurs when candidates or JSON
are requested. Call `search_prepared(time_bjd, relative_flux, flux_err,
config=config)` to consume an existing photometric preparation stage.

The first five float64 table columns are `period_days`, `epoch_bjd`, `duration_days`,
`snr`, and `depth`. The remaining columns are `planet_log_odds`,
`log_bf_planet_noise`, `log_bf_binary_noise`, `binary_secondary_ratio`,
`binary_secondary_phase`, `binary_odd_even_ratio`, `binary_period_days`,
`ingress_fraction`, and `selection_flags`. Depth is fractional relative flux.
JSON peak durations are in hours and include `depth_ppm`.

Each qualifying period retains its planetary MAP ephemeris and its primary-amplitude
maximum, before peak suppression. Identical maxima share a row. Selection flags
are 1 for the planetary MAP ephemeris, 2 for the amplitude maximum, and 3 for both.
The JSON schema is version 2: `candidates` first selects resolved peaks with
`planet_log_odds >= planet_log_odds_min` and orders them by planet/noise evidence.
The default boundary is zero. Remaining alternatives follow, ordered by model
odds; they are not hard-rejected. `model_preferred` records this decision.
Separating model preference from detection strength prevents saturated class
odds from promoting a weak period above a strongly detected planet.
`signal_candidates` independently ranks amplitude maxima. Each list has its own
`max_candidates` budget. All retained rows are available through `table` or
`--save-table`; neither capped peak list guarantees complete candidate retention.

The default duration bank contains 13 widths from 0.5 to approximately 7.28 hours.
Shapes use ingress fractions 0, 0.2 and 0.45 with four nested layers. The period grid
obeys `baseline * (P_next / P - 1) <= min_duration / period_oversample`.
Peak reporting suppresses periods within relative tolerance 0.002 and epochs
within one selected transit duration. It does not mask observations or rerun
the search after reporting a peak.
Before model-odds ranking, planetary ephemerides must be local maxima of planet
evidence within a drift neighborhood. By default the relative period radius is
twice the maximum configured usable duration divided by the observing baseline;
only ephemerides separated by at most their summed durations in phase compete.
This reports the fitted period of a resolved peak. Set `peak_drift_durations=0`
or `--peak-drift-durations 0` to disable this grouping. Closely spaced signals
remain in the full table even when peak reporting merges their ephemerides.

`noise_degree=0` uses diagonal precision. Degrees 1–4 fit localized polynomial
covariance modes. The comparison kernel folds precision-weighted sufficient
statistics and covariance-mode projections over twice each trial period.
An amplitude gate on the identical bank skips periods unable to reach the
reporting threshold. Every phase, duration and shape of each remaining period
contributes to both model evidences, including templates below the amplitude
threshold. The gate does not fix the epoch, duration or shape for comparison.
Nested interval sums retain only tile reductions and ephemeris winners. The duration
noise scale is estimated from standardized residual means; set
`correlated_noise=False` to use unit scales. The configured threshold applies
to the one-sided primary amplitude statistic in both reductions. Choose it for
the pipeline's operating point; it is not a planet probability.
`--signal-only` / `model_comparison=False` disables evidence calculation and
retains amplitude maxima.

## Conditional eclipse evidence

The comparison evaluates three hypotheses: zero eclipse amplitude, a planetary
eclipse surrogate and a binary eclipse surrogate. For each fixed template `h`,
the deficit data satisfy `y = baseline + depth * h + noise`. A constant baseline
is eliminated jointly under the supplied precision matrix `W = inverse(C)`:

```text
K = W - (W 1)(W 1).T / (1.T W 1)
a = h.T K y / rho_duration**2
q = h.T K h / rho_duration**2
v = 1 + tau**2 q
z = tau a / sqrt(v)
log Z(h, tau)/Z_noise = -log(v)/2 + z**2/2 + log(2 Phi(z))
```

`tau` is the scale of a normalized positive half-normal depth prior. `Phi` is
the standard normal CDF; a negative-tail expansion prevents numerical underflow.
Evidence includes primary/secondary covariance through the shared baseline and
noise model. Scenario evidence is summed with normalized mixture masses.
When duration noise scales differ, the null is normalized consistently:

```text
E0 = y.T K y
log L_noise(D) = constant - ((N-1) log(rho_D**2) + E0/rho_D**2)/2
Z_model(P)/Z_noise(P) = sum_ephemeris prior(ephemeris) L_noise(D) BF_model(ephemeris)
                       / sum_D prior(D) L_noise(D)
```

The fitted scales are held fixed in this conditional model. With unit scales
the null terms cancel and the expression is the prior-weighted mean of the
template evidence ratios.

Both models allow secondary-to-primary depth ratios `[0, .01, .1, .5, 1]` at
phases `[.25, .5, .75]` relative to the primary. Planet ratio masses are
`[.75, .15, .08, .015, .005]`; binary ratio masses are
`[.10, .10, .20, .25, .35]`. Phase masses are `[.1, .8, .1]` for both models.
Half the binary mass uses these secondary scenarios. The other half uses an
orbital period of `2P` and alternating observed eclipse depth ratios
`[.25, .5, .75, 1, 4/3, 2, 4]`, with uniform masses. Each binary family mixes
the ordinary and diluted depth scales equally. Binary diagnostics report the
largest weighted component at the reported ephemeris; the period score sums
all components and ephemerides.

Shape masses are normalized separately over the configured ingress bank:
`exp(-4 * ingress)` for planets and `exp(2 * ingress)` for binaries. These are
explicit surrogate assumptions, not fitted occurrence distributions. Both
models retain grazing profiles and a zero secondary. Unobserved secondary or
parity intervals contribute no fabricated measurements. The fixed phase/ratio
bank is a finite approximation, not continuous eccentric-orbit inference.

Default hypothesis masses are `.01` planet, `.01` binary and `.98` noise.
For each period, evidence is integrated over the entire finite bank, with
uniform phase and duration masses and the normalized shape masses. This gives
the binary hypothesis independent access to every phase, duration and shape.
The planetary MAP ephemeris is reported separately from the integrated score.
Only ephemerides reaching the amplitude threshold qualify for reporting.
The period ranking statistic is:

```text
Z_model(P) = sum_phase,duration,shape prior(phase,duration,shape) * Z_model(ephemeris)
planet_log_odds(P) = log(pi_planet * Z_planet(P))
                     - log(pi_noise * Z_noise + pi_binary * Z_binary(P))
```

The integration uses upper bounds from nonnegative two-amplitude fits to
primary/secondary and even/odd subspaces. A positive-amplitude component has
`log BF <= best_positive_cone_chi2_improvement/2`. An initial full scenario fit
at one incumbent ephemeris supplies lower bounds without restricting the bank.
The default `evidence_rtol=1e-12`
budgets omitted positive contributions across the entire bank; ill-conditioned
subspace bounds bypass pruning, with numerical slack on the remaining bounds.
Set `--evidence-rtol 0` to evaluate every component. The exact-arithmetic relative
omission bound is `rtol`; floating-point agreement must also be verified for the
chosen covariance conditioning. Near model/peak ties can change ordering within
numerical precision. No trial-by-observation result matrix is allocated.

Binary diagnostics can be unresolved when a negligible local scenario is skipped
while the period evidence remains bounded. Such fields are NaN in the GPU/NPZ
table and null in JSON; `binary_diagnostics_complete` identifies these rows.
Zero tolerance evaluates all local scenarios for complete diagnostics.

Default depth scales are `.01` planet, `.1` ordinary binary and `.01` diluted
binary. API and CLI parameters expose these masses and scales. With
`stellar_radius_solar`, the planet scale becomes
`(.10045 * planet_radius_scale_rj / stellar_radius_solar)**2`.
`target_flux_fraction` multiplies target planet and target binary depth scales;
the diluted-background scale remains independent. This fraction must describe
the supplied flux. Leave it at one when contamination has already been corrected.
The radius setting conditions an approximate depth prior; it does not implement
a full stellar atmosphere or orbital population model.

These scores are conditional on the fitted noise and discretized model bank.
They are not calibrated planet probabilities or whole-search false-alarm rates.
The finite surrogate does not model limb darkening, ellipsoidal modulation,
continuous eccentricity, centroid motion, stellar temperature or magnitude.
Photometrically indistinguishable binaries remain ambiguous. Injection recovery,
binary contamination and false-alarm calibration must be measured for the
pipeline's data distribution and candidate budget.

## Explicit covariance input

`NoiseModel`, `search_templates` and `period_grid` expose the numerical layer:

```python
from tess_search import NoiseModel, search_templates

model = NoiseModel(relative_time, deficit, variance, modes, eigenvalues, origin=first_bjd)
result = search_templates(model, periods_days, durations_days, reduce="period")
```

These arrays must be CUDA-compatible float64 inputs. Relative times must be
sorted and nonnegative. Modes have shape `(observations, rank)` and are
orthonormal in diagonal-whitened coordinates; eigenvalues are nonnegative.
The covariance is `sqrt(D) @ (I + U @ diag(eigenvalues) @ U.T) @ sqrt(D)`.
With `reduce="all"`, the result retains all templates reaching the threshold.
Candidate-buffer overflow reduces the period batch and retries the same bank;
overflow for a single period raises an error without truncating output.

No catalogs, mission-cache layout, stellar tables or external service calls
are required. All runtime modules and the CLI are included in this directory.
