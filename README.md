# tess-gpu-search

Author: Isak Paulo de Andrade Ruas, NROLABS.

Standalone CUDA eclipse-model search with a Python API and an NPZ/JSON CLI.
Scientific array operations use float64 on the GPU. Runtime dependencies are
NumPy, CUDA-enabled PyTorch 2.7.1 and Triton 3.3.1. The package requires Linux
and an NVIDIA GPU. Python 3.10 or newer is required.

## Install and execute

Create the repository's own environment, install CUDA-enabled PyTorch for the
host, then the package (the `targets` extra adds astropy and lightkurve for the
download tools, `dev` adds pyflakes):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -e '.[targets,dev]'
.venv/bin/tess-search --input curve.npz --output detections.json --save-table periods.npz
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
| `centroid` | Optional `(observations, 2)` centroid coordinates in detector pixels |
| `centroid_err` | Optional coordinate uncertainties with the same shape |

The paired columns `centr1`, `centr2` are also accepted, with optional
`centr1_err`, `centr2_err`. Missing coordinates remain missing measurements;
they do not remove photometric samples. Mixed files without astrometry are
filled with zero astrometric precision. When coordinate errors are absent,
their scale is estimated from robust adjacent differences within each group.

Raw preparation performs median normalization within each sector, weighted
time binning, robust detrending and positive-outlier clipping. Negative dips
are retained. Nonfinite observations and nonpositive errors are removed.

Detrending is organized in duration families. Each configured duration `D`
receives a biweight window `clamp(window_factor * D, window_days, window_max_days)`
(defaults 3, 0.4 and 3.5 days); durations sharing a window form one family with
its own detrended residual, duration noise scales and period grid. The grid
resolution follows the family's shortest duration, so long-duration templates
are never scanned on the finest phase grid. Observations closer than
`edge_trim_windows` (default 0.5) windows to a segment edge are dropped in each
family, because the one-sided biweight there cannot follow the ramps that
follow data gaps. Periodic peaks are searched from
`period_min` to `min(period_max, baseline)`; longer periods cannot show two
events and are covered by the unfolded single-event scan.

Duration-dependent noise estimates require at least 20 sufficiently populated
time bins. Sparse durations inherit the nearest supported noise scale, increased
by the square root of the duration ratio when extrapolating to longer durations
and floored by their own measured scatter. With no supported duration, the
reference is the unbinned standardized residual scatter under a coherent-noise
assumption. This fallback is a conservative working model, not a confidence
bound. `correlated_noise_estimation` reports bin counts, direct-estimate flags
and reference durations; sparse sampling never automatically resets the scale
to unit white noise.

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
    period_max=100.0,
    durations_hours=(0.5, 1.0, 2.0, 4.0),
    threshold=7.1,
    max_candidates=10,
    passes=3,
)
result = search_transits(time_bjd, flux, flux_err, sector, config=config,
                         centroid=centroid_pixels, centroid_err=centroid_errors)
peaks = result.candidates()
signal_peaks = result.candidates(ranking="signal")
events = result.single_events
gpu_table = result.table
json_mapping = result.to_dict()
```

The search runs in passes. Each pass scans every family, then resolves peaks
greedily from the per-period maxima: a row is accepted when at least
`min_events` transit epochs hold three or more observations, fewer than half
of its in-transit observations are shared with accepted ephemerides, its
box-template amplitude statistic recomputed without the observations of
accepted ephemerides reaches the threshold scaled by the square root of its
unshared fraction (at least half the threshold), and its statistic without
its strongest single event keeps at least half of the threshold. The last rule
rejects isolated events folded with an arbitrary long period. Remaining rows
are rescored the same way, which removes period aliases and harmonics of
accepted signals without masking. After a pass, all accepted ephemerides are masked (1.5
durations) in every family and the next pass searches the residual data; this
exposes signals whose period cells were occupied by a stronger signal. Up to
`passes` passes run; a pass without an accepted peak ends the loop. Peaks report
`snr` (isolated statistic), `snr_folded` (original per-period maximum),
`snr_without_strongest_event`, `events` (sampled epochs) and `pass`. Model evidence is evaluated only for the
`evidence_candidates` strongest peaks, each with the other accepted
ephemerides masked, so one peak's evidence describes that signal alone.

With `single_events=True` a final scan folds each family at one trial period
longer than the baseline. Templates above threshold that do not coincide with
accepted periodic ephemerides are reported in `single_events` with epoch,
duration, depth and point count; their period is unconstrained.

Input arrays may be NumPy arrays, numeric sequences or CUDA tensors. The output
table remains on the GPU. Conversion to Python occurs when candidates or JSON
are requested. Call `search_prepared(time_bjd, relative_flux, flux_err,
config=config)` to consume an existing photometric preparation stage.

The table holds the first-pass per-period maxima of every family. The first
five float64 columns are `period_days`, `epoch_bjd`, `duration_days`, `snr`, and
`depth`. The remaining columns are `planet_log_odds`, `log_bf_planet_noise`,
`log_bf_binary_noise`, `binary_secondary_ratio`, `binary_secondary_phase`,
`binary_odd_even_ratio`, `binary_period_days`, `ingress_fraction`, and
`selection_flags`; they are NaN except for rows that received evidence
(`selection_flags` 3). Depth is fractional relative flux. JSON peak durations
are in hours and include `depth_ppm`.

The JSON schema is version 3: `candidates` lists resolved peaks with
`planet_log_odds >= planet_log_odds_min` first, ordered by planet/noise
evidence, followed by the remaining peaks ordered by model odds; `model_preferred`
records this decision. `signal_candidates` lists the same peaks ordered by the
isolated amplitude statistic. `single_events` follows. Each list is capped by
`max_candidates`.

The default duration bank contains 13 widths from 0.5 to approximately 7.28 hours.
Shapes use ingress fractions 0, 0.2 and 0.45 with four nested layers. In extended
mode, quadratic limb-darkened disk profiles use weighted nested intervals. The period grid
obeys `baseline * (P_next / P - 1) <= min_duration / period_oversample`.
In the fused diagonal scan the box shape (ingress 0, first in
`ingress_fractions`) is evaluated at every phase bin; the trapezoidal shapes are
evaluated only in phase tiles where some box statistic reaches
`shape_gate_fraction * threshold` (default 0.6). A box recovers at least ~87%
of the amplitude of the most V-shaped bank template, so the gate removes no
template able to reach the threshold while skipping ~80% of the interval sums
in noise-dominated phase ranges. `shape_gate_fraction=0` evaluates every shape
everywhere. An optional coarse period prescan (`prescan_factor` > 1 with
`prescan_fraction`) scans a phase grid coarser by that factor first and keeps
periods whose maximum reaches the fraction of the threshold; it is disabled by
default because the per-period noise maximum at long periods approaches the
threshold and little is pruned. Template ranking inside a phase tile uses float32 cross ratios; the winner and
any near tie are recomputed exactly in float64, and reported scores and depths
are float64 throughout. Folded fields are stored interleaved per phase bin with
the bins repeated once, so interval sums are two contiguous loads without wrap
handling.

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

The earlier surrogate bank (`extended_physics=False`) allows secondary-to-primary depth ratios `[0, .01, .1, .5, 1]` at
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
uniform phase masses and normalized duration/orbit and shape masses. This gives
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

## Astrometry, stellar parameters and orbital geometry

Version 0.3 provides the opt-in experimental setting `extended_physics=True`
or `--extended-physics`. Its priors and candidate rankings require independent
calibration; the earlier surrogate remains the default. The photometric and
astrometric channels are evaluated inside each scenario before period
marginalization. Astrometric errors are conditionally independent of flux and
of the other coordinate. A separate constant offset for each file/sector is
eliminated exactly in each coordinate. A signed Gaussian amplitude describes
the centroid response to the same eclipse profile. Its default scale is
`.001` pixel for a target source and `.05` pixel for a displaced source.
The diluted family includes both unresolved and displaced-source components.
These are explicit source-location priors; correlated instrumental errors
require appropriate preparation and validation.

Ellipsoidal variability uses sine/cosine coefficients at twice the orbital
frequency with normalized Gaussian priors. The alternating `2P` binary family
uses the corresponding fundamental frequency of the trial `P`. Both hypotheses
allow variability. Its covariance with the eclipse templates and the observed
window is retained through a two-dimensional Gaussian integral; independent
SNR squares are not added. The diluted family includes weak and broad
modulation components, so an undetected ellipsoidal signal is not a binary veto.

Supply `stellar_radius_solar`, `stellar_mass_solar`, `stellar_teff` and, when
available, `stellar_logg`. If mass is absent, radius and cgs surface gravity
give `M = g R**2/G`. Kepler's law supplies `a/Rstar`; the duration prior is a
broad Student-t compatibility distribution in log duration. It includes a
transit impact-parameter reference and eccentric orbital velocity. The
binary prior retains a broad background-source duration component, since a
contaminating star need not share the target's density. The
ellipsoidal reference scale is `alpha * q/(1+q) * (Rstar/a_primary)**3`, with
`alpha=1.5`, a planet mass scale in Jupiter masses and binary mass ratio `.5`.
Without stellar context the configurable amplitude scales remain available.

Temperature supplies a reference planetary occultation ratio from photon-counted
blackbody emission integrated over a flat 600–1000 nm response, plus reflected
light at geometric albedo `.1`. This replaces the `.01` secondary-ratio node
when radius, mass and temperature are available. It is a reference component
alongside the zero and broad occultation components, not a measured atmosphere
or a calibrated mission throughput model. `stellar_tmag` is retained as context;
photometric precision uses supplied errors or measured scatter, rather than an
additional magnitude penalty.

`limb_darkening=(u1,u2)` specifies the quadratic intensity law. Defaults are
`(.3,.2)`; these values are not inferred from temperature. The occulted stellar
disk is integrated with radial Gaussian quadrature, then sampled into the
configured staircase bank. The representative planet radius scale and stellar
radius determine the reference radius ratio; ingress controls a reference
impact parameter. Depth is still marginalized independently. Increase
`shape_layers` to improve this finite-profile approximation. The secondary
shares the reference profile; the model is not a full two-star surface-brightness
or stellar-evolution calculation.

The orbital prior includes a circular component and a continuous eccentricity /
periastron component integrated numerically. Conjunction mean anomalies set the
secondary phase. The central, edge-on duration ratio is
`(1+e sin(omega))/(1-e sin(omega))`. Overlapping eclipse intervals are excluded
and the covered prior masses are normalized. Defaults use two eccentricity
nodes and four periastron nodes, plus the circular node. These are coarse
quadratures: increase `eccentricity_nodes` and `periastron_nodes` and check
convergence for inference. `eccentricity_max` defaults to `.8`. The planetary
eccentric component has density proportional to `(1-e)**2`; the binary component
is uniform, with half of each prior assigned to the circular component.

These settings are available as CLI arguments, including `--stellar-logg`,
`--stellar-teff`, `--limb-darkening`, `--eccentricity-nodes` and
`--periastron-nodes`. `--no-centroids` skips astrometry;
`--no-ellipsoidal` disables harmonic integration; `--no-extended-physics`
selects the earlier surrogate bank. Raw preparation detrends astrometry as
well as flux. A short biweight window can attenuate orbital modulation: use
prepared data that preserve the harmonics, or choose an appropriate window.
The physical kernel partitions duration and profile work across GPU programs.
Register and spill counts are reported as `kernel_registers` and
`kernel_spill_slots`; complex configurations can still spill registers. The
version 0.2 timing is not a throughput measurement for these added channels.

These scores are conditional on the fitted noise and discretized model bank.
They are not calibrated planet probabilities or whole-search false-alarm rates.
The finite conditional model does not infer stellar atmosphere/evolution grids,
source catalogs, inclination-dependent eccentric conjunction corrections or
instrumental flux/centroid cross-covariance.
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
