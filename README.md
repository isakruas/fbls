# tess-gpu-search

Author: Isak Paulo de Andrade Ruas, NROLABS.

Standalone CUDA transit-search package with a Python API and an NPZ/JSON CLI.
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
gpu_table = result.table
json_mapping = result.to_dict()
```

Input arrays may be NumPy arrays, numeric sequences or CUDA tensors. The output
table remains on the GPU. Conversion to Python occurs when candidates or JSON
are requested. Call `search_prepared(time_bjd, relative_flux, flux_err,
config=config)` to consume an existing photometric preparation stage.

The table has five float64 columns: `period_days`, `epoch_bjd`, `duration_days`,
`snr`, and `depth`. It contains one maximum for each period reaching the configured
threshold, before peak suppression. Depth is fractional relative flux. JSON peak
durations are in hours and include `depth_ppm`.

The default duration bank contains 13 widths from 0.5 to approximately 7.28 hours.
Shapes use ingress fractions 0 and 0.2 with four nested layers. The period grid
obeys `baseline * (P_next / P - 1) <= min_duration / period_oversample`.
Peak reporting suppresses periods within relative tolerance 0.002 and epochs
within one selected transit duration. It does not mask observations or rerun
the search after reporting a peak.

`noise_degree=0` uses diagonal precision and the fused period-maximization
kernel. Degrees 1–4 fit localized polynomial covariance modes. The duration
noise scale is estimated from standardized residual means; set
`correlated_noise=False` to use unit scales. The configured threshold applies
to the one-sided amplitude statistic. Choose it for the pipeline's operating
point; it is not a planet probability.

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
