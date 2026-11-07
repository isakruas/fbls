# fbls

FBLS (Family Box Least Squares): transit search for TESS light curves in which
the box least squares statistic is evaluated per family of transit durations, with the template
scans and the model comparison as CUDA kernels compiled by
[cuda-oxide](https://github.com/NVIDIA/cuda-rust). `docs/SPEC.md` describes
the pipeline and the numerical conventions.

## Requirements

- Linux, NVIDIA GPU (Ampere or newer), driver 580+.
- CUDA Toolkit 13.x on `PATH` (`nvcc`, `cuda.h`, libNVVM, nvJitLink).
- Clang/LLVM 21 with libclang; set `LIBCLANG_PATH` when installed outside the system paths.
- Rust 1.99.1 (rustup installs it from `rust-toolchain.toml`).
- `cargo-oxide`: `cargo +1.99.1 install --git https://github.com/NVIDIA/cuda-rust.git cargo-oxide`

## Build and run

```
cargo oxide doctor
cargo oxide build
target/release/fbls search --input star/sector-1.npz star/sector-2.npz --pmax 20 --output result.json
target/release/fbls search --input curve.npz --prepared --save-table table.csv
target/release/fbls validate --stars reference/stars.json --data data   # recovery of catalog ephemerides
target/release/fbls batch --list stars.txt --output-dir out --workers 2 --devices 0,1
target/release/fbls fetch --tic 301055848 --root data                 # SPOC light curves from MAST
target/release/fbls survey search --targets data/targets/<day>/<list>/targets_download.csv --root data
target/release/fbls survey vet --targets ... ; survey pixel --targets ... ; survey publish --targets ... --author "Name"
target/release/fbls survey refine --targets ...
cargo test --release            # host-side unit tests
```

`FBLS_PROFILE=1` adds per-kernel timings to the band statistics.

Batch mode reads one star per line (`name pmax file1 file2 ...`, `pmax` may be
`-` for the default) and runs `--workers` threads per listed device, each with
its own CUDA stream, writing `out/<name>.json` (or `out/<name>.error`). On a
cluster, run one process per node with all its GPUs in `--devices`.

## Input and output

Inputs are SPOC light-curve FITS files or NPZ files with columns `time_bjd`,
`flux`, optional `flux_err`, optional `sector` (scalar or vector). Output: JSON schema 3 with `candidates` (model-preferred
first), `signal_candidates` (by amplitude), `single_events`, `config` and
`statistics`.

## Data layout

```
data/curves/<tic>/sector-N.fits|npz   light curves, with sector-N.json provenance sidecars
data/mast/lc, data/mast/tp            SPOC products as downloaded
data/catalogs/                        TOI table, known signals (CTOI, planets, EB, TCE), ExoFOP TOI CSV
data/targets/<day>/<list>/            target lists and download manifests
data/survey/{search,vetting,pixel}/   stage outputs per star
public/                               candidate and CTOI tables, per-day folders with panel data
```

## Reference data

`reference/stars.json` lists confirmed and known TOI hosts sampled across
periods with their ExoFOP ephemerides; their light curves are the ones under
`data/curves/<tic>/`. `fbls validate` searches every star and reports whether
the catalog period (or an alias), depth and duration are recovered.
`tools/benchmark.py` runs the same stars through astropy BoxLeastSquares,
transitleastsquares and cuvarbase for sensitivity and run-time comparison.
