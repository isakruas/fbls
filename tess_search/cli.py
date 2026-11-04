"""NPZ input and atomic JSON/NPZ output for CUDA transit searches."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from . import __version__
from .api import DEFAULT_DURATIONS_HOURS, TABLE_COLUMNS, SearchConfig, search_prepared, search_transits


def _numbers(value):
    try:
        return tuple(float(x.strip()) for x in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use comma-separated numbers") from error


def read_inputs(paths):
    """Concatenate NPZ columns and retain file/sector normalization groups."""
    parts = []
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as source:
            if not {"time_bjd", "flux"}.issubset(source.files):
                raise ValueError(f"{path}: required columns are time_bjd and flux")
            t, flux = source["time_bjd"], source["flux"]
            if t.ndim != 1 or flux.shape != t.shape:
                raise ValueError(f"{path}: time_bjd and flux must be matching vectors")
            part = {"time_bjd": t, "flux": flux,
                    "file_index": np.full(len(t), index, dtype=np.int64)}
            if "flux_err" in source.files:
                part["flux_err"] = source["flux_err"]
                if part["flux_err"].shape != t.shape:
                    raise ValueError(f"{path}: flux_err must match time_bjd")
            label = source["sector"] if "sector" in source.files else np.int64(0)
            part["sector"] = np.full(len(t), label) if np.ndim(label) == 0 else label
            if part["sector"].shape != t.shape:
                raise ValueError(f"{path}: sector must be a scalar or an observation vector")
            parts.append(part)
    if len({"flux_err" in part for part in parts}) != 1:
        raise ValueError("All input files must either provide flux_err or omit it")
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def _normalization_groups(data):
    labels = torch.as_tensor(data["sector"], device="cuda")
    if not bool(torch.isfinite(labels).all() & (labels == labels.long()).all()):
        raise ValueError("Sector labels must be finite integers")
    files = torch.as_tensor(data["file_index"], device="cuda", dtype=torch.int64)
    _, inverse = torch.unique(torch.stack((files, labels.long()), dim=1), dim=0, return_inverse=True)
    return inverse


def _write_json(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if path == "-":
        sys.stdout.write(text)
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(destination)


def _write_table(path, table):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    values = table.detach().cpu().numpy()
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **{name: values[:, i] for i, name in enumerate(TABLE_COLUMNS)})
    temporary.replace(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tess-search", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="One or more NPZ light curves")
    parser.add_argument("--output", default="-", help="JSON destination; '-' writes to standard output")
    parser.add_argument("--save-table", type=Path, help="Save all qualifying period maxima as float64 NPZ columns")
    parser.add_argument("--prepared", action="store_true", help="Input flux is detrended fractional flux with baseline zero")
    parser.add_argument("--pmin", type=float, default=.5, help="Minimum period in days")
    parser.add_argument("--pmax", type=float, default=10., help="Maximum period in days")
    parser.add_argument("--durations-hours", type=_numbers, default=DEFAULT_DURATIONS_HOURS)
    parser.add_argument("--threshold", type=float, default=7.1)
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of reported peaks")
    parser.add_argument("--bin-minutes", type=float, default=10., help="Raw-input bin width; zero preserves cadence")
    parser.add_argument("--window-days", type=float, default=.75, help="Raw-input detrending window")
    parser.add_argument("--clip-sigma", type=float, default=4., help="Raw-input positive-outlier cutoff")
    parser.add_argument("--noise-degree", type=int, default=0, choices=range(5))
    parser.add_argument("--no-red-noise", action="store_true", help="Use unit duration-dependent noise scales")
    parser.add_argument("--ingress-fractions", type=_numbers, default=(0., .2))
    parser.add_argument("--shape-layers", type=int, default=4)
    parser.add_argument("--phase-oversample", type=int, default=4)
    parser.add_argument("--period-oversample", type=float, default=3.)
    parser.add_argument("--batch", type=int, help="Maximum periods processed in a fold batch")
    parser.add_argument("--capacity", type=int, default=131072, help="Covariance candidate-buffer capacity")
    parser.add_argument("--max-fold-mib", type=float, default=128.)
    args = parser.parse_args(argv)
    try:
        inputs = {path.resolve() for path in args.input}
        outputs = ([Path(args.output)] if args.output != "-" else []) + ([args.save_table] if args.save_table else [])
        if any(path.resolve() in inputs for path in outputs):
            raise ValueError("Output paths must differ from input paths")
        if len({path.resolve() for path in outputs}) != len(outputs):
            raise ValueError("JSON and table outputs require different paths")
        config = SearchConfig(period_min=args.pmin, period_max=args.pmax,
            durations_hours=args.durations_hours, threshold=args.threshold, max_candidates=args.limit,
            bin_minutes=args.bin_minutes, window_days=args.window_days, clip_sigma=args.clip_sigma,
            noise_degree=args.noise_degree, correlated_noise=not args.no_red_noise,
            ingress_fractions=args.ingress_fractions, shape_layers=args.shape_layers,
            phase_oversample=args.phase_oversample, period_oversample=args.period_oversample,
            period_batch=args.batch, candidate_capacity=args.capacity, max_fold_mib=args.max_fold_mib)
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA-capable GPU is required")
        data = read_inputs(args.input)
        if args.prepared:
            if "flux_err" not in data:
                raise ValueError("Prepared input requires flux_err")
            result = search_prepared(data["time_bjd"], data["flux"], data["flux_err"], config=config)
        else:
            result = search_transits(data["time_bjd"], data["flux"], data.get("flux_err"),
                                     _normalization_groups(data), config=config)
        report = result.to_dict()
        report["input_files"] = [str(path) for path in args.input]
        report["input_mode"] = "prepared" if args.prepared else "raw"
        if args.save_table:
            _write_table(args.save_table, result.table)
        _write_json(args.output, report)
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.exit(2, f"tess-search: {error}\n")
    except RuntimeError as error:
        parser.exit(1, f"tess-search: {error}\n")

