"""Blind survey: detector, catalog cross-match, vetting, tiers and dated public candidate files.

For every star of a target list: run the GPU search on all sectors, measure each
accepted peak with the other peaks masked (vetting on the detector's own
detrended family), cross-match against the catalog snapshots, run the pixel
difference-image test and TRICERATOPS (false positive probability) for new candidates.
Results: one JSON per star under `data/reports/`, and under `public/` the dated
tables `candidates_YYYY-MM-DD.csv` (every vetted signal processed that day),
`ctoi_YYYY-MM-DD.csv` (new candidates in the ExoFOP CTOI layout). A star is
processed once, so its candidates appear only on the day it was processed.
Catalog information is used only after the search.

Usage:
  python -m tess_survey.run --targets targets/targets_download.csv [--no-fpp] [--no-pixel]
  python -m tess_survey.run --tic 123456789
"""
import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = "Isak Paulo de Andrade Ruas"
TIER_ORDER = ("strong", "candidate", "inconclusive", "rejected")


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value


def tier(event):
    """Tier from vetting, model evidence and FPP; no learned classifier is involved."""
    m, fpp = event["metrics"], event.get("fpp") or {}
    if m["verdict"] == "fail":
        return "rejected: vetting"
    if fpp.get("nfpp") is not None and fpp["nfpp"] > .1:
        return "rejected: NFPP (likely nearby star)"
    preferred = bool(event["peak"].get("model_preferred"))
    if event.get("search") == "single":
        return "candidate: single event"
    if fpp.get("fpp") is not None:
        if fpp["fpp"] < .01 and fpp["nfpp"] < .001 and preferred:
            return "strong: FPP < 1% and NFPP < 0.1%"
        if fpp["fpp"] < .5 and fpp["nfpp"] < .001:
            return "candidate: likely planet"
        return "inconclusive"
    return "candidate: no FPP" if preferred else "inconclusive: binary-like evidence"


def star_events(tic, root, config, snapshots, fpp=False, pixel=True):
    """Detector peaks of one star with vetting, cross-match, pixel test and FPP."""
    import torch
    from tess_search import search_transits
    from tess_search.core import prepare_families, gpu64
    from .star import load_star
    from .catalog import known_by_star, match_known, match_single, known_masks
    from .vetting import fold, measure
    from .pixels import pixel_test, apply_pixel_test
    from .fpp import fpp_job
    star = load_star(root, tic)
    time_t = gpu64(star["time_bjd"])
    result = search_transits(time_t, gpu64(star["flux"]), gpu64(star["flux_err"]),
                             torch.as_tensor(star["sector"], device="cuda").long(), config=config)
    durations = gpu64(config.durations_hours) / 24
    families = prepare_families(time_t, gpu64(star["flux"]), gpu64(star["flux_err"]),
                                torch.as_tensor(star["sector"], device="cuda").long(), durations,
                                bin_minutes=0., window_min=config.window_days, window_factor=config.window_factor,
                                window_max=config.window_max_days, clip_sigma=config.clip_sigma,
                                edge_trim=config.edge_trim_windows)
    residuals = []
    for curve, indices in families:
        t = curve.time.cpu().numpy()
        position = np.searchsorted(star["time_bjd"], t)
        corrected = np.full(len(star["time_bjd"]), np.nan)
        corrected[np.clip(position, 0, len(corrected) - 1)] = curve.flux.cpu().numpy()
        residuals.append((float(durations[indices].min()), float(durations[indices].max()), corrected))
    known = known_by_star(str(snapshots)).get(int(tic), [])
    baseline = float(star["time_bjd"][-1] - star["time_bjd"][0])
    peaks = result.candidates(ranking="signal")
    events = []
    for i, peak in enumerate(peaks):
        others = [(p["period_days"], p["epoch_bjd"], p["duration_hours"]) for j, p in enumerate(peaks) if j != i]
        match = match_known(peak, known)
        masks = others + known_masks(known, peak)
        d = peak["duration_hours"] / 24
        corrected = min(residuals, key=lambda r: abs(math.log((r[0] * r[1]) ** .5 / d)))[2]
        event = {"tic": int(tic), "search": "periodic", "peak": peak, "known": match, "baseline_days": baseline}
        try:
            prep = fold(star["time_bjd"], corrected, star["sector"], peak["period_days"], peak["epoch_bjd"],
                        peak["duration_hours"], masks)
            event["metrics"] = measure(prep, star["star"], star["centr1"], star["centr2"], star["crowdsap"], star["meta"])
        except ValueError as error:
            event["error"] = str(error)
            events.append(event)
            continue
        if pixel and event["metrics"]["verdict"] == "pass":
            try:
                apply_pixel_test(event["metrics"], pixel_test(star, peak["period_days"], peak["epoch_bjd"], peak["duration_hours"], root))
            except Exception as error:
                event["metrics"]["pixel_test"] = {"error": str(error)[-300:]}
        if fpp and event["metrics"]["verdict"] == "pass" and match is None:
            event["fpp"] = fpp_job(star, prep, event["metrics"], root)
        event["tier"] = tier(event)
        events.append(event)
    for single in result.single_events:
        event = {"tic": int(tic), "search": "single", "peak": single, "known": match_single(single, known),
                 "baseline_days": baseline, "metrics": {"period_days": None, "epoch_bjd": single["epoch_bjd"],
                                                        "duration_hours": single["duration_hours"],
                                                        "depth_ppm": single["depth_ppm"], "depth_err_ppm": None,
                                                        "snr": single["snr"], "transits": 1, "flags": {},
                                                        "verdict": "pass", "star": star["star"]}}
        event["tier"] = "candidate: single event"
        events.append(event)
    return {"tic": int(tic), "processed_utc": datetime.now(timezone.utc).isoformat(), "config": result.config.__dict__,
            "statistics": {k: v for k, v in result.statistics.items() if k in ("observations", "baseline_days", "periods_tested",
                                                                              "resolved_peaks", "single_events", "seconds_total")},
            "events": json_safe(events)}


def rank(label):
    return next((i for i, o in enumerate(TIER_ORDER) if label.startswith(o)), len(TIER_ORDER))


def write_public(reports, public, author):
    """Dated candidate table and CTOI rows for the stars processed in this run.

    Each star has one report, written once, so a candidate appears in the public
    files of the day it was processed and never again."""
    from .ctoi import ctoi_row, COLUMNS
    public.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows, ctoi_rows = [], []
    for report in reports:
        for e in report["events"]:
            if "metrics" not in e or "tier" not in e:
                continue
            m, peak, known = e["metrics"], e["peak"], e.get("known") or {}
            rows.append({"tic": e["tic"], "search": e["search"], "tier": e["tier"], "known": known.get("name"),
                         "known_source": known.get("source"), "known_disposition": known.get("disposition"),
                         "period_days": m.get("period_days"), "epoch_bjd": m.get("epoch_bjd"),
                         "duration_hours": m.get("duration_hours"), "depth_ppm": m.get("depth_ppm"),
                         "radius_earth": m.get("radius_earth"), "detector_snr": peak.get("snr"), "vetting_snr": m.get("snr"),
                         "transits": m.get("transits"), "planet_log_odds": peak.get("planet_log_odds"),
                         "model_preferred": peak.get("model_preferred"), "fpp": (e.get("fpp") or {}).get("fpp"),
                         "nfpp": (e.get("fpp") or {}).get("nfpp"),
                         "pixel_offset_arcsec": (m.get("pixel_test") or {}).get("offset_arcsec"),
                         "flags": ";".join(f"{k}={v}" for k, v in m.get("flags", {}).items())})
            if e.get("known") is None and e["tier"].startswith(("strong", "candidate")):
                ctoi_rows.append(ctoi_row(e, author))
    rows.sort(key=lambda r: (rank(r["tier"]), r["known"] is not None, -(r["detector_snr"] or 0)))

    def append(path, fieldnames, new_rows):
        existing = []
        if path.exists():
            with path.open() as f:
                existing = list(csv.DictReader(f))
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(existing + new_rows)
    if rows:
        append(public / f"candidates_{today}.csv", list(rows[0]), rows)
    if ctoi_rows:
        append(public / f"ctoi_{today}.csv", COLUMNS, ctoi_rows)
    return len(rows), len(ctoi_rows)


def main():
    from tess_search import SearchConfig
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", type=Path, help="CSV with a tic_id column")
    parser.add_argument("--tic", type=int, nargs="*", help="Explicit TIC identifiers")
    parser.add_argument("--root", type=Path, default=ROOT / "data")
    parser.add_argument("--snapshots", type=Path, default=ROOT / "data" / "snapshots")
    parser.add_argument("--public", type=Path, default=ROOT / "public")
    parser.add_argument("--pmax", type=float, default=100.)
    parser.add_argument("--no-fpp", action="store_true", help="Skip TRICERATOPS for new candidates")
    parser.add_argument("--no-pixel", action="store_true", help="Skip the TPF difference-image test")
    parser.add_argument("--author", default=AUTHOR)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    tics = list(args.tic or [])
    if args.targets:
        with args.targets.open() as f:
            tics += [int(r["tic_id"]) for r in csv.DictReader(f)]
    if not tics:
        parser.error("give --targets or --tic")
    reports_dir = args.root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    config = SearchConfig(period_max=args.pmax, durations_hours=tuple(round(.5 * 1.25 ** i, 6) for i in range(19)))
    done = []
    pending = [t for t in tics if not (reports_dir / f"{t}.json").exists()][:args.limit]
    print(f"{len(tics)} stars, {len(pending)} pending", flush=True)
    for n, tic in enumerate(pending, 1):
        started = time.perf_counter()
        try:
            report = star_events(tic, args.root, config, args.snapshots, not args.no_fpp, not args.no_pixel)
        except Exception as error:
            report = {"tic": tic, "processed_utc": datetime.now(timezone.utc).isoformat(), "error": str(error)[-500:], "events": []}
        report["seconds"] = time.perf_counter() - started
        (reports_dir / f"{tic}.json").write_text(json.dumps(report, indent=1) + "\n")
        done.append(report)
        summary = [f"{e['search']} P={e['metrics'].get('period_days')} {e.get('known', {}) and e['known'].get('name') or 'not catalogued'} -> {e['tier']}"
                   for e in report["events"] if "tier" in e]
        print(f"[{n}/{len(pending)}] TIC {tic} {report['seconds']:.0f}s " + ("; ".join(summary) or report.get("error", "no events")), flush=True)
    rows, new = write_public(done, args.public, args.author)
    print(f"public: {rows} vetted signals, {new} new CTOI rows")


if __name__ == "__main__":
    main()
