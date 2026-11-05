"""Blind recovery of catalogued planets over a labeled star group.

Runs the detector on every star of the group (the eight most recent sectors), then
matches the catalog planets of each star to the reported peaks by transit times:
the candidate's predicted transits must coincide, within one duration, with the
catalog transits sampled in the data; a period that disagrees is recorded as
ambiguous (few transits separated by long gaps admit several periods). Coverage requires at
least two sampled transit epochs with three observations each.
Reports are written per star under `data/evaluation/<group>/` and summarized in
`recovery.json`; the control group must be evaluated once, with the frozen
detector, and never used for tuning.

Usage:
  python -m fbls.survey.evaluate --group targets/development_targets.csv --name development [--workers 4]
"""
import argparse
import csv
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BINS = ((.5, 3.), (3., 10.), (10., 30.), (30., 100.))


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def planets_of(tic, snapshots):
    """Confirmed planets (TOI CP/KP, published transiting planets) and TOI false positives of a star."""
    from .catalog import known_by_star
    rows = []
    for k in known_by_star(str(snapshots)).get(int(tic), []):
        if k["source"] == "TOI" and k["disposition"] in ("CP", "KP", "FP"):
            kind = "planet" if k["disposition"] != "FP" else "false_positive"
        elif k["source"] == "planet" and k["transiting"]:
            kind = "planet"
        else:
            continue
        if not finite(k["period"]) or k["period"] <= 0:
            continue
        row = {"name": k["name"], "kind": kind, "period": k["period"], "epoch": k["epoch"], "duration_hours": k["duration_hours"],
               "period_err": k.get("period_err"), "epoch_err": k.get("epoch_err")}
        same = next((r for r in rows if abs(r["period"] / k["period"] - 1) < 2e-3), None)
        if same is None:
            rows.append(row)
        elif (row["period_err"] or math.inf) < (same["period_err"] or math.inf):
            same.update(row, name=same["name"], kind=same["kind"])
    return rows


def sampled_transits(time, period, epoch, duration):
    """Transit times of an ephemeris that fall on at least three observations."""
    cycles = np.arange(np.ceil((time[0] - epoch) / period), np.floor((time[-1] - epoch) / period) + 1)
    transits = epoch + cycles * period
    sampled = np.array([np.sum(np.abs(time - t) <= duration / 2) >= 3 for t in transits], dtype=bool)
    return transits[sampled], cycles[sampled]


def coincident(times, others, tolerance):
    """Fraction of `times` within `tolerance` of some entry of `others`."""
    if not len(times) or not len(others):
        return 0.
    gaps = np.min(np.abs(times[:, None] - others[None, :]), axis=1)
    return float(np.mean(gaps <= tolerance))


def rank_of(candidates, target, time):
    """Rank of the first candidate whose transits are the catalog transits: at least half of the
    catalog transits sampled in the data coincide, within one duration, with a predicted transit of
    the candidate, and more than half of the candidate's sampled transits coincide with catalog
    transits. The period is not required to agree: with few transits in data separated by long gaps,
    several periods place transits on the same events and the catalog period (from more data) is
    only one of them; the hit records whether the period agrees within 1%. The tolerance grows with
    the propagated 3-sigma error of the catalog ephemeris, up to a quarter period; the coincidence
    fractions tolerate transit-timing variations. A harmonic that predicts extra transits on
    observed, flat data fails the second condition."""
    p, e, h = target["period"], target["epoch"], target["duration_hours"]
    if not finite(e) or not finite(h) or not candidates:
        return None
    d = h / 24
    transits, cycles = sampled_transits(time, p, e, d)
    if not len(transits):
        return None
    drift = 3 * np.hypot(np.abs(cycles) * (target.get("period_err") or 0.), target.get("epoch_err") or 0.)
    tolerance = d + np.minimum(drift, p / 4)
    for rank, c in enumerate(candidates, 1):
        predicted, _ = sampled_transits(time, c["period_days"], c["epoch_bjd"], max(d, c["duration_hours"] / 24))
        if not len(predicted):
            continue
        gaps = np.min(np.abs(transits[:, None] - predicted[None, :]), axis=1)
        if np.mean(gaps <= tolerance) >= .5 and coincident(predicted, transits, float(tolerance.max())) > .5:
            return rank
    return None


def ranked(peaks, ranking):
    """Stored peaks in the order of `TransitSearchResult.candidates(ranking=...)`."""
    rows = [dict(r) for r in peaks]
    if ranking == "signal":
        rows.sort(key=lambda r: -r["snr"])
    else:
        def key(r):
            odds, evidence, preferred = r.get("planet_log_odds"), r.get("log_bf_planet_noise"), r.get("model_preferred", False)
            return (0 if preferred else 1, -(evidence if preferred and evidence is not None else
                                             (odds if odds is not None else -math.inf)))
        rows.sort(key=key)
    return rows


def match(peaks, targets, time_all, sectors, baseline, config_pmax):
    out = []
    for target in targets:
        row = dict(target, sectors=sectors, baseline_days=baseline, in_range=.5 <= target["period"] <= config_pmax,
                   sampled_events=sampled_events(time_all, target))
        row["eligible"] = bool(row["in_range"] and row["sampled_events"] is not None and row["sampled_events"] >= 2)
        for method, ranking in (("signal", "signal"), ("model", "planet")):
            row[method + "_rank"] = rank_of(ranked(peaks, ranking), target, time_all)
        hit = next((c for c in ranked(peaks, "signal") if rank_of([c], target, time_all)), None)
        if hit:
            row["hit"] = {k: hit.get(k) for k in ("period_days", "duration_hours", "snr", "depth_ppm", "events", "pass", "planet_log_odds", "model_preferred")}
            row["hit"]["period_match"] = bool(abs(hit["period_days"] / target["period"] - 1) < .01)
        out.append(row)
    return out


def sampled_events(time, target):
    p, e, h = target["period"], target["epoch"], target["duration_hours"]
    if not finite(e) or not finite(h):
        return None
    cycles = np.round((time - e) / p)
    near = np.abs(time - (e + cycles * p)) <= h / 48
    _, counts = np.unique(cycles[near], return_counts=True)
    return int((counts >= 3).sum())


def evaluate_star(tic, root, snapshots, config):
    import torch
    from fbls import search_transits
    from fbls.core import gpu64
    from .star import load_star
    star = load_star(root, tic)
    result = search_transits(gpu64(star["time_bjd"]), gpu64(star["flux"]), gpu64(star["flux_err"]),
                             torch.as_tensor(star["sector"], device="cuda").long(), config=config)
    time_all = star["time_bjd"]
    baseline = float(time_all[-1] - time_all[0])
    targets = [dict(t, tic=int(tic)) for t in planets_of(tic, snapshots)]
    out = match(result.candidates(ranking="signal"), targets, time_all, len(set(star["sector"].astype(int))), baseline, config.period_max)
    return {"tic": int(tic), "seconds": result.statistics["seconds_total"], "baseline_days": baseline,
            "peaks": result.candidates(ranking="signal"), "single_events": result.single_events, "recovery": out}


def rematch(folder, root, snapshots, pmax):
    """Recompute the catalog matching of every stored report from its peaks, without re-running the detector."""
    from .star import load_star
    for path in folder.glob("*.json"):
        if path.name == "recovery.json":
            continue
        r = json.loads(path.read_text())
        if "peaks" not in r:
            continue
        star = load_star(root, r["tic"])
        time_all = star["time_bjd"]
        targets = [dict(t, tic=r["tic"]) for t in planets_of(r["tic"], snapshots)]
        r["recovery"] = match(r["peaks"], targets, time_all, len(set(star["sector"].astype(int))), r["baseline_days"], pmax)
        path.write_text(json.dumps(r, indent=1) + "\n")


def summarize(folder):
    rows, seconds, stars = [], 0., 0
    for path in folder.glob("*.json"):
        if path.name == "recovery.json":
            continue
        r = json.loads(path.read_text())
        stars += 1
        seconds += r.get("seconds", 0.)
        rows.extend(r.get("recovery", []))
    report = {"stars": stars, "seconds_total": seconds, "generated_utc": datetime.now(timezone.utc).isoformat(), "counts": {}}
    for kind in ("planet", "false_positive"):
        group = [r for r in rows if r["kind"] == kind and r["eligible"]]
        entry = {"eligible": len(group)}
        for method in ("signal", "model"):
            for n in (1, 5, 20):
                entry[f"{method}_top{n}"] = sum(r[method + "_rank"] is not None and r[method + "_rank"] <= n for r in group)
        entry["model_preferred"] = sum(bool((r.get("hit") or {}).get("model_preferred")) for r in group)
        entry["period_ambiguous"] = sum(bool(r.get("hit")) and not r["hit"]["period_match"] for r in group)
        entry["by_period"] = {f"{lo:g}-{hi:g}": {
            "eligible": sum(lo <= r["period"] < hi or (hi == 100. and r["period"] == hi) for r in group),
            "signal_top20": sum((lo <= r["period"] < hi or (hi == 100. and r["period"] == hi)) and r["signal_rank"] is not None and r["signal_rank"] <= 20 for r in group)}
            for lo, hi in BINS}
        report["counts"][kind] = entry
    report["missed_planets"] = [{k: r[k] for k in ("tic", "name", "period", "duration_hours", "sectors", "sampled_events")}
                                for r in rows if r["kind"] == "planet" and r["eligible"] and r["signal_rank"] is None]
    (folder / "recovery.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    from fbls import SearchConfig
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", type=Path, required=True, help="CSV with a tic_id column")
    parser.add_argument("--name", required=True, help="Evaluation name (output folder under data/evaluation)")
    parser.add_argument("--root", type=Path, default=ROOT / "data")
    parser.add_argument("--snapshots", type=Path, default=ROOT / "data" / "snapshots")
    parser.add_argument("--pmax", type=float, default=100.)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent processes sharing the GPU")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    folder = args.root / "evaluation" / args.name
    folder.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        rematch(folder, args.root, args.snapshots, args.pmax)
        report = summarize(folder)
        print(json.dumps(report["counts"], indent=1))
        print("missed planets:", len(report["missed_planets"]))
        return
    with args.group.open() as f:
        tics = [int(r["tic_id"]) for r in csv.DictReader(f)]
    if args.workers > 1 and args.shard == 0 and "--shard" not in sys.argv:
        children = [subprocess.Popen([sys.executable, "-m", "fbls.survey.evaluate", "--group", str(args.group), "--name", args.name,
                                      "--root", str(args.root), "--snapshots", str(args.snapshots), "--pmax", str(args.pmax),
                                      "--workers", str(args.workers), "--shard", str(k)]) for k in range(args.workers)]
        for child in children:
            child.wait()
        report = summarize(folder)
        print(json.dumps(report["counts"], indent=1))
        print("missed planets:", len(report["missed_planets"]))
        return
    config = SearchConfig(period_max=args.pmax, durations_hours=tuple(round(.5 * 1.25 ** i, 6) for i in range(19)))
    mine = [t for i, t in enumerate(tics) if i % args.workers == args.shard and not (folder / f"{t}.json").exists()]
    for n, tic in enumerate(mine, 1):
        started = time.perf_counter()
        try:
            report = evaluate_star(tic, args.root, args.snapshots, config)
        except Exception as error:
            report = {"tic": tic, "error": str(error)[-500:], "recovery": []}
        report["seconds"] = time.perf_counter() - started
        (folder / f"{tic}.json").write_text(json.dumps(report, indent=1) + "\n")
        found = [f"{r['name']} rank {r['signal_rank']}" for r in report["recovery"] if r.get("eligible")]
        print(f"[shard {args.shard}: {n}/{len(mine)}] TIC {tic} {report['seconds']:.0f}s " + ("; ".join(found) or report.get("error", "no eligible targets")), flush=True)
    if args.workers == 1:
        report = summarize(folder)
        print(json.dumps(report["counts"], indent=1))
        print("missed planets:", len(report["missed_planets"]))


if __name__ == "__main__":
    main()
