"""Blind survey in stages, each one consuming the previous stage's files and skipping work already done.

  search   FBLS on the GPU: peaks, single events and the detrended
           families the peaks use                                      -> data/survey/search/<tic>.json, <tic>.npz
  vet      catalog cross-match, duplicates and vetting of every peak  -> data/survey/vetting/<tic>.json
  pixel    difference-image test of the signals that passed vetting   -> data/survey/pixel/<tic>.json
  fpp      TRICERATOPS for passing signals that are not catalogued    -> data/survey/fpp/<tic>.json
  publish  tiers, candidate table and ExoFOP CTOI rows                -> public/candidates_YYYY-MM-DD.csv, ctoi_YYYY-MM-DD.csv

Only the search uses the GPU; the later stages read its files and can be
repeated whenever a rule changes. A stage never overwrites an
existing output. Catalog information is used only after the search.

Usage:
  fbls-survey search --targets targets/targets_download.csv
  fbls-survey vet --targets targets/targets_download.csv
  fbls-survey pixel --targets ... ; fbls-survey fpp --targets ... ; fbls-survey publish --targets ...
"""
import argparse
import copy
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUTHOR = "Isak Paulo de Andrade Ruas"
TIER_ORDER = ("strong", "candidate", "inconclusive", "rejected", "duplicate")
STAGES = ("search", "vet", "pixel", "fpp", "publish")
FOLDERS = {"search": "survey/search", "vet": "survey/vetting", "pixel": "survey/pixel", "fpp": "survey/fpp"}


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


def read(root, stage, tic):
    path = Path(root) / FOLDERS[stage] / f"{tic}.json"
    return json.loads(path.read_text()) if path.exists() else None


def write(root, stage, tic, payload):
    folder = Path(root) / FOLDERS[stage]
    folder.mkdir(parents=True, exist_ok=True)
    payload = dict(payload, processed_utc=datetime.now(timezone.utc).isoformat())
    (folder / f"{tic}.json").write_text(json.dumps(json_safe(payload), indent=1) + "\n")


def search_config(stored=None, pmax=100.):
    from fbls import SearchConfig
    if stored is None:
        return SearchConfig(period_max=pmax, durations_hours=tuple(round(.5 * 1.25 ** i, 6) for i in range(19)))
    return SearchConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in stored.items()})


def related_peak(peak, stronger):
    """Index of a stronger peak whose period is this one's period or a harmonic (1, 2, 1/2, 3, 1/3)
    within 1%: the same eclipsing system seen again (secondary eclipse, alias), not a new candidate."""
    for j, other in enumerate(stronger):
        if any(abs(peak["period_days"] / (other["period_days"] * ratio) - 1) < .01 for ratio in (1, 2, .5, 3, 1 / 3)):
            return j
    return None


def tier(event):
    """Tier from vetting, model evidence and FPP; no learned classifier is involved. Without stellar
    radius and temperature the radius, density and FPP tests are meaningless, so the signal stays a candidate."""
    m, fpp = event["metrics"], event.get("fpp") or {}
    if m["verdict"] == "fail":
        return "rejected: vetting"
    if fpp.get("nfpp") is not None and fpp["nfpp"] > .1:
        return "rejected: NFPP (likely nearby star)"
    preferred = bool(event["peak"].get("model_preferred"))
    if event.get("search") == "single":
        return "candidate: single event"
    star = m.get("star") or {}
    if not (star.get("radius") and star.get("teff")):
        return "candidate: no stellar parameters"
    if fpp.get("fpp") is not None:
        if fpp["fpp"] < .01 and fpp["nfpp"] < .001 and preferred:
            return "strong: FPP < 1% and NFPP < 0.1%"
        if fpp["fpp"] < .5 and fpp["nfpp"] < .001:
            return "candidate: likely planet"
        return "inconclusive"
    return "candidate: no FPP" if preferred else "inconclusive: binary-like evidence"


def rank(label):
    return next((i for i, o in enumerate(TIER_ORDER) if label.startswith(o)), len(TIER_ORDER))


def family_index(residuals, peak):
    """Family whose duration range is closest to the peak duration."""
    d = peak["duration_hours"] / 24
    return min(range(len(residuals)), key=lambda i: abs(math.log((residuals[i][0] * residuals[i][1]) ** .5 / d)))


def detrended(star, config):
    """Per-family detrended residuals on the GPU: (shortest duration, longest duration, residual array) each."""
    import torch
    from fbls.core import prepare_families, gpu64
    time_t = gpu64(star["time_bjd"])
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
    return residuals


def residuals_path(root, tic):
    return Path(root) / FOLDERS["search"] / f"{tic}.npz"


def save_residuals(root, tic, star, config, peaks):
    """Detrended residuals of the families the peaks use, so the later stages need no GPU."""
    residuals = detrended(star, config)
    used = sorted({family_index(residuals, p) for p in peaks})
    arrays = {f"family_{i}": residuals[i][2].astype(np.float32) for i in used}
    arrays["bounds"] = np.array([[residuals[i][0], residuals[i][1]] for i in used])
    np.savez_compressed(residuals_path(root, tic), **arrays)


def load_residuals(root, tic):
    with np.load(residuals_path(root, tic)) as data:
        bounds = data["bounds"]
        keys = [k for k in data.files if k.startswith("family_")]
        return [(float(lo), float(hi), data[k].astype(float)) for (lo, hi), k in zip(bounds, keys)]


def folded(star, residuals, peak, masks):
    """Phase-folded detrended curve of one peak, from the family closest to its duration."""
    from .vetting import fold
    corrected = residuals[family_index(residuals, peak)][2]
    return fold(star["time_bjd"], corrected, star["sector"], peak["period_days"], peak["epoch_bjd"], peak["duration_hours"], masks)


def search_star(tic, root, config):
    import torch
    from fbls import search_transits
    from fbls.core import gpu64
    from .star import load_star
    star = load_star(root, tic)
    result = search_transits(gpu64(star["time_bjd"]), gpu64(star["flux"]), gpu64(star["flux_err"]),
                             torch.as_tensor(star["sector"], device="cuda").long(), config=config)
    peaks = result.candidates(ranking="signal")
    if peaks:
        save_residuals(root, tic, star, result.config, peaks)
    return {"tic": int(tic), "config": result.config.__dict__, "baseline_days": float(star["time_bjd"][-1] - star["time_bjd"][0]),
            "statistics": {k: v for k, v in result.statistics.items()
                           if k in ("observations", "baseline_days", "periods_tested", "resolved_peaks", "single_events", "seconds_total")},
            "peaks": peaks, "single_events": result.single_events}


def complete_residuals(tic, root, search):
    """Residual file of a star searched before residuals were saved: detrending only, no new search."""
    from .star import load_star
    save_residuals(root, tic, load_star(root, tic), search_config(search["config"]), search["peaks"])


def vet_star(tic, root, snapshots, search):
    from .star import load_star
    from .catalog import known_by_star, match_known, match_single, known_masks
    from .vetting import measure
    peaks = search["peaks"]
    if not peaks and not search["single_events"]:
        return {"tic": int(tic), "baseline_days": search["baseline_days"], "events": []}
    star = load_star(root, tic)
    residuals = load_residuals(root, tic) if peaks else []
    known = known_by_star(str(snapshots)).get(int(tic), [])
    parents = [related_peak(peak, peaks[:i]) for i, peak in enumerate(peaks)]
    events = []
    for i, peak in enumerate(peaks):
        event = {"id": i, "tic": int(tic), "search": "periodic", "peak": peak, "known": match_known(peak, known)}
        if parents[i] is not None:
            event["related"] = {"period_days": peaks[parents[i]]["period_days"], "snr": peaks[parents[i]]["snr"]}
            event["tier"] = "duplicate: harmonic or secondary of a stronger signal"
            events.append(event)
            continue
        others = [(p["period_days"], p["epoch_bjd"], p["duration_hours"]) for j, p in enumerate(peaks)
                  if j != i and parents[j] != i]
        event["masks"] = others + known_masks(known, peak)
        try:
            prep = folded(star, residuals, peak, event["masks"])
            event["metrics"] = measure(prep, star["star"], star["centr1"], star["centr2"], star["crowdsap"], star["meta"])
        except ValueError as error:
            event["error"] = str(error)
        events.append(event)
    for k, single in enumerate(search["single_events"]):
        events.append({"id": len(peaks) + k, "tic": int(tic), "search": "single", "peak": single, "known": match_single(single, known),
                       "metrics": {"period_days": None, "epoch_bjd": single["epoch_bjd"], "duration_hours": single["duration_hours"],
                                   "depth_ppm": single["depth_ppm"], "depth_err_ppm": None, "snr": single["snr"], "transits": 1,
                                   "flags": {}, "verdict": "pass", "star": star["star"]}})
    return {"tic": int(tic), "baseline_days": search["baseline_days"], "events": events}


def passing(event, pixel=None):
    """A periodic, non-duplicate signal that passed vetting (and the pixel test when given)."""
    if event.get("search") != "periodic" or "related" in event or "metrics" not in event:
        return False
    if event["metrics"]["verdict"] != "pass":
        return False
    if pixel is not None and str(event["id"]) in pixel.get("events", {}):
        from .pixels import apply_pixel_test
        metrics = copy.deepcopy(event["metrics"])
        apply_pixel_test(metrics, pixel["events"][str(event["id"])])
        return metrics["verdict"] == "pass"
    return True


def pixel_star(tic, root, vetting):
    from .star import load_star
    from .pixels import pixel_test
    star = load_star(root, tic)
    out = {}
    for event in vetting["events"]:
        if passing(event):
            p = event["peak"]
            try:
                out[str(event["id"])] = pixel_test(star, p["period_days"], p["epoch_bjd"], p["duration_hours"], root)
            except Exception as error:
                out[str(event["id"])] = {"error": str(error)[-300:]}
    return {"tic": int(tic), "events": out}


def fpp_star(tic, root, search, vetting, pixel):
    from .star import load_star
    from .fpp import fpp_job
    out = {}
    todo = [e for e in vetting["events"] if e.get("known") is None and passing(e, pixel)]
    if todo:
        star = load_star(root, tic)
        residuals = load_residuals(root, tic)
        for event in todo:
            prep = folded(star, residuals, event["peak"], [tuple(m) for m in event["masks"]])
            out[str(event["id"])] = fpp_job(star, prep, event["metrics"], root)
    return {"tic": int(tic), "events": out}


def assemble(tic, root):
    """Final events of one star from the stage files: pixel result applied, FPP attached, tier set."""
    from .pixels import apply_pixel_test
    vetting, pixel, fpp = read(root, "vet", tic), read(root, "pixel", tic) or {}, read(root, "fpp", tic) or {}
    if vetting is None:
        return []
    events = []
    for event in vetting["events"]:
        if "metrics" not in event:
            if "tier" in event:
                events.append(event)
            continue
        key = str(event["id"])
        if key in pixel.get("events", {}):
            apply_pixel_test(event["metrics"], pixel["events"][key])
        if key in fpp.get("events", {}):
            event["fpp"] = fpp["events"][key]
        event["tier"] = tier(event)
        events.append(event)
    return events


def publish(tics, root, public, author):
    """Candidate table (every vetted signal) and ExoFOP CTOI rows (new periodic candidates: ExoFOP needs
    period, epoch, depth and duration for cTOI consideration, so single events stay in the table only)."""
    from .ctoi import ctoi_row, COLUMNS
    rows, ctoi_rows = [], []
    for tic in tics:
        for e in assemble(tic, root):
            if "metrics" not in e:
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
            if e.get("known") is None and e["search"] == "periodic" and e["tier"].startswith(("strong", "candidate")):
                ctoi_rows.append(ctoi_row(e, author))
    rows.sort(key=lambda r: (rank(r["tier"]), r["known"] is not None, -(r["detector_snr"] or 0)))
    public = Path(public)
    public.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for name, fieldnames, content in (("candidates", list(rows[0]) if rows else None, rows), ("ctoi", COLUMNS, ctoi_rows)):
        if content:
            with (public / f"{name}_{today}.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(content)
    return len(rows), len(ctoi_rows)


def run_stage(stage, tics, args):
    root = args.root
    def complete(tic):
        search = read(root, "search", tic)
        return search is not None and (not search.get("peaks") or residuals_path(root, tic).exists())

    pending = []
    for tic in tics:
        if stage == "search":
            if not complete(tic):
                pending.append(tic)
            continue
        if read(root, stage, tic) is not None or not complete(tic):
            continue
        inputs = {"vet": ("search",), "pixel": ("vet",), "fpp": ("search", "vet", "pixel")}[stage]
        if all(read(root, s, tic) is not None for s in inputs):
            pending.append(tic)
    pending = pending[:args.limit]
    print(f"{stage}: {len(tics)} stars, {len(pending)} pending", flush=True)
    config = search_config(pmax=args.pmax)
    for n, tic in enumerate(pending, 1):
        started = time.perf_counter()
        try:
            if stage == "search" and read(root, "search", tic) is not None:
                complete_residuals(tic, root, read(root, "search", tic))
                print(f"[{n}/{len(pending)}] TIC {tic} residuals saved", flush=True)
                continue
            if stage == "search":
                out = search_star(tic, root, config)
            elif stage == "vet":
                out = vet_star(tic, root, args.snapshots, read(root, "search", tic))
            elif stage == "pixel":
                out = pixel_star(tic, root, read(root, "vet", tic))
            else:
                out = fpp_star(tic, root, read(root, "search", tic), read(root, "vet", tic), read(root, "pixel", tic))
        except Exception as error:
            out = {"tic": tic, "error": str(error)[-500:]}
        out["seconds"] = time.perf_counter() - started
        write(root, stage, tic, out)
        if stage == "search":
            summary = f"{len(out.get('peaks', []))} peaks, {len(out.get('single_events', []))} single events"
        elif stage == "vet":
            summary = "; ".join(f"P={e['peak'].get('period_days')} {e.get('known') and e['known'].get('name') or 'not catalogued'} "
                                f"{e.get('tier') or e.get('metrics', {}).get('verdict') or e.get('error')}" for e in out.get("events", []))
        else:
            summary = f"{len(out.get('events', {}))} signals"
        print(f"[{n}/{len(pending)}] TIC {tic} {out['seconds']:.0f}s " + (out.get("error") or summary or "no events"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--targets", type=Path, help="CSV with a tic_id column")
    parser.add_argument("--tic", type=int, nargs="*", help="Explicit TIC identifiers")
    parser.add_argument("--root", type=Path, default=ROOT / "data")
    parser.add_argument("--snapshots", type=Path, default=ROOT / "data" / "snapshots")
    parser.add_argument("--public", type=Path, default=ROOT / "public")
    parser.add_argument("--pmax", type=float, default=100.)
    parser.add_argument("--author", default=AUTHOR)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    tics = list(args.tic or [])
    if args.targets:
        with args.targets.open() as f:
            tics += [int(r["tic_id"]) for r in csv.DictReader(f)]
    if not tics:
        parser.error("give --targets or --tic")
    if args.stage == "publish":
        rows, new = publish(tics, args.root, args.public, args.author)
        print(f"public: {rows} vetted signals, {new} new CTOI rows")
    else:
        run_stage(args.stage, tics, args)


if __name__ == "__main__":
    main()
