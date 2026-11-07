"""Blind survey in stages, each one consuming the previous stage's files and skipping work already done.

  search   FBLS on the GPU: peaks, single events and the detrended
           families the peaks use                                      -> data/survey/search/<tic>.json, <tic>.npz
  vet      catalog cross-match, duplicates and vetting of every peak  -> data/survey/vetting/<tic>.json
  pixel    difference-image test of the signals that passed vetting   -> data/survey/pixel/<tic>.json
  fpp      TRICERATOPS for passing signals that are not catalogued    -> data/survey/fpp/<tic>.json
  refine   every sector of each CTOI star (and of each inconclusive
           star with further sectors): search, vetting, pixel, FPP
           and figures on the full baseline                          -> public/YYYY-MM-DD/refine/<tic>/, inconclusive/<tic>/
  publish  tiers, candidate table and ExoFOP CTOI rows                -> public/candidates.csv, ctoi.csv (cumulative) and
           public/YYYY-MM-DD/ (stars first published that day, with the figures of its CTOI rows)

Only the search uses the GPU; the later stages read its files and can be
repeated whenever a rule changes. A stage never overwrites an
existing output. Catalog information is used only after the search.

Usage:
  fbls-survey search --targets targets/<day>/<list>/targets_download.csv --workers 4
  fbls-survey vet --targets targets/<day>/<list>/targets_download.csv
  fbls-survey pixel --targets ... ; fbls-survey fpp --targets ... ; fbls-survey refine --targets ... ; fbls-survey publish --targets ...
"""
import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUTHOR = "Isak Paulo de Andrade Ruas"
TIER_ORDER = ("strong", "candidate", "inconclusive", "rejected", "duplicate")
STAGES = ("search", "vet", "pixel", "fpp", "refine", "publish")
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


def read_file(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def write_file(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload, processed_utc=datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(json_safe(payload), indent=1) + "\n")


def read(root, stage, tic):
    return read_file(Path(root) / FOLDERS[stage] / f"{tic}.json")


def write(root, stage, tic, payload):
    write_file(Path(root) / FOLDERS[stage] / f"{tic}.json", payload)


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


def save_residuals(path, star, config, peaks):
    """Detrended residuals of the families the peaks use, so the later stages need no GPU."""
    residuals = detrended(star, config)
    used = sorted({family_index(residuals, p) for p in peaks})
    arrays = {f"family_{i}": residuals[i][2].astype(np.float32) for i in used}
    arrays["bounds"] = np.array([[residuals[i][0], residuals[i][1]] for i in used])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_residuals(path):
    with np.load(path) as data:
        bounds = data["bounds"]
        keys = [k for k in data.files if k.startswith("family_")]
        return [(float(lo), float(hi), data[k].astype(float)) for (lo, hi), k in zip(bounds, keys)]


def folded(star, residuals, peak, masks):
    """Phase-folded detrended curve of one peak, from the family closest to its duration."""
    from .vetting import fold
    corrected = residuals[family_index(residuals, peak)][2]
    return fold(star["time_bjd"], corrected, star["sector"], peak["period_days"], peak["epoch_bjd"], peak["duration_hours"], masks)


def search_star(tic, root, config, star=None, residuals_file=None):
    import torch
    from fbls import search_transits
    from fbls.core import gpu64
    from .star import load_star
    star = star or load_star(root, tic)
    result = search_transits(gpu64(star["time_bjd"]), gpu64(star["flux"]), gpu64(star["flux_err"]),
                             torch.as_tensor(star["sector"], device="cuda").long(), config=config)
    peaks = result.candidates(ranking="signal")
    if peaks:
        save_residuals(residuals_file or residuals_path(root, tic), star, result.config, peaks)
    return {"tic": int(tic), "config": result.config.__dict__, "baseline_days": float(star["time_bjd"][-1] - star["time_bjd"][0]),
            "sectors": sorted(int(x) for x in set(star["sector"])),
            "statistics": {k: v for k, v in result.statistics.items()
                           if k in ("observations", "baseline_days", "periods_tested", "resolved_peaks", "single_events", "seconds_total")},
            "peaks": peaks, "single_events": result.single_events}


def complete_residuals(tic, root, search):
    """Residual file of a star searched before residuals were saved: detrending only, no new search."""
    from .star import load_star
    save_residuals(residuals_path(root, tic), load_star(root, tic), search_config(search["config"]), search["peaks"])


def vet_star(tic, root, snapshots, search, star=None, residuals_file=None):
    from .star import load_star
    from .catalog import known_by_star, match_known, match_single, known_masks
    from .vetting import measure
    peaks = search["peaks"]
    if not peaks and not search["single_events"]:
        return {"tic": int(tic), "baseline_days": search["baseline_days"], "events": []}
    star = star or load_star(root, tic)
    residuals = load_residuals(residuals_file or residuals_path(root, tic)) if peaks else []
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


def pixel_star(tic, root, vetting, star=None):
    from .star import load_star
    from .pixels import pixel_test
    star = star or load_star(root, tic)
    out = {}
    for event in vetting["events"]:
        if passing(event):
            p = event["peak"]
            try:
                out[str(event["id"])] = pixel_test(star, p["period_days"], p["epoch_bjd"], p["duration_hours"], root)
            except Exception as error:
                out[str(event["id"])] = {"error": str(error)[-300:]}
    return {"tic": int(tic), "events": out}


def fpp_star(tic, root, search, vetting, pixel, star=None, residuals_file=None):
    from .star import load_star
    from .fpp import fpp_job
    out = {}
    todo = [e for e in vetting["events"] if e.get("known") is None and passing(e, pixel)]
    if todo:
        star = star or load_star(root, tic)
        residuals = load_residuals(residuals_file or residuals_path(root, tic))
        for event in todo:
            prep = folded(star, residuals, event["peak"], [tuple(m) for m in event["masks"]])
            out[str(event["id"])] = fpp_job(star, prep, event["metrics"], root)
    return {"tic": int(tic), "events": out}


def period_maxima(star, config):
    """(period, SNR) of the period maxima above threshold: the detector run again on one star."""
    import torch
    from fbls import search_transits
    from fbls.core import gpu64
    result = search_transits(gpu64(star["time_bjd"]), gpu64(star["flux"]), gpu64(star["flux_err"]),
                             torch.as_tensor(star["sector"], device="cuda").long(), config=config)
    table = result.table.detach().cpu().numpy()
    return table[:, 0], table[:, 3]


def assemble(tic, root):
    """Final events of one star from the survey stage files."""
    return assemble_from(read(root, "vet", tic), read(root, "pixel", tic), read(root, "fpp", tic))


def assemble_from(vetting, pixel, fpp):
    """Final events: pixel result applied, FPP attached, tier set."""
    from .pixels import apply_pixel_test
    pixel, fpp = pixel or {}, fpp or {}
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


def candidate_rows(tics, root, author):
    """Table row of every vetted signal and CTOI row of every new periodic candidate, per star."""
    from .ctoi import ctoi_row
    rows, ctoi = {}, {}
    for tic in tics:
        for e in assemble(tic, root):
            if "metrics" not in e:
                continue
            m, peak, known = e["metrics"], e["peak"], e.get("known") or {}
            rows.setdefault(tic, []).append({
                "tic": e["tic"], "signal": e["id"], "search": e["search"], "tier": e["tier"], "known": known.get("name"),
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
                ctoi.setdefault(tic, []).append((e, ctoi_row(e, author)))
    return rows, ctoi


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def published_before(public, today):
    """Stars already listed in an earlier day's folder."""
    seen = set()
    for day in sorted(Path(public).glob("*-*-*")):
        if day.name == today or not (day / "candidates.csv").exists():
            continue
        with (day / "candidates.csv").open() as f:
            seen.update(int(r["tic"]) for r in csv.DictReader(f))
    return seen


def figures_for(e, star, residuals, maxima, title, folder, root):
    from .figures import candidate_figures, pixel_figure
    from .pixels import sector_images
    prep = folded(star, residuals, e["peak"], [tuple(m) for m in e["masks"]])
    candidate_figures(prep, e["metrics"], title, folder, maxima)
    if any(d.get("used") for d in (e["metrics"].get("pixel_test") or {}).get("sectors", [])):
        p = e["peak"]
        pixel_figure(sector_images(star, p["period_days"], p["epoch_bjd"], p["duration_hours"], root), title, folder)


def event_title(e):
    m = e["metrics"]
    return f"TIC {e['tic']}  P = {m['period_days']:.4f} d  depth {m['depth_ppm']:.0f} ppm  {m['transits']} transits  {e['tier']}"


def survey_day():
    """The day folder under public/: FBLS_DAY when set (run.sh fixes it once per run, so a run that
    crosses midnight stays in one folder), otherwise today in UTC."""
    return os.environ.get("FBLS_DAY") or datetime.now(timezone.utc).strftime("%Y-%m-%d")


def refine_folder(public, tic, today, kind="refine"):
    """The folder of a star's refinement (`refine` for CTOI stars, `inconclusive` for the others):
    an earlier day's if it exists, otherwise today's."""
    for day in sorted(Path(public).glob("*-*-*")):
        if (day / kind / str(tic) / "ctoi.json").exists():
            return day / kind / str(tic)
    return Path(public) / today / kind / str(tic)


def searched_sectors(root, tic):
    """Sectors the survey searched: recorded in the search file, or the search window of the files on disk."""
    from .star import search_window
    recorded = (read(root, "search", tic) or {}).get("sectors")
    if recorded:
        return recorded
    return search_window(int(f.stem.split("-")[1]) for f in Path(root, "curves", str(tic)).glob("sector-*.npz"))


def available_sectors(tic):
    """Sectors with a SPOC 120 s light curve of the star, from the local MAST sector manifests."""
    sys.path.insert(0, str(ROOT / "targets"))
    from select_targets import products
    return sorted(products().get(int(tic), {}))


def same_signal(event, peak):
    """The refined event that is the survey peak: same period (or a harmonic) and the same transit phase."""
    from .catalog import wrap
    p0, p1 = peak["period_days"], event["peak"]["period_days"]
    d = max(peak["duration_hours"], event["peak"]["duration_hours"]) / 24
    for ratio in (1, 2, .5):
        if abs(p1 / (p0 * ratio) - 1) < .01 and abs(wrap(event["peak"]["epoch_bjd"] - peak["epoch_bjd"], min(p0, p1))) < d:
            return True
    return False


def refine_star(tic, peaks, root, snapshots, folder, config, author):
    """Every sector of the star: download, search, vetting, pixel test, FPP and figures, each step
    skipped when its file is already in `folder`. Ends with `ctoi.json`: the refined CTOI row of each
    survey candidate that survives, or the reason it did not."""
    from .star import load_star
    from .ctoi import ctoi_row
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / "download.json").exists():
        result = subprocess.run([sys.executable, str(ROOT / "targets" / "download.py"), "--root", str(root), "--tic", str(tic)],
                                capture_output=True, text=True)
        write_file(folder / "download.json", {"tic": tic, "returncode": result.returncode, "log": result.stdout[-2000:]})
    star = load_star(root, tic, sectors=None)
    residuals_file = folder / "search.npz"
    search = read_file(folder / "search.json")
    if search is None:
        search = search_star(tic, root, config, star, residuals_file)
        write_file(folder / "search.json", search)
    vetting = read_file(folder / "vetting.json")
    if vetting is None:
        vetting = vet_star(tic, root, snapshots, search, star, residuals_file)
        write_file(folder / "vetting.json", vetting)
    pixel = read_file(folder / "pixel.json")
    if pixel is None:
        pixel = pixel_star(tic, root, vetting, star)
        write_file(folder / "pixel.json", pixel)
    fpp = read_file(folder / "fpp.json")
    if fpp is None:
        fpp = fpp_star(tic, root, search, vetting, pixel, star, residuals_file)
        write_file(folder / "fpp.json", fpp)
    events = assemble_from(vetting, pixel, fpp)
    residuals = load_residuals(residuals_file) if search["peaks"] else []
    maxima = period_maxima(star, search_config(search["config"])) if search["peaks"] else None
    out = {"tic": tic, "sectors": search.get("sectors"), "baseline_days": search["baseline_days"], "candidates": []}
    for peak in peaks:
        found = next((e for e in events if e["search"] == "periodic" and "metrics" in e and same_signal(e, peak)), None)
        entry = {"survey_period_days": peak["period_days"], "survey_snr": peak["snr"]}
        if found is None:
            entry["outcome"] = "not recovered on the full baseline"
        else:
            entry.update(outcome="recovered", tier=found["tier"], signal=found["id"], snr=found["metrics"].get("snr"),
                         period_days=found["metrics"]["period_days"], transits=found["metrics"].get("transits"))
            if found["tier"].startswith(("strong", "candidate")) and found.get("known") is None:
                entry["ctoi"] = ctoi_row(found, author)
                figures_for(found, star, residuals, maxima, event_title(found), folder / "figures" / str(found["id"]), root)
        out["candidates"].append(entry)
    write_file(folder / "ctoi.json", out)
    return out


def refine(tics, root, snapshots, public, author, config):
    """Refine, once, every star with a CTOI row and every star with an inconclusive uncatalogued
    periodic signal that has sectors beyond the ones the survey searched. A star already refined
    by either path (a CTOI that the full baseline did not confirm) is not refined again."""
    today = survey_day()
    _, ctoi = candidate_rows(tics, root, author)
    groups = {("refine", tic): [e["peak"] for e, _ in group] for tic, group in ctoi.items()}
    for tic in tics:
        peaks = [e["peak"] for e in assemble(tic, root) if e["search"] == "periodic" and "metrics" in e
                 and e.get("known") is None and e["tier"].startswith("inconclusive")]
        if peaks and tic not in ctoi and refinement(public, tic) is None:
            groups[("inconclusive", tic)] = peaks
    done = skipped = 0
    for (kind, tic), peaks in groups.items():
        folder = refine_folder(public, tic, today, kind)
        if (folder / "ctoi.json").exists():
            continue
        if kind == "inconclusive":
            extra = [x for x in available_sectors(tic) if x not in searched_sectors(root, tic)]
            if not extra:
                skipped += 1
                continue
        started = time.perf_counter()
        out = refine_star(tic, peaks, root, snapshots, folder, config, author)
        done += 1
        print(f"{kind} TIC {tic} {time.perf_counter() - started:.0f}s sectors {out['sectors']}: "
              + "; ".join(f"P={c['survey_period_days']:.3f} -> {c['outcome']}" + (f", {c['tier']}" if "tier" in c else "")
                          for c in out["candidates"]), flush=True)
    print(f"refine: {len(ctoi)} candidate stars, {len(groups) - len(ctoi)} inconclusive stars, "
          f"{done} refined now, {skipped} inconclusive without further sectors")


def refinement(public, tic):
    """The refinement of a star (CTOI or inconclusive), None when it was not refined."""
    today = survey_day()
    for kind in ("refine", "inconclusive"):
        out = read_file(refine_folder(public, tic, today, kind) / "ctoi.json")
        if out is not None:
            return out
    return None


NOT_RECOVERED = "inconclusive: not recovered on the full baseline"


def apply_refinement(rows, refined):
    """Table rows of a refined star with the refinement's outcome as their tier: the refined tier of
    a recovered signal, NOT_RECOVERED for one the full baseline did not confirm."""
    for row in rows:
        for c in refined["candidates"]:
            if row["search"] == "periodic" and row["period_days"] is not None and \
                    math.isclose(row["period_days"], c["survey_period_days"], rel_tol=1e-6):
                row["tier"] = c["tier"] if c["outcome"] == "recovered" else NOT_RECOVERED
    return rows


def searched_stars(root):
    """Every star with a survey search file."""
    return sorted(int(p.stem) for p in (Path(root) / FOLDERS["search"]).glob("*.json") if p.stem.isdigit())


def publish(tics, root, public, author):
    """Cumulative `public/candidates.csv` and `public/ctoi.csv` (every searched star, whatever list it
    came from, current rules), plus the folder of the day with the stars first published today and,
    for each of its CTOI rows, the supporting figures ExoFOP asks for (with the period maxima of a
    fresh detector run on that star). A refined star contributes its refined CTOI rows instead of the
    survey ones, and its table rows carry the refinement's outcome. ExoFOP needs period, epoch, depth
    and duration for cTOI consideration, so single events stay in the candidate table only."""
    from .ctoi import COLUMNS
    from .star import load_star
    public = Path(public)
    public.mkdir(parents=True, exist_ok=True)
    today = survey_day()
    rows, ctoi = candidate_rows(sorted(set(tics) | set(searched_stars(root))), root, author)
    for tic in rows:
        refined = refinement(public, tic)
        if refined is not None:
            apply_refinement(rows[tic], refined)
            ctoi[tic] = [(None, c["ctoi"]) for c in refined["candidates"] if "ctoi" in c]
        if tic in ctoi and not ctoi[tic]:
            del ctoi[tic]
    order = lambda r: (rank(r["tier"]), r["known"] is not None, -(r["detector_snr"] or 0))
    everything = sorted((r for group in rows.values() for r in group), key=order)
    if everything:
        write_csv(public / "candidates.csv", list(everything[0]), everything)
    write_csv(public / "ctoi.csv", COLUMNS, [row for group in ctoi.values() for _, row in group])
    seen = published_before(public, today)
    new = [tic for tic in rows if tic not in seen]
    day = public / today
    if new:
        day.mkdir(exist_ok=True)
        write_csv(day / "candidates.csv", list(everything[0]), sorted((r for t in new for r in rows[t]), key=order))
        new_ctoi = [(e, row) for t in new for e, row in ctoi.get(t, [])]
        if new_ctoi:
            write_csv(day / "ctoi.csv", COLUMNS, [row for _, row in new_ctoi])
            for e, _ in new_ctoi:
                if e is None:
                    continue  # refined: its figures are in the refine folder
                folder = day / "figures" / f"{e['tic']}_{e['id']}"
                if folder.exists():
                    continue  # drawn by an earlier publish today
                star = load_star(root, e["tic"])
                maxima = period_maxima(star, search_config(read(root, "search", e["tic"])["config"]))
                figures_for(e, star, load_residuals(residuals_path(root, e["tic"])), maxima, event_title(e), folder, root)
    return len(everything), sum(len(g) for g in ctoi.values()), len(new)


def run_stage(stage, tics, args):
    root = args.root
    def complete(tic):
        """Search done without error, with its residual file when it found peaks; errors are retried."""
        search = read(root, "search", tic)
        return search is not None and "peaks" in search and (not search["peaks"] or residuals_path(root, tic).exists())

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
            if stage == "search" and "peaks" in (read(root, "search", tic) or {}):
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
    parser.add_argument("--workers", type=int, default=1, help="Processes sharing the stage (the GPU, for search)")
    parser.add_argument("--shard", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    tics = list(args.tic or [])
    if args.targets:
        with args.targets.open() as f:
            tics += [int(r["tic_id"]) for r in csv.DictReader(f)]
    if not tics:
        parser.error("give --targets or --tic")
    if args.stage == "publish":
        rows, ctoi, new = publish(tics, args.root, args.public, args.author)
        print(f"public: {rows} vetted signals, {ctoi} CTOI rows, {new} stars first published today")
    elif args.stage == "refine":
        refine(tics, args.root, args.snapshots, args.public, args.author, search_config(pmax=args.pmax))
    elif args.workers > 1 and args.shard is None:
        children = [subprocess.Popen([sys.executable, "-m", "fbls.survey.run", *sys.argv[1:], "--shard", str(k)]) for k in range(args.workers)]
        for child in children:
            child.wait()
    else:
        if args.shard is not None:
            tics = tics[args.shard::args.workers]
        run_stage(args.stage, tics, args)


if __name__ == "__main__":
    main()
