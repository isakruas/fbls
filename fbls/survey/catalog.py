"""Catalogued signals per star and the rules that match a detected signal to them.

Sources are the snapshots written by `targets/select_targets.py fetch-snapshots`:
`catalogs/toi.json` (Exoplanet Archive TOI table) and `catalogs/known.json`
(ExoFOP CTOIs, published planets, TESS eclipsing binaries, SPOC TCEs).
"""
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np


def wrap(phase, period):
    return (phase + period / 2) % period - period / 2


def positive(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


@lru_cache(maxsize=None)
def known_by_star(snapshots):
    """{tic: [signal, ...]} with period, epoch, duration, errors, disposition and transiting flag."""
    snapshots = Path(snapshots)
    out = defaultdict(list)
    for r in json.loads((snapshots / "catalogs" / "toi.json").read_text()):
        out[int(r["tid"])].append({
            "source": "TOI", "name": f"TOI {r['toi']}", "toi": r["toi"], "tic": int(r["tid"]),
            "period": positive(r.get("pl_orbper")), "epoch": positive(r.get("pl_tranmid")),
            "period_err": positive(r.get("pl_orbpererr1")), "epoch_err": positive(r.get("pl_tranmiderr1")),
            "duration_hours": positive(r.get("pl_trandurh")), "disposition": r.get("tfopwg_disp"), "transiting": True})
    for r in json.loads((snapshots / "catalogs" / "known.json").read_text()):
        out[int(r["tic"])].append({"source": r["source"], "name": r["name"], "toi": None, "tic": int(r["tic"]),
                                   "period": positive(r.get("period")), "epoch": positive(r.get("epoch")),
                                   "period_err": positive(r.get("period_err")), "epoch_err": positive(r.get("epoch_err")),
                                   "duration_hours": positive(r.get("duration_hours")),
                                   "disposition": r.get("disposition"), "transiting": bool(r.get("transiting"))})
    return dict(out)


def match_known(event, known):
    """Match a periodic signal to a catalogued one: same period or a harmonic (1, 2, 1/2, 3, 1/3) and
    the same phase while the catalog ephemeris is still predictive at our epoch. The phase tolerance
    grows with the propagated 3-sigma drift; past a quarter period the phase is unknown and the period
    alone decides. Single-transit entries (epoch without period) match when one of our transits falls
    on their event; non-transiting entries (binaries matched by period) use the period alone."""
    duration = event["duration_hours"] / 24
    for k in known:
        p, e = k["period"], k["epoch"]
        d = max((k["duration_hours"] or 0) / 24, duration)
        found = None
        if p is None and e is not None:
            if abs(wrap(e - event["epoch_bjd"], event["period_days"])) < d + 3 * (k.get("epoch_err") or 0):
                found = {"period_ratio": None, "phase_checked": True}
        elif p is not None:
            for ratio in (1, 2, .5, 3, 1 / 3):
                if abs(event["period_days"] / (p * ratio) - 1) < .01:
                    cycles = abs(event["epoch_bjd"] - e) / p if e is not None else 0
                    drift = 3 * np.hypot(cycles * (k.get("period_err") or 0), k.get("epoch_err") or 0)
                    if e is None or not k["transiting"] or drift > min(p, p * ratio) / 4:
                        found = {"period_ratio": ratio, "phase_checked": False}
                    elif abs(wrap(event["epoch_bjd"] - e, min(p, p * ratio))) < d + drift:
                        found = {"period_ratio": ratio, "phase_checked": True}
                    if found:
                        break
        if found:
            return {"source": k["source"], "name": k["name"], "toi": k.get("toi"), "disposition": k["disposition"], **found}
    return None


def match_single(event, known):
    """Match an isolated event: a predicted transit of a periodic signal (within the propagated 3-sigma
    drift) or the event time of a single-transit entry."""
    t0, duration = event["epoch_bjd"], event["duration_hours"] / 24
    for k in known:
        p, e = k["period"], k["epoch"]
        if e is None or not k["transiting"]:
            continue
        d = max((k["duration_hours"] or 0) / 24, duration)
        if p is None:
            hit = abs(t0 - e) < d + 3 * (k.get("epoch_err") or 0)
        else:
            drift = 3 * np.hypot(abs(t0 - e) / p * (k.get("period_err") or 0), k.get("epoch_err") or 0)
            hit = drift < p / 4 and abs(wrap(t0 - e, p)) < d + drift
        if hit:
            return {"source": k["source"], "name": k["name"], "toi": k.get("toi"), "disposition": k["disposition"],
                    "period_ratio": None, "phase_checked": True}
    return None


def known_masks(known, event=None):
    """(period, epoch, duration_hours) of every catalogued transiting signal with a full ephemeris,
    leaving out the entries that describe `event` itself (the same planet is often listed several times)."""
    return [(k["period"], k["epoch"], k["duration_hours"]) for k in known
            if k["transiting"] and None not in (k["period"], k["epoch"], k["duration_hours"])
            and (event is None or match_known(event, [k]) is None)]
