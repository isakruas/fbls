"""Light curves of one star from the data directory written by targets/download.py."""
import json
from pathlib import Path

import numpy as np


# Sectors searched per star. Eight sectors hold two transits of any period up to 100 days. The period
# grid grows with the time span, so sectors more than MAX_GAP_SECTORS apart (about a year) are never
# combined: the search uses the largest group of sectors within that gap, trimmed to the eight spanning
# the shortest time, the most recent on ties.
SEARCH_SECTORS = 8
MAX_GAP_SECTORS = 13


def search_window(sectors, count=SEARCH_SECTORS, max_gap=MAX_GAP_SECTORS):
    """The sectors to search: the largest group whose consecutive gaps are at most `max_gap` sectors
    (shortest span, then most recent, on ties), reduced to the `count` spanning the shortest time."""
    sectors = sorted(sectors)
    if not sectors:
        return []
    groups = [[sectors[0]]]
    for a, b in zip(sectors, sectors[1:]):
        (groups[-1].append(b) if b - a <= max_gap else groups.append([b]))
    best = max(groups, key=lambda g: (len(g), -(g[-1] - g[0]), g[-1]))
    if len(best) <= count:
        return best
    windows = [best[i:i + count] for i in range(len(best) - count + 1)]
    return min(windows, key=lambda w: (w[-1] - w[0], -w[-1]))


def load_star(root, tic, sectors=SEARCH_SECTORS):
    """The search window of a star (see `search_window`; None: every sector): time, flux, flux_err,
    centroids, sector labels, metadata."""
    folder = Path(root) / "curves" / str(tic)
    files = {int(p.stem.split("-")[1]): p for p in folder.glob("sector-*.npz")}
    chosen = sorted(files) if sectors is None else search_window(files, sectors)
    files = [files[k] for k in chosen]
    if not files:
        raise FileNotFoundError(f"no light curves for TIC {tic}")
    arrays = {k: [] for k in ("time_bjd", "flux", "flux_err", "centr1", "centr2", "sector")}
    meta = {}
    for path in files:
        sector = int(path.stem.split("-")[1])
        with np.load(path) as data:
            n = len(data["time_bjd"])
            for key in ("time_bjd", "flux", "flux_err"):
                arrays[key].append(np.asarray(data[key], float))
            for key in ("centr1", "centr2"):
                arrays[key].append(np.asarray(data[key], float) if key in data.files else np.full(n, np.nan))
            arrays["sector"].append(np.full(n, sector))
        sidecar = path.with_suffix(".json")
        meta[sector] = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    star = {k: np.concatenate(v) for k, v in arrays.items()}
    order = np.argsort(star["time_bjd"], kind="stable")
    star = {k: v[order] for k, v in star.items()}
    info = next((m.get("star") for m in meta.values() if m.get("star", {}).get("radius")), None) or \
        next((m.get("star") for m in meta.values() if m.get("star")), {})
    crowd = [m["crowdsap"] for m in meta.values() if m.get("crowdsap") is not None]
    star.update(tic=int(tic), meta=meta, star=info, crowdsap=float(np.median(crowd)) if crowd else None)
    return star


def downloaded_stars(root):
    return sorted(int(p.name) for p in (Path(root) / "curves").glob("*") if p.name.isdigit() and any(p.glob("*.npz")))
