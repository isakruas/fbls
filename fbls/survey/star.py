"""Light curves of one star from the data directory written by targets/download.py."""
import json
from pathlib import Path

import numpy as np


# Sectors searched per star: the most recent ones. Eight sectors hold two transits of any period up to 100 days.
SEARCH_SECTORS = 8


def load_star(root, tic, sectors=SEARCH_SECTORS):
    """The most recent `sectors` sectors of a star (None: every sector): time, flux, flux_err,
    centroids, sector labels, metadata."""
    folder = Path(root) / "curves" / str(tic)
    files = sorted(folder.glob("sector-*.npz"), key=lambda p: int(p.stem.split("-")[1]))
    files = files if sectors is None else files[-sectors:]
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
