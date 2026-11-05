"""Supporting figures of a candidate for ExoFOP and the paper, one file each in the candidate's folder:
the detrended light curve with the transits marked, the folded curve with the fitted trapezoid and its
residuals, each transit on its own, odd against even transits, the period maxima of the search above
threshold, and the full orbit with the strongest secondary window."""
from pathlib import Path

import numpy as np

from .vetting import trapezoid, wrap

BIN_MINUTES = 10.
MAX_INDIVIDUAL = 8
DARK, ORANGE, BLUE = "#1f2429", "#b4521a", "#2a6f97"


def binned(x, y, width):
    cells = np.floor(x / width).astype(int)
    _, inverse, counts = np.unique(cells, return_inverse=True, return_counts=True)
    mean_x = np.bincount(inverse, x) / counts
    mean_y = np.bincount(inverse, y) / counts
    scatter = np.sqrt(np.maximum(np.bincount(inverse, y ** 2) / counts - mean_y ** 2, 0)) / np.sqrt(counts)
    return mean_x, mean_y, scatter


def candidate_figures(prep, metrics, title, folder, maxima=None):
    """Write the six figures into `folder` and return their paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    period, duration, epoch = prep["period"], prep["duration"], prep["epoch"]
    valid = prep["valid"]
    time, phase, flux = prep["time"][valid], prep["phase"][valid], prep["corrected"][valid] * 1e3
    phase_h = phase * 24
    width_h = BIN_MINUTES / 60
    depth = (metrics.get("depth_ppm") or 0) / 1e6
    t14 = (metrics.get("fitted_duration_hours") or metrics["duration_hours"]) / 24
    ingress = metrics.get("ingress_fraction") or .2
    model = lambda p: trapezoid(p, depth, t14, ingress) * 1e3
    near = np.abs(phase_h) < 3 * duration * 24
    number = np.round((time - epoch) / period)
    transits = sorted(n for n in np.unique(number[near]) if (near & (number == n)).sum() >= 3)
    written = []

    def save(fig, name):
        fig.suptitle(title, fontsize=10)
        fig.savefig(folder / name, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(folder / name)

    fig, ax = plt.subplots(figsize=(13, 3.2))
    x, y, _ = binned(time, flux, BIN_MINUTES / 1440)
    ax.plot(x, y, ".", color=DARK, ms=2, alpha=.6)
    for n in transits:
        ax.axvline(epoch + n * period, color=ORANGE, lw=1, alpha=.8)
    ax.set_xlabel("BJD")
    ax.set_ylabel("flux (ppt)")
    ax.set_title(f"detrended light curve, {BIN_MINUTES:.0f}-min bins; {len(transits)} transits marked", fontsize=10)
    save(fig, "light_curve.png")

    fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True, height_ratios=[2, 1])
    x, y, e = binned(phase_h[near], flux[near], width_h)
    ax.errorbar(x, y, e, fmt=".", color=DARK, ms=4, lw=.8, label=f"{BIN_MINUTES:.0f}-min bins")
    order = np.argsort(phase_h[near])
    ax.plot(phase_h[near][order], model(phase[near][order]), color=ORANGE, lw=1.5, label="trapezoid fit")
    ax.set_ylabel("flux (ppt)")
    ax.set_title("folded on the period", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    x, y, e = binned(phase_h[near], flux[near] - model(phase[near]), width_h)
    ax_res.errorbar(x, y, e, fmt=".", color=DARK, ms=4, lw=.8)
    ax_res.axhline(0, color=ORANGE, lw=1)
    ax_res.set_xlabel("hours from mid-transit")
    ax_res.set_ylabel("residual (ppt)")
    save(fig, "folded.png")

    fig, ax = plt.subplots(figsize=(6, 1.2 + .9 * min(len(transits), MAX_INDIVIDUAL)))
    shown = transits[:MAX_INDIVIDUAL]
    step = max(4 * depth * 1e3, 4 * np.nanstd(flux[near]) / np.sqrt(max(1, (near & (number == shown[0])).sum() / 5)) if shown else 1.)
    for k, n in enumerate(shown):
        select = near & (number == n)
        x, y, e = binned(phase_h[select], flux[select], width_h)
        ax.errorbar(x, y - k * step, e, fmt=".", color=DARK, ms=3, lw=.6)
        ax.plot(np.sort(phase_h[select]), model(np.sort(phase[select])) - k * step, color=ORANGE, lw=1)
        ax.text(3 * duration * 24, -k * step + .2 * step, f"BJD {epoch + n * period:.2f}", fontsize=7, ha="right", color=BLUE)
    ax.set_xlabel("hours from mid-transit")
    ax.set_ylabel("flux (ppt), offset per transit")
    ax.set_title("individual transits" + ("" if len(shown) == len(transits) else f", first {len(shown)} of {len(transits)}"), fontsize=10)
    save(fig, "transits.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    for parity, color, label in ((1, BLUE, "odd"), (0, ORANGE, "even")):
        select = near & (number % 2 == parity)
        if select.sum() > 3:
            x, y, e = binned(phase_h[select], flux[select], width_h)
            ax.errorbar(x, y, e, fmt=".", color=color, ms=4, lw=.8, label=label)
    sigma = metrics.get("odd_even_sigma")
    ax.set_title("odd / even transits" + (f", depths {sigma:.1f} sigma apart" if sigma is not None else ""), fontsize=10)
    ax.set_xlabel("hours from mid-transit")
    ax.set_ylabel("flux (ppt)")
    ax.legend(fontsize=8, frameon=False)
    save(fig, "odd_even.png")

    if maxima is not None and len(maxima[0]):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(maxima[0], maxima[1], ".", color=DARK, ms=3)
        ax.axvline(period, color=ORANGE, lw=1, alpha=.8, label=f"P = {period:.3f} d")
        ax.set_xscale("log")
        ax.set_xlabel("period (d)")
        ax.set_ylabel("SNR")
        ax.set_title("period maxima above threshold, first pass", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        save(fig, "period_maxima.png")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    orbit = wrap(phase + period / 2, period) / period + .5
    x, y, e = binned(orbit, flux, max(duration / period / 2, 1 / 400))
    ax.errorbar(x, y, e, fmt=".", color=DARK, ms=3, lw=.6)
    if metrics.get("secondary_phase") is not None:
        half = duration / period / 2
        ax.axvspan(metrics["secondary_phase"] - half, metrics["secondary_phase"] + half, color=ORANGE, alpha=.3,
                   label=f"strongest secondary window, {metrics.get('secondary_sigma') or 0:.1f} sigma")
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_xlabel("orbital phase (transit at 0.5)")
    ax.set_ylabel("flux (ppt)")
    ax.set_title("full orbit", fontsize=10)
    save(fig, "orbit.png")
    return written


def pixel_figure(images, title, folder):
    """Out-of-transit and difference images per sector, with the target (x) and the fitted source (+)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    images = [i for i in images if i["source"] is not None]
    if not images:
        return None
    fig, axes = plt.subplots(2, len(images), figsize=(3.2 * len(images), 6.2), squeeze=False)
    for col, image in enumerate(images):
        for row, (data, label) in enumerate(((image["oot"], "out of transit"), (image["difference"], "out - in transit"))):
            ax = axes[row, col]
            ax.imshow(data, origin="lower", cmap="gray_r", interpolation="nearest")
            ax.plot(*image["target"], "x", color=BLUE, ms=9, mew=2, label="target")
            ax.plot(image["source"]["x"], image["source"]["y"], "+", color=ORANGE, ms=11, mew=2, label="difference source")
            ax.set_title(f"sector {image['sector']}, {label}" + (f"\n{image['transits']} transits, source SNR {image['source']['snr']:.1f}" if row else ""), fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(title, fontsize=10)
    path = Path(folder) / "pixels.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
