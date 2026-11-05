"""Supporting figure of a candidate for ExoFOP and the paper: the detrended light curve with the
transits marked, the folded curve with the fitted trapezoid and its residuals, each transit on
its own, odd against even transits, and the full orbit with the strongest secondary window."""
import numpy as np

from .vetting import trapezoid, wrap

BIN_MINUTES = 10.
DARK, ORANGE, BLUE = "#1f2429", "#b4521a", "#2a6f97"


def binned(x, y, width):
    cells = np.floor(x / width).astype(int)
    _, inverse, counts = np.unique(cells, return_inverse=True, return_counts=True)
    mean_x = np.bincount(inverse, x) / counts
    mean_y = np.bincount(inverse, y) / counts
    scatter = np.sqrt(np.maximum(np.bincount(inverse, y ** 2) / counts - mean_y ** 2, 0)) / np.sqrt(counts)
    return mean_x, mean_y, scatter


def candidate_figure(prep, metrics, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
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

    fig = plt.figure(figsize=(14, 8))
    grid = GridSpec(3, 4, figure=fig, height_ratios=[1.1, 1.3, 1.], hspace=.55, wspace=.35)

    ax = fig.add_subplot(grid[0, :])
    x, y, _ = binned(time, flux, BIN_MINUTES / 1440)
    ax.plot(x, y, ".", color=DARK, ms=2, alpha=.6)
    for n in transits:
        ax.axvline(epoch + n * period, color=ORANGE, lw=1, alpha=.8)
    ax.set_xlabel("BJD")
    ax.set_ylabel("flux (ppt)")
    ax.set_title(f"detrended light curve, {BIN_MINUTES:.0f}-min bins; {len(transits)} transits marked", fontsize=10)

    ax = fig.add_subplot(grid[1, :2])
    x, y, e = binned(phase_h[near], flux[near], width_h)
    ax.errorbar(x, y, e, fmt=".", color=DARK, ms=4, lw=.8, label=f"{BIN_MINUTES:.0f}-min bins")
    ax.plot(phase_h[near][np.argsort(phase_h[near])], model(np.sort(phase[near])), color=ORANGE, lw=1.5, label="trapezoid fit")
    ax.set_ylabel("flux (ppt)")
    ax.set_title("folded on the period", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.tick_params(labelbottom=False)
    ax_res = fig.add_subplot(grid[2, :2], sharex=ax)
    x, y, e = binned(phase_h[near], flux[near] - model(phase[near]), width_h)
    ax_res.errorbar(x, y, e, fmt=".", color=DARK, ms=4, lw=.8)
    ax_res.axhline(0, color=ORANGE, lw=1)
    ax_res.set_xlabel("hours from mid-transit")
    ax_res.set_ylabel("residual (ppt)")

    ax = fig.add_subplot(grid[1, 2])
    offset = 0.
    step = max(4 * depth * 1e3, 4 * np.nanstd(flux[near]) / np.sqrt(max(1, (near & (number == transits[0])).sum() / 5)) if transits else 1.)
    for n in transits:
        select = near & (number == n)
        x, y, e = binned(phase_h[select], flux[select], width_h)
        ax.errorbar(x, y - offset, e, fmt=".", color=DARK, ms=3, lw=.6)
        ax.plot(np.sort(phase_h[select]), model(np.sort(phase[select])) - offset, color=ORANGE, lw=1)
        ax.text(3 * duration * 24, -offset + .2 * step, f"{epoch + n * period:.2f}", fontsize=7, ha="right", color=BLUE)
        offset += step
    ax.set_xlabel("hours from mid-transit")
    ax.set_title("individual transits (offset)", fontsize=10)

    ax = fig.add_subplot(grid[1, 3])
    for parity, color, label in ((1, BLUE, "odd"), (0, ORANGE, "even")):
        select = near & (number % 2 == parity)
        if select.sum() > 3:
            x, y, e = binned(phase_h[select], flux[select], width_h)
            ax.errorbar(x, y, e, fmt=".", color=color, ms=4, lw=.8, label=label)
    sigma = metrics.get("odd_even_sigma")
    ax.set_title("odd / even" + (f", {sigma:.1f} sigma apart" if sigma is not None else ""), fontsize=10)
    ax.set_xlabel("hours from mid-transit")
    ax.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(grid[2, 2:])
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

    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
