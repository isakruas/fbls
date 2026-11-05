"""Supporting figure of a candidate for ExoFOP: folded light curve with the fitted trapezoid,
odd and even transits, and the full orbit with the strongest secondary window."""
import numpy as np

from .vetting import trapezoid, wrap

BIN_MINUTES = 10.


def binned(x, y, width):
    cells = np.floor(x / width).astype(int)
    keys, inverse, counts = np.unique(cells, return_inverse=True, return_counts=True)
    mean_x = np.bincount(inverse, x) / counts
    mean_y = np.bincount(inverse, y) / counts
    scatter = np.sqrt(np.maximum(np.bincount(inverse, y ** 2) / counts - mean_y ** 2, 0)) / np.sqrt(counts)
    return mean_x, mean_y, scatter


def candidate_figure(prep, metrics, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    period, duration = prep["period"], prep["duration"]
    valid = prep["valid"]
    phase_h, flux = prep["phase"][valid] * 24, prep["corrected"][valid] * 1e3
    width_h = BIN_MINUTES / 60
    depth = (metrics.get("depth_ppm") or 0) / 1e6
    t14 = (metrics.get("fitted_duration_hours") or metrics["duration_hours"]) / 24
    ingress = metrics.get("ingress_fraction") or .2
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    near = np.abs(phase_h) < 3 * duration * 24
    for ax, select, label in ((axes[0], near, "all transits"),):
        x, y, e = binned(phase_h[select], flux[select], width_h)
        ax.errorbar(x, y, e, fmt=".", color="#1f2429", ms=4, lw=.8, label=f"{BIN_MINUTES:.0f}-min bins")
        grid = np.linspace(-3 * duration, 3 * duration, 600)
        ax.plot(grid * 24, trapezoid(grid, depth, t14, ingress) * 1e3, color="#b4521a", lw=1.5, label="trapezoid fit")
        ax.set_xlabel("hours from mid-transit")
        ax.set_ylabel("relative flux (ppt)")
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.set_title(label, fontsize=10)
    number = np.round((prep["time"][valid] - prep["epoch"]) / period)
    for parity, color, label in ((1, "#2a6f97", "odd"), (0, "#b4521a", "even")):
        select = near & (number % 2 == parity)
        if select.sum() > 3:
            x, y, e = binned(phase_h[select], flux[select], width_h)
            axes[1].errorbar(x, y, e, fmt=".", color=color, ms=4, lw=.8, label=label)
    axes[1].set_xlabel("hours from mid-transit")
    axes[1].set_title(f"odd / even, {metrics.get('odd_even_sigma') or 0:.1f} sigma apart", fontsize=10)
    axes[1].legend(fontsize=8, frameon=False)
    orbit = wrap(prep["phase"][valid] + period / 2, period) / period + .5
    x, y, e = binned(orbit, flux, max(duration / period / 2, 1 / 400))
    axes[2].errorbar(x, y, e, fmt=".", color="#1f2429", ms=3, lw=.6)
    if metrics.get("secondary_phase") is not None:
        axes[2].axvspan(metrics["secondary_phase"] - duration / period / 2, metrics["secondary_phase"] + duration / period / 2,
                        color="#b4521a", alpha=.25, label=f"secondary window, {metrics.get('secondary_sigma') or 0:.1f} sigma")
        axes[2].legend(fontsize=8, frameon=False, loc="lower right")
    axes[2].set_xlabel("orbital phase")
    axes[2].set_title("full orbit", fontsize=10)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
