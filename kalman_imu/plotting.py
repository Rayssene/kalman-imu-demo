"""Comparison figure: estimated tilt angles and their errors over time."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: render straight to file
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .filters import rmse  # noqa: E402

# Fixed categorical order (blue, orange, aqua); ground truth in neutral ink.
COLORS = {
    "true": "#0b0b0b",
    "accel": "#2a78d6",
    "complementary": "#eb6834",
    "kalman": "#1baf7a",
}
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _style_axis(ax: plt.Axes) -> None:
    """Recessive grid and spines so the data carries the figure."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.xaxis.label.set_color(TEXT_SECONDARY)


def plot_comparison(t: np.ndarray, true_angle: np.ndarray, accel: np.ndarray,
                    complementary: np.ndarray, kalman: np.ndarray,
                    output: str | Path, disturbance_windows=((4.0, 5.0), (9.0, 10.0), (15.0, 16.0))) -> Path:
    """Plot all estimates against the truth (top) and their errors (bottom).

    Angles are given in radians and plotted in degrees. Returns the saved path.
    """
    deg = np.rad2deg
    estimates = [
        ("Accelerometer only", accel, COLORS["accel"], 0.8, 0.55),
        ("Complementary filter", complementary, COLORS["complementary"], 1.6, 1.0),
        ("Kalman filter", kalman, COLORS["kalman"], 2.0, 1.0),
    ]

    fig, (ax_angle, ax_err) = plt.subplots(
        2, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    fig.patch.set_facecolor(SURFACE)

    for ax in (ax_angle, ax_err):
        _style_axis(ax)
        # Shade the vibration bursts that corrupt the accelerometer.
        for i, (start, end) in enumerate(disturbance_windows):
            ax.axvspan(start, end, color="#f0efec", zorder=0,
                       label="Vibration burst" if (i == 0 and ax is ax_angle) else None)

    # Top: angles.
    for name, est, color, lw, alpha in estimates:
        label = f"{name} (RMSE {deg(rmse(est, true_angle)):.2f}°)"
        ax_angle.plot(t, deg(est), color=color, linewidth=lw, alpha=alpha, label=label)
    ax_angle.plot(t, deg(true_angle), color=COLORS["true"], linewidth=1.4,
                  linestyle="--", label="True angle")
    ax_angle.set_ylabel("Tilt angle [°]")
    fig.suptitle("Tilt estimation from a noisy IMU", x=0.01, ha="left",
                 fontsize=13, color=COLORS["true"])
    # Legend above the plot so it never hides data; truth listed first.
    handles, labels = ax_angle.get_legend_handles_labels()
    order = [len(handles) - 1] + list(range(1, len(handles) - 1)) + [0]
    ax_angle.legend([handles[i] for i in order], [labels[i] for i in order],
                    loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3,
                    fontsize=9, frameon=False, borderaxespad=0.3)

    # Bottom: errors relative to the truth.
    for name, est, color, lw, alpha in estimates:
        ax_err.plot(t, deg(est - true_angle), color=color, linewidth=lw, alpha=alpha)
    ax_err.axhline(0.0, color=COLORS["true"], linewidth=0.8)
    ax_err.set_ylabel("Error [°]")
    ax_err.set_xlabel("Time [s]")
    ax_err.set_xlim(t[0], t[-1])

    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output
