"""The focus-band figure.  Every point drawn here is a row of ``band_<b>_focus.csv``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .projection_robustness import FrameMark, RobustnessAudit  # noqa: E402

#: Fragment colours in fixed slot order, assigned by the table's group order
#: so a fragment keeps its colour across every panel and every run.  The first
#: three slots of the reference categorical palette, validated all-pairs for
#: scatter use.  A fourth group or beyond falls back to neutral ink.
FRAGMENT_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
UNCAPTURED_COLOUR = "#8a8984"
WINDOW_FILL = "#e4e3df"
CONTROL_FILL = "#eeeef6"
GRID = "#e6e5e1"

MARK_STYLE = {
    "low_capture": "v",
    "high_capture_control": "^",
    "typical": "D",
    "most_mixed": "*",
    "user": "o",
}


def _colour(index: int) -> str:
    return FRAGMENT_COLOURS[index] if index < len(FRAGMENT_COLOURS) else INK_SECONDARY


def _style(axis) -> None:
    axis.grid(True, color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK_SECONDARY)
    axis.tick_params(colors=INK_SECONDARY, labelsize=8)


def _shade_windows(axis, windows: Sequence[Any], label_top: bool = False) -> None:
    for window in windows:
        fill = CONTROL_FILL if window.role == "control" else WINDOW_FILL
        axis.axvspan(window.first - 0.5, window.last + 0.5, color=fill, zorder=0, linewidth=0)
        if label_top:
            axis.annotate(
                window.name + (" (control)" if window.role == "control" else ""),
                xy=((window.first + window.last) / 2.0, 0.02),
                xycoords=("data", "axes fraction"),
                ha="center", va="bottom", fontsize=7, color=INK_SECONDARY,
            )


def _overlay_marks(axis, frames: np.ndarray, x: np.ndarray, y: np.ndarray,
                   marks: Sequence[FrameMark], legend: bool) -> None:
    lookup = {int(f): i for i, f in enumerate(frames)}
    for mark in marks:
        index = [lookup[f] for f in mark.frames if f in lookup]
        if not index:
            continue
        axis.scatter(
            x[index], y[index],
            marker=MARK_STYLE.get(mark.role, "o"),
            s=46, facecolor="white" if mark.role == "user" else INK,
            edgecolor=INK, linewidth=1.1, zorder=5,
            label=f"{mark.label} ({', '.join(str(f) for f in mark.frames)})" if legend else None,
        )


def _shared_legends(axes, series_axis, marks_axis) -> None:
    """One legend for the series and one for the marks, below the panels.

    Colour follows the fragment in every panel, so it is keyed once; the
    marks are keyed once, each with the frames it stands for.
    """
    series: dict = {}
    marks: dict = {}
    for axis in axes.flat:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label.startswith("_"):
                continue
            target = marks if axis is axes[0, 0] and "(" in label else series
            target.setdefault(label, handle)
    from matplotlib.lines import Line2D

    series["one frame (dot colour in d, e: that fragment)"] = Line2D(
        [], [], linestyle="none", marker="o", markersize=3.5,
        markerfacecolor=INK_SECONDARY, markeredgewidth=0,
    )
    for legend_axis, entries, title, columns in (
        (series_axis, series, "series", 2),
        (marks_axis, marks, "marked frames", 1),
    ):
        legend_axis.axis("off")
        if entries:
            legend_axis.legend(
                list(entries.values()), list(entries.keys()), loc="upper left",
                ncol=columns, fontsize=7.5, frameon=False, title=title, title_fontsize=8,
            )


def plot_focus_band(
    result: RobustnessAudit,
    band: int,
    marks: Sequence[FrameMark],
    windows: Sequence[Any],
    stem: Path,
    quality_threshold: Optional[float] = None,
) -> List[str]:
    table = result.table
    bi = table.band_position(band)
    frames = table.frames
    order = np.argsort(frames)
    frames_sorted = frames[order]
    captured = table.captured[order, bi]
    raw = result.arrays.raw[order, bi, :]
    weights = table.weights[order, bi, :]
    uncaptured = 1.0 - captured
    ia = table.group_position(result.pair[0])
    ib = table.group_position(result.pair[1])
    in_window = np.zeros(len(frames_sorted), dtype=bool)
    for window in windows:
        if window.role != "control":
            in_window |= (frames_sorted >= window.first) & (frames_sorted <= window.last)

    figure, grid = plt.subplots(
        4, 2, figsize=(12.0, 12.4), constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 0.30]},
    )
    axes = grid[:3, :]
    legend_axis = grid[3, 0]
    legend_axis_marks = grid[3, 1]
    series_axes = [axes[0, 0], axes[1, 0], axes[2, 0]]
    for axis in series_axes[1:]:
        axis.sharex(series_axes[0])

    # a: capture against frame
    axis = axes[0, 0]
    _shade_windows(axis, windows, label_top=True)
    axis.plot(frames_sorted, captured, color=INK_SECONDARY, linewidth=0.8, label="captured_projection")
    if quality_threshold is not None:
        axis.axhline(quality_threshold, color=INK, linewidth=0.8, linestyle=(0, (4, 3)),
                     label=f"existing reporting threshold {quality_threshold:g}")
    _overlay_marks(axis, frames_sorted, frames_sorted.astype(float), captured, marks, legend=True)
    axis.set_ylabel("captured projection")
    axis.set_title(f"a   band {band}: captured projection by MD frame", loc="left", fontsize=10)

    # b: raw weights against frame
    axis = axes[1, 0]
    _shade_windows(axis, windows)
    for gi, name in enumerate(table.groups):
        axis.plot(frames_sorted, raw[:, gi], color=_colour(gi), linewidth=0.8, label=f"W {name}")
    axis.plot(frames_sorted, uncaptured, color=UNCAPTURED_COLOUR, linewidth=0.8,
              linestyle=(0, (3, 2)), label="1 - captured (uncaptured)")
    axis.set_ylabel("raw weight (fraction of the band)")
    axis.set_title("b   raw PAW-projector weights W_g, before normalization", loc="left", fontsize=10)

    # c: normalized weights against frame
    axis = axes[2, 0]
    _shade_windows(axis, windows)
    for gi, name in enumerate(table.groups):
        axis.plot(frames_sorted, weights[:, gi], color=_colour(gi), linewidth=0.8, label=f"w {name}")
    axis.set_ylabel("normalized fraction w_g")
    axis.set_xlabel("MD frame")
    axis.set_title("c   normalized fractions w_g = W_g / captured", loc="left", fontsize=10)

    # d, e: capture against each pair fragment's normalized fraction
    for axis, gi, letter in ((axes[0, 1], ia, "d"), (axes[1, 1], ib, "e")):
        name = table.groups[gi]
        axis.scatter(captured[~in_window], weights[~in_window, gi], s=6,
                     color=_colour(gi), alpha=0.55, linewidth=0, label="_frames")
        if np.any(in_window):
            axis.scatter(captured[in_window], weights[in_window, gi], s=20, marker="s",
                         facecolor="none", edgecolor=INK_SECONDARY, linewidth=0.9,
                         label="frames in a crossing window")
        _overlay_marks(axis, frames_sorted, captured, weights[:, gi], marks, legend=False)
        axis.set_xlabel("captured projection")
        axis.set_ylabel(f"normalized {name} fraction")
        axis.set_title(f"{letter}   captured projection vs normalized {name}", loc="left", fontsize=10)

    # f: raw weight of one pair fragment against the other
    axis = axes[2, 1]
    axis.scatter(raw[~in_window, ia], raw[~in_window, ib], s=6, color=INK_SECONDARY,
                 alpha=0.5, linewidth=0, label="_frames")
    if np.any(in_window):
        axis.scatter(raw[in_window, ia], raw[in_window, ib], s=20, marker="s",
                     facecolor="none", edgecolor=INK_SECONDARY, linewidth=0.9,
                     label="frames in a crossing window")
    _overlay_marks(axis, frames_sorted, raw[:, ia], raw[:, ib], marks, legend=False)
    top = float(max(np.max(raw[:, ia]), np.max(raw[:, ib]), 1e-3))
    axis.plot([0.0, top], [0.0, top], color=GRID, linewidth=0.8, zorder=0, label="W equal")
    axis.set_xlabel(f"raw {result.pair[0]} weight W")
    axis.set_ylabel(f"raw {result.pair[1]} weight W")
    axis.set_title(f"f   raw {result.pair[0]} vs raw {result.pair[1]}", loc="left", fontsize=10)

    for axis in axes.flat:
        _style(axis)
    _shared_legends(axes, legend_axis, legend_axis_marks)

    figure.suptitle(
        f"Band {band}: projection robustness (pair {result.pair[0]}/{result.pair[1]}). "
        "Nothing is filtered; every point is a row of the focus CSV.",
        fontsize=10, color=INK,
    )
    written = []
    for extension in ("png", "pdf"):
        path = stem.with_suffix(f".{extension}")
        figure.savefig(path, dpi=200)
        written.append(str(path))
    plt.close(figure)
    return written


def plot_sensitivity(result: RobustnessAudit, band: int, stem: Path) -> List[str]:
    """Mixed-sample count against the normalized threshold, one line per raw minimum.

    Answers the Issue #4 question at a glance: if the lines for a non-zero raw
    minimum collapse to zero while the unconditioned line does not, the
    mixing lives only in small absolute weights.
    """
    table = result.table
    bi = table.band_position(band)
    ia = table.group_position(result.pair[0])
    ib = table.group_position(result.pair[1])
    w_a, w_b = table.weights[:, bi, ia], table.weights[:, bi, ib]
    raw_a, raw_b = result.arrays.raw[:, bi, ia], result.arrays.raw[:, bi, ib]
    taus = result.grids["normalized_thresholds"]
    raw_minima = list(result.grids["raw_minima"])
    # Ordinal blue ramp, dark to light, stopping at step 250 so the lightest
    # line still clears 2:1 against the light surface.
    ramp = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#3987e5",
            "#6da7ec", "#86b6ef"]
    figure, axis = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    if len(raw_minima) > len(ramp):
        # Evenly spaced, always keeping the unconditioned (0) and the largest.
        picks = np.unique(np.round(np.linspace(0, len(raw_minima) - 1, len(ramp))).astype(int))
        shown = [raw_minima[i] for i in picks]
    else:
        shown = raw_minima
    for k, r_min in enumerate(shown):
        counts = [
            int(np.count_nonzero((w_a >= t) & (w_b >= t) & (raw_a >= r_min) & (raw_b >= r_min)))
            for t in taus
        ]
        axis.plot(taus, counts, color=ramp[k], linewidth=1.6, marker="o", markersize=4,
                  label=f"raw >= {r_min:g} on each")
    _style(axis)
    axis.set_xlabel("normalized threshold on each pair fragment")
    axis.set_ylabel(f"mixed samples on band {band} (of {len(table.frames)})")
    axis.set_title(
        f"Band {band}: {result.pair[0]}/{result.pair[1]}-mixed samples, no capture condition",
        loc="left", fontsize=10,
    )
    axis.legend(fontsize=7, frameon=False, title="raw minimum", title_fontsize=7)
    written = []
    for extension in ("png", "pdf"):
        path = stem.with_suffix(f".{extension}")
        figure.savefig(path, dpi=200)
        written.append(str(path))
    plt.close(figure)
    return written
