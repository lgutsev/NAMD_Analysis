"""Reviewer-facing figures, built only from the machine-readable outputs.

Every function here reads ``ensemble_curves.json``, ``run_summary.json`` or
``per_history.csv`` and plots what it finds.  Nothing is recomputed from the
SHPROP histories, so any number on any axis can be traced to a file on disk --
which is the point: a figure that cannot be checked against its source is not
evidence.

Two labelling rules are enforced rather than left to the caller:

* **A shaded band is never a confidence interval.** The band across histories
  is an inter-quantile range over electronic initial conditions at a *fixed*
  nuclear trajectory. Every band carries that in its legend, and no standard
  error over histories is ever drawn.
* **The two decomposition terms are not branching fractions.** Their figure
  carries the bookkeeping caveat in the axis label and the caption.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "PlotError",
    "BAND_LABEL",
    "BOOKKEEPING_CAPTION",
    "load_curves",
    "load_run_summary",
    "load_per_history",
    "plot_run_populations",
    "plot_fixed_vs_dynamic",
    "plot_run_comparison",
    "plot_delta_distributions",
    "plot_decomposition_distributions",
    "write_figure_manifest",
]

BAND_LABEL = "inter-quartile range across histories (electronic initial\nconditions, fixed nuclear trajectory; NOT a CI or SEM)"

BOOKKEEPING_CAPTION = (
    "occupation_redistribution and character_evolution come from an exact, "
    "endpoint-unbiased split of the population change. The split is one of "
    "infinitely many exact splits: these are a bookkeeping convention and NOT "
    "physical branching fractions of the Hamiltonian dynamics."
)

HIERARCHY_CAPTION = (
    "passes are re-traversals of one recycled nuclear trajectory; histories "
    "within a configuration share that trajectory; A/B/C are distinct "
    "interface configurations, not replicates. "
    "Histories are not independent nuclear configurations."
)

#: Consistent across every figure so runs can be compared by eye.
RUN_COLOURS = ("#1b6ca8", "#c0392b", "#117a65", "#8e44ad", "#b9770e")
GROUP_COLOURS = {
    "perovskite": "#5d6d7e",
    "BCF": "#c0392b",
    "PCBM": "#1b6ca8",
    "VBM": "#7f8c8d",
    "CBM": "#16a085",
}


class PlotError(ValueError):
    """Raised when a figure cannot be built from the outputs given."""


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, out: Path, stem: str, manifest: List[Dict[str, Any]],
          sources: Sequence[str], caption: str) -> List[str]:
    """Write PNG and PDF, and record where the numbers came from."""
    written = []
    for suffix in ("png", "pdf"):
        path = Path(out) / f"{stem}.{suffix}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        written.append(path.name)
    _pyplot().close(fig)
    manifest.append({
        "figure": stem,
        "files": written,
        "sources": list(sources),
        "caption": caption,
    })
    return written


def load_curves(path) -> Dict[str, Any]:
    """``ensemble_curves.json`` as written by ``character-ensemble``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "time_ns" not in data or "groups" not in data:
        raise PlotError(f"{path}: not an ensemble_curves.json")
    return data


def load_run_summary(path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "quantities" not in data or "run" not in data:
        raise PlotError(f"{path}: not a run_summary.json")
    return data


def load_per_history(path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise PlotError(f"{path}: no rows")
    return rows


def _column(rows: Sequence[Dict[str, str]], name: str) -> np.ndarray:
    if name not in rows[0]:
        raise PlotError(f"per_history.csv has no column {name!r}")
    values = []
    for row in rows:
        raw = row[name]
        values.append(float(raw) if raw not in ("", None) else np.nan)
    return np.asarray(values, dtype=float)


def plot_run_populations(
    curves: Dict[str, Any],
    run: str,
    out: Path,
    manifest: List[Dict[str, Any]],
    source: str,
    groups: Optional[Sequence[str]] = None,
) -> List[str]:
    """Median subsystem population per fragment, with the history band."""
    plt = _pyplot()
    time = np.asarray(curves["time_ns"], dtype=float)
    names = list(groups or curves.get("dynamic_groups") or curves["groups"])
    names = [g for g in names if not g.startswith("fixed:")]
    if not names:
        raise PlotError("no dynamic groups in the curves file")

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    n_hist = None
    for g in names:
        block = curves["groups"].get(g)
        if not block or "median" in block and block.get("error"):
            continue
        if "error" in block:
            continue
        median = np.asarray(block["median"], dtype=float)
        n_hist = block.get("n_histories", n_hist)
        colour = GROUP_COLOURS.get(g, "#34495e")
        ax.plot(time, median, color=colour, lw=1.6, label=f"{g} (median)")
        if "q25" in block and "q75" in block:
            ax.fill_between(
                time, np.asarray(block["q25"], dtype=float),
                np.asarray(block["q75"], dtype=float),
                color=colour, alpha=0.20, lw=0,
            )
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("projection-weighted diagonal\nsubsystem population")
    ax.set_title(
        f"Run {run}: subsystem population"
        + (f", {n_hist} histories" if n_hist else "")
    )
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25, lw=0.5)
    handles, labels = ax.get_legend_handles_labels()
    band = plt.Rectangle((0, 0), 1, 1, fc="#34495e", alpha=0.20, ec="none")
    ax.legend(handles + [band], labels + [BAND_LABEL], fontsize=7.5,
              loc="upper right", framealpha=0.9)
    caption = (
        f"Run {run}. Median projection-weighted DIAGONAL subsystem population "
        f"across {n_hist or 'the'} SHPROP histories; shaded band is the "
        "inter-quartile range across histories, i.e. variation over electronic "
        "initial conditions at fixed nuclear trajectory. It is not a "
        "confidence interval and not a standard error. The coherence term is "
        "absent from SHPROP and PROCAR alike and is omitted, not estimated. "
        + HIERARCHY_CAPTION
    )
    return _save(fig, out, f"run_{run}_populations", manifest, [source], caption)


def _quoted_stats(stats: Optional[Dict[str, Any]], group: str, late: bool) -> str:
    """Medians read from the run summary, never recomputed from a thinned curve."""
    if not stats:
        return ""
    q = stats.get("quantities", {})

    def median(name):
        block = q.get(name, {}).get(group)
        return block.get("median") if isinstance(block, dict) else None

    net_dyn = median("net_late" if late else "net_full")
    net_fix = median("net_late_fixed" if late else "net_full_fixed")
    parts = []
    if net_fix is not None and net_dyn is not None:
        parts.append(f"median net: fixed {net_fix:+.4f} / dynamic {net_dyn:+.4f}")
    elif net_dyn is not None:
        parts.append(f"median net dynamic {net_dyn:+.4f}")
    if late:
        rng_dyn, rng_fix = median("range_late"), median("range_late_fixed")
        if rng_fix is not None and rng_dyn is not None:
            parts.append(f"median range: fixed {rng_fix:.4f} / dynamic {rng_dyn:.4f}")
        elif rng_dyn is not None:
            parts.append(f"median range dynamic {rng_dyn:.4f}")
    return "   ".join(parts)


def plot_fixed_vs_dynamic(
    curves: Dict[str, Any],
    run: str,
    out: Path,
    manifest: List[Dict[str, Any]],
    source: str,
    late_start_ns: float = 0.1,
    pairs: Sequence[str] = ("BCF", "PCBM"),
    stats: Optional[Dict[str, Any]] = None,
    stats_source: Optional[str] = None,
) -> List[str]:
    """The central R3.2 figure: fixed-column against dynamic projection.

    ``stats`` is a ``run_summary.json``.  When given, every quoted net and
    range is read from it rather than recomputed from the curves, because the
    curves are **thinned for display** by ``--curve-stride``: striding moves
    the endpoints and hides the extrema, so a number recomputed from them can
    differ from the authoritative value by more than the effect being
    discussed.  Without it, no numbers are quoted at all.
    """
    plt = _pyplot()
    time = np.asarray(curves["time_ns"], dtype=float)
    available = curves["groups"]
    usable = [g for g in pairs if g in available and f"fixed:{g}" in available]
    if not usable:
        raise PlotError(
            "no fragment has both a dynamic and a fixed curve; rerun "
            "character-ensemble with --fixed-state-map"
        )

    fig, axes = plt.subplots(
        len(usable), 2, figsize=(10.4, 3.5 * len(usable)), squeeze=False
    )
    for r, g in enumerate(usable):
        dyn = np.asarray(available[g]["median"], dtype=float)
        fix = np.asarray(available[f"fixed:{g}"]["median"], dtype=float)
        for c, (lo, hi, label) in enumerate((
            (float(time[0]), float(time[-1]), "full trajectory"),
            (late_start_ns, float(time[-1]), f"late window, t >= {late_start_ns} ns"),
        )):
            ax = axes[r][c]
            mask = (time >= lo) & (time <= hi)
            if mask.sum() < 2:
                ax.text(0.5, 0.5, "window holds fewer than two samples",
                        ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.set_axis_off()
                continue
            ax.plot(time[mask], fix[mask], color="#7f8c8d", lw=1.4, ls="--",
                    label=f"{g}, fixed column map")
            ax.plot(time[mask], dyn[mask], color=GROUP_COLOURS.get(g, "#1b6ca8"),
                    lw=1.6, label=f"{g}, dynamic projection")
            title = f"{g}, {label}"
            quoted = _quoted_stats(stats, g, c == 1)
            if quoted:
                title += "\n" + quoted
            ax.set_title(title, fontsize=8.5)
            ax.set_xlabel("time (ns)")
            ax.set_ylabel("population")
            ax.grid(alpha=0.25, lw=0.5)
            ax.legend(fontsize=7.5, loc="best", framealpha=0.9)
    fig.suptitle(
        f"Run {run}: fixed-column vs dynamic projection-weighted population",
        y=1.002, fontsize=11,
    )
    caption = (
        f"Run {run}. Median across histories of the fixed-column reading "
        "(dashed) against the projection-weighted diagonal reading (solid), "
        f"over the full trajectory and the late window t >= {late_start_ns} ns. "
        "A difference in NET change means the fixed labelling was wrong "
        "somewhere; a difference in RANGE means the population is not static "
        "even where the net change agrees. Neither curve is subtracted from the "
        "other. Quoted medians are read from the run summary, not from the "
        "displayed curves, which are thinned for plotting. " + HIERARCHY_CAPTION
    )
    sources = [source] + ([stats_source] if stats_source else [])
    return _save(fig, out, f"run_{run}_fixed_vs_dynamic", manifest, sources, caption)


def plot_run_comparison(
    curve_sets: Dict[str, Dict[str, Any]],
    out: Path,
    manifest: List[Dict[str, Any]],
    sources: Sequence[str],
    groups: Sequence[str] = ("perovskite", "BCF", "PCBM"),
) -> List[str]:
    """One column per configuration, identical axes. Nothing pooled."""
    plt = _pyplot()
    runs = list(curve_sets)
    if not runs:
        raise PlotError("no runs to compare")
    fig, axes = plt.subplots(
        1, len(runs), figsize=(4.6 * len(runs), 4.4), squeeze=False, sharey=True
    )
    for c, run in enumerate(runs):
        ax = axes[0][c]
        curves = curve_sets[run]
        time = np.asarray(curves["time_ns"], dtype=float)
        for g in groups:
            block = curves["groups"].get(g)
            if not block or "error" in block:
                continue
            colour = GROUP_COLOURS.get(g, "#34495e")
            median = np.asarray(block["median"], dtype=float)
            ax.plot(time, median, color=colour, lw=1.5, label=g)
            if "q25" in block and "q75" in block:
                ax.fill_between(
                    time, np.asarray(block["q25"], dtype=float),
                    np.asarray(block["q75"], dtype=float),
                    color=colour, alpha=0.18, lw=0,
                )
        n = next(
            (b.get("n_histories") for b in curves["groups"].values()
             if isinstance(b, dict) and b.get("n_histories")), None
        )
        ax.set_title(f"Run {run}" + (f" ({n} histories)" if n else ""), fontsize=10)
        ax.set_xlabel("time (ns)")
        ax.grid(alpha=0.25, lw=0.5)
        if c == 0:
            ax.set_ylabel("projection-weighted diagonal\nsubsystem population")
            ax.set_ylim(-0.02, 1.02)
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "Configurations side by side, identical axes",
        y=1.01, fontsize=11,
    )
    caption = (
        "Each configuration plotted separately on identical axes; bands are "
        "inter-quartile across that configuration's histories. A, B and C are "
        "distinct interface configurations, so they are NOT averaged together: "
        "a mean over different physical systems is not a measurement of "
        "anything, and a difference between them is a result. " + HIERARCHY_CAPTION
    )
    return _save(fig, out, "run_comparison_populations", manifest, sources, caption)


def _strip(ax, data: Sequence[np.ndarray], labels: Sequence[str],
           colours: Sequence[str], seed: int = 0) -> None:
    """Box plus jittered points, so every history is visible."""
    rng = np.random.default_rng(seed)
    clean = [d[np.isfinite(d)] for d in data]
    positions = np.arange(1, len(clean) + 1)
    ax.boxplot(clean, positions=positions, widths=0.5, showfliers=False,
               medianprops={"color": "black", "lw": 1.6})
    for pos, values, colour in zip(positions, clean, colours):
        if values.size == 0:
            continue
        x = pos + rng.uniform(-0.13, 0.13, values.size)
        ax.scatter(x, values, s=9, color=colour, alpha=0.55, lw=0, zorder=3)
    ax.axhline(0.0, color="#7f8c8d", lw=0.8, ls=":")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8.5)


def plot_delta_distributions(
    per_run_rows: Dict[str, List[Dict[str, str]]],
    out: Path,
    manifest: List[Dict[str, Any]],
    sources: Sequence[str],
    donor: str = "BCF",
    acceptor: str = "PCBM",
    sign_tolerance: float = 1.0e-3,
) -> List[str]:
    """Per-history net change, run identity preserved."""
    plt = _pyplot()
    runs = list(per_run_rows)
    if not runs:
        raise PlotError("no runs")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), squeeze=False)
    for c, (prefix, title) in enumerate((
        ("net_full", "full trajectory"),
        ("net_late", "late window"),
    )):
        ax = axes[0][c]
        data, labels, colours = [], [], []
        for r, run in enumerate(runs):
            rows = per_run_rows[run]
            for g in (acceptor, donor):
                column = f"{prefix}_{g}"
                try:
                    values = _column(rows, column)
                except PlotError:
                    values = np.array([np.nan])
                data.append(values)
                finite = values[np.isfinite(values)]
                gain = int((finite > sign_tolerance).sum())
                labels.append(f"{run}\n{g}\n{gain}/{finite.size} gain")
                colours.append(RUN_COLOURS[r % len(RUN_COLOURS)])
        _strip(ax, data, labels, colours, seed=c)
        ax.set_title(f"per-history net change, {title}", fontsize=10)
        ax.set_ylabel(f"dP ({title})")
        ax.grid(alpha=0.22, lw=0.5, axis="y")
    fig.suptitle(
        "Distribution of per-history net change, run identity preserved",
        y=1.01, fontsize=11,
    )
    caption = (
        "One point per SHPROP history; box is the inter-quartile range and the "
        "line the median. Counts under each group are histories with net gain "
        f"above {sign_tolerance:g}. Runs are kept separate so a heterogeneous "
        "picture is visible as heterogeneity rather than averaged away. "
        + HIERARCHY_CAPTION
    )
    return _save(fig, out, "delta_distributions", manifest, sources, caption)


def plot_decomposition_distributions(
    per_run_rows: Dict[str, List[Dict[str, str]]],
    out: Path,
    manifest: List[Dict[str, Any]],
    sources: Sequence[str],
    groups: Sequence[str] = ("BCF", "PCBM"),
) -> List[str]:
    """The two bookkeeping terms, side by side, with the caveat attached."""
    plt = _pyplot()
    runs = list(per_run_rows)
    if not runs:
        raise PlotError("no runs")
    fig, axes = plt.subplots(1, len(groups), figsize=(5.6 * len(groups), 4.8),
                             squeeze=False)
    for c, g in enumerate(groups):
        ax = axes[0][c]
        data, labels, colours = [], [], []
        for r, run in enumerate(runs):
            rows = per_run_rows[run]
            for prefix, short in (
                ("occupation_redistribution", "occupation"),
                ("character_evolution", "character"),
            ):
                try:
                    values = _column(rows, f"{prefix}_{g}")
                except PlotError:
                    values = np.array([np.nan])
                data.append(values)
                labels.append(f"{run}\n{short}")
                colours.append(RUN_COLOURS[r % len(RUN_COLOURS)])
        _strip(ax, data, labels, colours, seed=c)
        ax.set_title(f"{g}: the two terms of the exact split", fontsize=10)
        ax.set_ylabel(
            "summed contribution to dP\n(bookkeeping convention, NOT a\n"
            "physical branching fraction)"
        )
        ax.grid(alpha=0.22, lw=0.5, axis="y")
    fig.suptitle(
        "Decomposition terms per history -- an exact split, not a mechanism ratio",
        y=1.01, fontsize=11,
    )
    caption = BOOKKEEPING_CAPTION + " " + HIERARCHY_CAPTION
    return _save(fig, out, "decomposition_distributions", manifest, sources, caption)


def write_figure_manifest(out: Path, manifest: Sequence[Dict[str, Any]]) -> Path:
    """Every figure, its files, its sources and its caption."""
    path = Path(out) / "figures.json"
    path.write_text(json.dumps({
        "figures": list(manifest),
        "note": (
            "every plotted quantity is read from the machine-readable output "
            "listed under sources; nothing is recomputed from the SHPROP "
            "histories at plot time, so each figure can be checked against its "
            "source file"
        ),
        "band_meaning": BAND_LABEL.replace("\n", " "),
        "bookkeeping_caveat": BOOKKEEPING_CAPTION,
        "hierarchy": HIERARCHY_CAPTION,
    }, indent=2), encoding="utf-8")
    return path
