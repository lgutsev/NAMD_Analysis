"""CLI for ``namd-analysis regime-analysis``.

Analyses declared windows separately and asks whether one kinetic description
is adequate for both.  The early window defaults to 0-100 ps because that is
the interval being characterized; the late window has no default, because no
manuscript fit window is encoded anywhere in this repository and inventing one
would put a number nobody chose into the science.
"""

from __future__ import annotations

import argparse
import glob as globlib
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .kinetics import KineticsError, parse_edges
from .observables import LEGEND
from .populations import ConfigError, StateMap, group_series, load_population_set
from .provenance import environment, fingerprint
from .regimes import (
    DEFAULT_EARLY_WINDOW_NS,
    EARLY_WINDOW_NOTE,
    LATE_WINDOW_NOTE,
    RegimeError,
    compare_piecewise,
    early_transient_narrative,
    fit_regime,
)
from .report import prepare_output, write_csv, write_json
from .transient import TransientError, Window, parse_window

REGIME_ROW_HEADER = [
    "regime",
    "group",
    "window_start_ns",
    "window_end_ns",
    "initial_population",
    "final_population",
    "net_change",
    "peak_population",
    "peak_time_ns",
    "minimum_population",
    "integrated_population_ns",
    "observable_class",
]


def _expand(patterns: Sequence[str]) -> List[Path]:
    found: List[Path] = []
    for pattern in patterns:
        matches = sorted(globlib.glob(pattern))
        if not matches:
            candidate = Path(pattern)
            if candidate.exists():
                matches = [str(candidate)]
            else:
                raise SystemExit(f"error: no file matches {pattern!r}")
        found.extend(Path(item) for item in matches)
    unique, seen = [], set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def build_parser(prog: str = "namd-analysis regime-analysis") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Characterize an early and a late dynamical regime separately, then "
            "compare one global kinetic model against early-plus-late. The early "
            "window defaults to 0-100 ps; the late window must be supplied, "
            "because no manuscript fit window is encoded in this repository."
        ),
        epilog=LATE_WINDOW_NOTE,
    )
    parser.add_argument("--files", nargs="+", required=True, help="SHPROP histories or globs")
    parser.add_argument("--config", required=True, help="SHPROP state-map JSON")
    parser.add_argument(
        "--early-window",
        default=f"{DEFAULT_EARLY_WINDOW_NS[0]:g}:{DEFAULT_EARLY_WINDOW_NS[1]:g}",
        help=(
            "START:END in ns for the fast transient (default 0:0.1, i.e. 0-100 ps). "
            "A choice, not a constant"
        ),
    )
    parser.add_argument(
        "--late-window",
        required=True,
        help=(
            "START:END in ns for the slow regime. REQUIRED: the manuscript fit "
            "window is not encoded anywhere here, so it is never defaulted"
        ),
    )
    parser.add_argument(
        "--scheme",
        required=True,
        help="kinetic graph, e.g. 'BCF->PCBM,PCBM->VBM'. Applied to both windows",
    )
    parser.add_argument(
        "--criterion",
        choices=("aic", "aicc", "bic"),
        default="aicc",
        help="information criterion for the global-vs-piecewise comparison",
    )
    parser.add_argument("--n-starts", type=int, default=5, help="multistart attempts per fit")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="whole-file resamples for rate intervals; needs at least two histories",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--acceptor", default="PCBM", help="group named in the reviewer questions"
    )
    parser.add_argument("--donor", default="BCF")
    parser.add_argument("--ground", default="VBM")
    parser.add_argument("--out", required=True, help="new output directory")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _plot(out: Path, time_ns, series, early: Window, late: Window, regimes) -> List[str]:
    """Early and late on separate axes, so a ns scale cannot hide 100 ps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs: List[str] = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, window, title in (
        (axes[0], early, "early regime"),
        (axes[1], late, "late regime"),
    ):
        low, high = window.resolve(time_ns)
        mask = (time_ns >= low) & (time_ns <= high)
        for name, values in series:
            axis.plot(time_ns[mask] * 1000.0, np.asarray(values)[mask], label=name)
        axis.set(
            xlabel="Time (ps)",
            ylabel="Population",
            title=f"{title}  [{low:g}, {high:g}] ns",
            ylim=(-0.03, 1.03),
        )
        axis.legend(fontsize=8)
    # Separate axes on purpose: sharing one would compress the fast transient
    # into the first pixel of a nanosecond scale.
    fig.suptitle(
        "early and late regimes on independent axes; the fast transient is not "
        "compressed by the late time scale",
        fontsize=8,
    )
    for suffix in ("png", "pdf"):
        path = out / f"regime_populations.{suffix}"
        fig.savefig(path, dpi=200)
        outputs.append(str(path))
    plt.close(fig)

    # Fitted curves against the observations, per regime.
    fitted = [r for r in regimes if r.fit is not None]
    if fitted:
        fig, axes = plt.subplots(
            1, len(fitted), figsize=(5.5 * len(fitted), 4.0),
            constrained_layout=True, squeeze=False,
        )
        for column, regime in enumerate(fitted):
            axis = axes[0][column]
            fit = regime.fit
            for index, name in enumerate(fit.groups):
                line = axis.plot(
                    fit.time_ns * 1000.0, fit.observed[:, index], lw=1.2, label=f"{name} observed"
                )
                axis.plot(
                    fit.time_ns * 1000.0,
                    fit.model[:, index],
                    "--",
                    lw=1.0,
                    color=line[0].get_color(),
                    label=f"{name} fitted",
                )
            axis.set(
                xlabel="Time (ps)",
                ylabel="Population",
                title=f"{regime.name}  [{regime.start_ns:g}, {regime.end_ns:g}] ns",
                ylim=(-0.03, 1.03),
            )
            axis.legend(fontsize=7, ncol=2)
        for suffix in ("png", "pdf"):
            path = out / f"regime_fits.{suffix}"
            fig.savefig(path, dpi=200)
            outputs.append(str(path))
        plt.close(fig)
    return outputs


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = _expand(args.files)
        state_map = StateMap.from_json(args.config)
        population = load_population_set(paths, state_map)
        series = [(g.name, g.values) for g in group_series(population, state_map)]
        groups = [name for name, _ in series]
        observed = np.column_stack([values for _, values in series])
        time_ns = population.time_ns

        early_window = parse_window(args.early_window, name="early")
        late_window = parse_window(args.late_window, name="late")
        edges = parse_edges(args.scheme, groups)

        per_file = None
        if population.stack is not None and args.bootstrap:
            columns = [state_map.groups[name] for name in groups]
            per_file = np.stack(
                [population.stack[:, :, cols].sum(axis=2) for cols in columns], axis=2
            )

        early = fit_regime(
            "early", early_window, time_ns, observed, groups, edges,
            per_file=per_file, n_starts=args.n_starts,
            n_bootstrap=args.bootstrap, seed=args.bootstrap_seed,
        )
        late = fit_regime(
            "late", late_window, time_ns, observed, groups, edges,
            per_file=per_file, n_starts=args.n_starts,
            n_bootstrap=args.bootstrap, seed=args.bootstrap_seed,
        )
        comparison = compare_piecewise(
            time_ns, observed, groups, edges, early, late,
            criterion=args.criterion, n_starts=args.n_starts,
        )
        narrative = early_transient_narrative(
            early, late, acceptor=args.acceptor, donor=args.donor, ground=args.ground
        )
    except (
        RegimeError, TransientError, KineticsError, ConfigError, ValueError, OSError
    ) as exc:
        print(f"error: {exc}")
        return 2

    out = prepare_output(args.out, overwrite=args.overwrite)
    rows = []
    for regime in (early, late):
        for metric in regime.metrics:
            rows.append(
                [
                    regime.name, metric.group, regime.start_ns, regime.end_ns,
                    metric.initial_population, metric.final_population,
                    metric.net_change, metric.peak_population, metric.peak_time_ns,
                    metric.minimum_population, metric.integrated_population_ns,
                    "observed_from_SHPROP",
                ]
            )
    write_csv(out / "regime_observables.csv", REGIME_ROW_HEADER, rows)

    rate_rows = []
    for regime in (early, late):
        if regime.fit is None:
            continue
        for rate in regime.fit.rates:
            entry = rate.as_dict()
            rate_rows.append(
                [
                    regime.name, entry.get("name"), entry.get("value_per_ns"),
                    entry.get("standard_error_per_ns"), entry.get("relative_standard_error"),
                    entry.get("identified"), entry.get("timescale_ns"), "model_inferred",
                ]
            )
    write_csv(
        out / "regime_rates.csv",
        ["regime", "rate", "value_per_ns", "standard_error_per_ns",
         "relative_standard_error", "identified", "timescale_ns", "observable_class"],
        rate_rows,
    )

    figures = _plot(out, time_ns, series, early_window, late_window, [early, late])

    payload = {
        "command": "regime-analysis",
        "environment": environment(),
        "inputs": fingerprint(paths + [Path(args.config)]),
        "state_map": state_map.as_dict(),
        "windows": {
            "early": {
                "requested": args.early_window,
                "resolved_ns": [early.start_ns, early.end_ns],
                "source": "default" if args.early_window == "0:0.1" else "explicit_cli",
                "note": EARLY_WINDOW_NOTE,
            },
            "late": {
                "requested": args.late_window,
                "resolved_ns": [late.start_ns, late.end_ns],
                "source": "explicit_cli",
                "note": LATE_WINDOW_NOTE,
            },
        },
        "scheme": args.scheme,
        "regimes": [early.as_dict(), late.as_dict()],
        "model_comparison": comparison,
        "early_transient_questions": narrative,
        "observable_class_legend": LEGEND,
        "interpretation_limits": [
            "the early window is a declared choice, not a feature the data selected",
            "the late window is supplied by the user; no manuscript fit window is "
            "encoded in this repository",
            "an unidentifiable window means the data do not constrain the rates, "
            "not that the rates are small",
            "the model comparison is descriptive: a piecewise preference does not "
            "establish a mechanism and does not locate a transition, because the "
            "boundary was chosen rather than fitted",
            "net population change is not a flux and co-movement is not transfer",
        ],
        "figures": figures,
    }
    write_json(out / "report.json", payload)

    from .regime_summary import write as write_regime_summary

    summary = write_regime_summary(out / "early_late_summary.md", payload)

    print(f"early  [{early.start_ns:g}, {early.end_ns:g}] ns: "
          f"{early.n_points} points, {early.identifiability['status']}")
    print(f"late   [{late.start_ns:g}, {late.end_ns:g}] ns: "
          f"{late.n_points} points, {late.identifiability['status']}")
    print(f"model comparison ({args.criterion}): {comparison.get('status')}"
          + (f" -> {comparison['preferred']} by {abs(comparison['delta']):.4g}"
             if comparison.get("preferred") else ""))
    print("  this is model comparison, not proof of a mechanistic transition")
    for finding in narrative["findings"][:3]:
        print(f"  {finding['question']}")
        print(f"    {finding['answer'][:160]}")
    print(f"summary: {summary}")
    print(f"written to {out}")
    return 0
