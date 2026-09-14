"""Command-line interface.

Every subcommand writes a JSON report with the input fingerprints, the
options used, and the checks that ran, so a figure can be traced back to the
files and settings that produced it.
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import __version__
from .audit import DEFAULT_WARNING_FRACTION, PAIR_HEADER, audit_run
from .fitting import FitError, fit_single_exponential, bootstrap_single_exponential
from .comparison import compare_schemes, compare_runs
from .inventory import INVENTORY_HEADER, rows as inventory_rows, scan, summarize
from .master import (
    INPUT_HEADER,
    build_master,
    input_rows,
    sem_table,
    write_master,
)
from .nac_policy import NacPolicy
from .io.hefei import read_shprop
from .io.xdatcar import read_xdatcar
from .kinetics import (
    ASYMPTOTIC_UNDERSTATEMENT_FACTOR,
    RATE_HEADER,
    SINK_HEADER,
    bootstrap_rates,
    fit_master_equation,
    parse_edges,
    rate_rows,
    sink_rows,
    sink_sweep,
)
from .populations import (
    StateMap,
    group_series,
    load_population_set,
    per_file_group_populations,
    survival,
    trapezoid,
)
from .plotting import (
    plot_kinetics,
    plot_populations,
    plot_sink_sweep,
    plot_spectra,
    plot_vacf,
)
from .provenance import environment, fingerprint, launcher_manifests
from .report import prepare_output, write_csv, write_json
from .spectra import (
    DEFAULT_BANDS,
    PEAK_HEADER,
    SYSTEM_HEADER,
    band_rows,
    compare,
    load_spectrum,
    peak_rows,
    system_rows,
)
from .units import NAC_UNITS
from .vacf import CONVENTIONS, WINDOWS, trajectory_spectrum


def _write_report(out, payload, args):
    paths = [item["path"] for item in payload.get("inputs", [])]
    if getattr(args, "config", None):
        paths.append(args.config)
        payload["config_input"] = fingerprint([args.config])[0]
    imported = launcher_manifests(paths, getattr(args, "launcher_manifest", []))
    previous = payload.get("launcher_manifests", [])
    payload["launcher_manifests"] = list({record["input"]["path"]: record for record in previous + imported}.values())
    payload["options"] = {key: value for key, value in vars(args).items() if key != "func"}
    write_json(out / "report.json", payload)


def _expand(patterns: Sequence[str]) -> List[Path]:
    """Expand glob patterns ourselves, so quoted patterns work on any shell."""
    found: List[Path] = []
    for pattern in patterns:
        matches = sorted(globlib.glob(pattern))
        if not matches:
            candidate = Path(pattern)
            if candidate.exists():
                found.append(candidate)
                continue
            raise SystemExit(f"error: no file matches {pattern!r}")
        found.extend(Path(m) for m in matches)
    unique: List[Path] = []
    seen = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _parse_range(text: str, name: str) -> Tuple[float, float]:
    try:
        low, high = text.split(":")
        values = (float(low), float(high))
    except ValueError as exc:
        raise SystemExit(f"error: {name} must be LOW:HIGH, got {text!r}") from exc
    if values[1] <= values[0]:
        raise SystemExit(f"error: {name} must increase, got {text!r}")
    return values


def _parse_bands(text: Optional[str]) -> Tuple[Tuple[float, float], ...]:
    if not text:
        return DEFAULT_BANDS
    return tuple(_parse_range(part, "--bands") for part in text.split(","))


def _parse_labels(items: Sequence[str]) -> Dict[str, str]:
    labels = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"error: --label expects FILE=NAME, got {item!r}")
        key, _, value = item.partition("=")
        labels[key] = value
    return labels


# --------------------------------------------------------------------------
# inventory


def cmd_inventory(args: argparse.Namespace) -> int:
    entries = scan(args.root, max_depth=args.max_depth)
    out = prepare_output(args.out, overwrite=args.overwrite)
    payload = {
        "command": "inventory",
        "environment": environment(),
        "root": str(Path(args.root).resolve()),
        "max_depth": args.max_depth,
        "summary": summarize(entries),
        "runs": [entry.as_dict() for entry in entries],
    }
    _write_report(out, payload, args)
    write_csv(out / "inventory.csv", INVENTORY_HEADER, inventory_rows(entries))
    summary = payload["summary"]
    print(f"{summary['n_run_directories']} run directories under {args.root}")
    print(
        f"{summary['n_failed_legacy_fits']} of {summary['n_with_legacy_fit']} legacy "
        f"fits fall below R^2 = {summary['failed_legacy_fit_threshold_r2']}"
    )
    for failed in summary["failed_legacy_fits"]:
        print(f"  {failed['name']}: tau = {failed['tau_ns']} ns, R^2 = {failed['r_squared']}")
    print(f"written to {out}")
    return 0


# --------------------------------------------------------------------------
# audit


def cmd_audit(args: argparse.Namespace) -> int:
    policy = NacPolicy.from_json(args.nac_policy) if args.nac_policy else None
    result = audit_run(
        args.directory,
        nac_unit=args.nac_unit,
        dt_fs=args.dt_fs,
        threshold_mev=args.threshold_mev,
        max_pairs=args.max_pairs,
        policy=policy,
        warn_fraction=args.warn_fraction,
    )
    out = prepare_output(args.out, overwrite=args.overwrite)
    inputs = [
        Path(args.directory) / name
        for name in ("inp", "EIGTXT", "NATXT", "INICON", "DEPHTIME")
        if (Path(args.directory) / name).is_file()
    ]
    if args.nac_policy:
        inputs.append(Path(args.nac_policy))
    payload = {
        "command": "audit",
        "environment": environment(),
        "inputs": fingerprint(inputs),
        "audit": result.as_dict(),
    }
    _write_report(out, payload, args)
    write_csv(out / "pairs.csv", PAIR_HEADER, [pair.as_row() for pair in result.pairs])
    print(
        f"{result.directory.name}: {result.nframes} frames, {result.nstates} states, "
        f"dt = {result.dt_fs} fs (from {result.dt_source})"
    )
    for check in result.checks:
        if check["status"] != "ok":
            print(f"  [{check['status']}] {check['check']}: {check['detail']}")
    couplings = result.couplings
    limit = couplings["timestep_limit"]
    print(
        f"  mean |NAC| off-diagonal = {couplings['mean_abs_offdiagonal']:.4g} meV, "
        f"max = {couplings['max_abs_offdiagonal']:.4g} meV "
        f"(input unit declared as {result.nac_unit})"
    )
    print(
        f"  hbar/dt = {limit['hbar_over_dt_eV']:.4g} eV at dt = {limit['dt_fs']:g} fs; "
        f"the largest |NAC| is {limit['global_max_over_hbar_dt']:.4g} x that scale"
    )
    ceiling = couplings["engineered_ceiling"]
    if ceiling["detected"]:
        print(
            f"  {ceiling['samples_at_ceiling']} samples "
            f"({ceiling['fraction_at_ceiling']:.2%}) sit at exactly "
            f"{ceiling['ceiling_eV']:.4g} eV: {ceiling['interpretation']}"
        )
    print(f"written to {out}")
    return 0


# --------------------------------------------------------------------------
# populations


def cmd_populations(args: argparse.Namespace) -> int:
    state_map = StateMap.from_json(args.config)
    paths = _expand(args.files)
    population = load_population_set(paths, state_map)
    series = group_series(population, state_map)
    survival_series = survival(population, state_map)

    fit = None
    fit_error = None
    fit_uncertainty = None
    if args.bootstrap and not args.fit_group:
        raise ValueError("--bootstrap requires --fit-group")
    if args.bootstrap and args.bootstrap < 20:
        raise ValueError("--bootstrap needs at least 20 resamples")
    if args.fit_group:
        target = next((s for s in series if s.name == args.fit_group), None)
        if target is None:
            raise SystemExit(
                f"error: --fit-group {args.fit_group!r} is not a declared group"
            )
        try:
            fit = fit_single_exponential(
                population.time_ns,
                target.values,
                t_start=args.fit_start_ns,
                t_end=args.fit_end_ns,
                group=target.name,
                time_unit="ns",
            )
        except FitError as exc:
            fit_error = str(exc)
        if fit is not None and args.bootstrap:
            per_file = per_file_group_populations(population, state_map)
            index = list(state_map.groups).index(target.name)
            fit_uncertainty = bootstrap_single_exponential(
                population.time_ns, per_file[:, :, index], n_resamples=args.bootstrap,
                seed=args.bootstrap_seed, t_start=args.fit_start_ns, t_end=args.fit_end_ns,
                group=target.name, time_unit="ns")

    out = prepare_output(args.out, overwrite=args.overwrite)

    header = ["time_ns"]
    columns: List[np.ndarray] = [population.time_ns]
    for group in series:
        header.append(f"{group.name}")
        columns.append(group.values)
        if group.sem is not None:
            header.append(f"{group.name}_sem")
            columns.append(group.sem)
    header.append("total_mapped_population")
    columns.append(population.total_population)
    if survival_series is not None:
        header.append("survival")
        columns.append(survival_series)
    table = np.column_stack(columns)
    write_csv(out / "populations.csv", header, table.tolist())

    if fit is not None:
        write_csv(
            out / "fit.csv",
            ["quantity", "value", "unit"],
            [
                ["group", fit.group, ""],
                ["tau", fit.tau, fit.tau_unit],
                ["amplitude", fit.amplitude, "population"],
                ["window_start", fit.window[0], "ns"],
                ["window_end", fit.window[1], "ns"],
                ["n_points", fit.n_points, ""],
                ["r_squared", fit.r_squared, ""],
                ["rms_residual", fit.rms_residual, "population"],
                ["max_abs_residual", fit.max_abs_residual, "population"],
                ["decay_fraction_in_window", fit.decay_fraction_in_window, ""],
                ["windows_per_tau", fit.windows_per_tau, ""],
            ],
        )

    figures = plot_populations(
        population.time_ns,
        series,
        out / "populations",
        title=state_map.name,
        survival=survival_series,
        fit=fit,
    )

    payload = {
        "command": "populations",
        "environment": environment(),
        "inputs": fingerprint(paths),
        "state_map": {
            "name": state_map.name,
            "time_column": state_map.time_column,
            "time_unit": state_map.time_unit,
            "population_columns": state_map.population_columns,
            "groups": state_map.groups,
            "complete_population": state_map.complete_population,
            "recombined_group": state_map.recombined_group,
            "ungrouped_columns": state_map.ungrouped_columns(),
            "notes": state_map.notes,
        },
        "averaging": {
            "n_files": population.n_files,
            "weighting": "equal weight per file",
            "sem": (
                "between-file standard error; not an estimate of uncertainty from "
                "independent MD sampling and can be too small for correlated "
                "initial conditions"
                if population.sem is not None
                else "omitted: a single file gives no between-file spread"
            ),
            "time_grid": "identical across files; no interpolation or truncation",
            "n_time_points": int(population.time_ns.size),
            "time_span_ns": [float(population.time_ns[0]), float(population.time_ns[-1])],
        },
        "conservation": {
            **population.conservation,
            "claim": (
                "the declared groups exhaust the normalized population"
                if state_map.complete_population
                else "partial map; the mapped columns need not sum to one"
            ),
        },
        "groups": [group.summary(population.time_ns) for group in series],
        "survival": (
            {
                "definition": f"1 - P({state_map.recombined_group})",
                "initial": float(survival_series[0]),
                "final": float(survival_series[-1]),
                "window_integral_ns": trapezoid(survival_series, population.time_ns),
            }
            if survival_series is not None
            else None
        ),
        "fit": fit.as_dict() if fit is not None else None,
        "fit_error": fit_error,
        "fit_uncertainty": fit_uncertainty,
        "figures": [str(path) for path in figures],
        "interpretation_limits": [
            "population leaving a group is not by itself recombination or "
            "extraction; its destination must be inspected",
            "net accumulation is not directional flux or collected charge",
            "forward and backward rates, first-passage yields and extraction "
            "efficiencies are not inferred from averaged populations",
        ],
    }
    _write_report(out, payload, args)

    print(f"{population.n_files} file(s), {population.time_ns.size} time points")
    print(
        "  mapped population sums to "
        f"[{population.conservation['min']:.6f}, {population.conservation['max']:.6f}]"
    )
    for group in series:
        info = group.summary(population.time_ns)
        print(
            f"  {group.name}: {info['initial']:.4f} -> {info['final']:.4f} "
            f"(net {info['net_change']:+.4f})"
        )
    if fit is not None:
        print(f"  fit {fit.group}: tau = {fit.tau:.6g} ns, R^2 = {fit.r_squared:.4f}")
        for warning in fit.warnings:
            print(f"    warning: {warning}")
    elif fit_error:
        print(f"  fit not performed: {fit_error}")
    print(f"written to {out}")
    return 0


# --------------------------------------------------------------------------
# average-shprop


def cmd_average_shprop(args: argparse.Namespace) -> int:
    state_map = StateMap.from_json(args.config)
    paths = _expand(args.files)
    master = build_master(
        paths, state_map, average_extra_columns=args.average_extra_columns
    )

    out = prepare_output(args.out, overwrite=args.overwrite)
    written = write_master(master, out / "SHPROP.master")

    # Read the written file back through the ordinary reader and re-check it,
    # so the artifact that leaves here is the artifact that was validated.
    readback = read_shprop(written)
    if readback.shape != master.table.shape or not np.allclose(
        readback, master.table, rtol=0, atol=0
    ):
        raise RuntimeError(
            f"{written} does not read back as the table that was computed"
        )

    table = sem_table(master)
    write_csv(out / "population_sem.csv", table["header"], table["rows"])
    write_csv(out / "input_files.csv", INPUT_HEADER, input_rows(master))

    payload = {
        "command": "average-shprop",
        "environment": environment(),
        "inputs": master.rows,
        "master": {
            **master.as_dict(),
            "output": str(written),
            "number_format": "%.17g, which round-trips an IEEE double exactly",
            "readback_verified": True,
            "output_conservation_after_readback": float(
                np.max(
                    np.abs(readback[:, master.population_columns].sum(axis=1) - 1.0)
                )
            )
            if state_map.complete_population
            else None,
        },
    }
    _write_report(out, payload, args)

    print(
        f"{master.n_files} source file(s), {master.table.shape[0]} rows, "
        f"{master.table.shape[1]} columns -> {written}"
    )
    print(
        "  averaged population columns: "
        + ", ".join(str(c) for c in master.population_columns)
    )
    for column in master.extra_columns:
        print(f"  column {column}: {master.extra_column_policy[column]}")
    if state_map.complete_population:
        print(
            "  population conservation in the master deviates from one by at most "
            f"{master.master_conservation['max_abs_deviation_from_one']:.3g} "
            f"(checked independently in all {master.n_files} source files)"
        )
    if master.population_sem is None:
        print("  no between-file SEM: one file gives no spread")
    print(f"written to {out}")
    return 0


# --------------------------------------------------------------------------
# kinetics


def _parse_sweep(text: str) -> List[float]:
    """LOW:HIGH:COUNT on a log grid, or a comma-separated list of rates."""
    if "," in text or ":" not in text:
        try:
            values = [float(part) for part in text.split(",") if part.strip()]
        except ValueError as exc:
            raise SystemExit(f"error: --sink-rates {text!r} is not a list of numbers") from exc
        if not values:
            raise SystemExit("error: --sink-rates listed no rates")
        if any(not np.isfinite(v) or v < 0 for v in values):
            raise SystemExit(
                f"error: --sink-rates {text!r} contains a negative or non-finite rate"
            )
        return values
    pieces = text.split(":")
    if len(pieces) != 3:
        raise SystemExit(
            f"error: --sink-rates must be LOW:HIGH:COUNT or a comma-separated list, got {text!r}"
        )
    try:
        low, high, count = float(pieces[0]), float(pieces[1]), int(pieces[2])
    except ValueError as exc:
        raise SystemExit(f"error: --sink-rates {text!r} is malformed") from exc
    if low <= 0 or high <= low or count < 2:
        raise SystemExit(
            "error: --sink-rates needs 0 < LOW < HIGH and COUNT >= 2 (the grid is logarithmic)"
        )
    return [float(v) for v in np.geomspace(low, high, count)]


def cmd_kinetics(args: argparse.Namespace) -> int:
    state_map = StateMap.from_json(args.config)
    if not state_map.complete_population:
        raise SystemExit(
            "error: kinetics needs complete_population: true in the configuration. "
            "A master equation over a subset of states is not a closed system, and "
            "rates fitted to one are not meaningful."
        )
    paths = _expand(args.files)
    population = load_population_set(paths, state_map)
    series = group_series(population, state_map)
    names = [s.name for s in series]

    observed_full = np.column_stack([s.values for s in series])
    time_full = population.time_ns
    lo = time_full[0] if args.fit_start_ns is None else args.fit_start_ns
    hi = time_full[-1] if args.fit_end_ns is None else args.fit_end_ns
    mask = (time_full >= lo) & (time_full <= hi)
    if np.count_nonzero(mask) < 3:
        raise SystemExit(f"error: the window [{lo}, {hi}] ns holds too few samples")
    time_ns = time_full[mask]
    observed = observed_full[mask]

    edges = parse_edges(args.scheme, names)

    weights = None
    weight_notes: List[str] = []
    if args.weight_by_sem:
        if population.sem is None:
            raise SystemExit(
                "error: --weight-by-sem needs more than one input file"
            )
        sem = np.column_stack(
            [s.sem if s.sem is not None else np.ones_like(s.values) for s in series]
        )[mask]
        positive = sem[sem > 0]
        if positive.size == 0:
            raise SystemExit(
                "error: every between-file SEM is zero, so --weight-by-sem has no "
                "information to weight with; the input files are identical"
            )
        # Floor on the MEDIAN positive spread, not the maximum. A group that is
        # never populated has SEM exactly zero in every file, and a floor tied
        # to the maximum would hand it the largest weight in the problem.
        floor = float(np.median(positive))
        weights = 1.0 / np.maximum(sem, floor)
        zero_sem_groups = [
            name for index, name in enumerate(names) if np.all(sem[:, index] <= 0)
        ]
        if zero_sem_groups:
            weight_notes.append(
                "these groups have zero spread between files and are weighted at "
                "the median level rather than infinitely: " + ", ".join(zero_sem_groups)
            )

    fit = fit_master_equation(time_ns, observed, names, edges, weights=weights)

    bootstrap = {}
    bootstrap_diagnostics = {}
    bootstrap_note = "not requested"
    if args.bootstrap:
        per_file = per_file_group_populations(population, state_map)
        if per_file is None or per_file.shape[0] < 2:
            bootstrap_note = "skipped: bootstrapping whole files needs at least two files"
        else:
            bootstrap, bootstrap_diagnostics = bootstrap_rates(
                time_ns, per_file[:, mask, :], names, edges,
                n_resamples=args.bootstrap, seed=args.bootstrap_seed,
                weights=weights,
            )
            per_rate = bootstrap_diagnostics.get("per_rate", {})
            for estimate in fit.rates:
                if estimate.name in bootstrap:
                    estimate.bootstrap_ci_per_ns = bootstrap[estimate.name]
                record = per_rate.get(estimate.name, {})
                estimate.bootstrap_successes = record.get("bootstrap_successes")
                estimate.bootstrap_identified = record.get("bootstrap_identified")
                estimate.bootstrap_identified_fraction = record.get(
                    "bootstrap_identified_fraction"
                )
                estimate.bootstrap_ci_status = record.get("bootstrap_ci_status")
            converged = bootstrap_diagnostics.get("converged_resamples", 0)
            bootstrap_note = (
                f"{converged} of {args.bootstrap} resamples of the "
                f"{per_file.shape[0]} input files converged, 2.5-97.5 percentile, "
                + ("weighted the same way as the point estimate" if weights is not None
                   else "unweighted, matching the point estimate")
                + "; this is the spread between the files supplied, not an "
                "ensemble error bar"
            )
            if bootstrap_diagnostics.get("note"):
                bootstrap_note += ". " + bootstrap_diagnostics["note"]
            for name in bootstrap_diagnostics.get(
                "suppressed_for_low_identified_fraction", []
            ):
                record = bootstrap_diagnostics["per_rate"][name]
                print(
                    f"  {name}: bootstrap interval suppressed because the "
                    "transition was identifiable in only "
                    f"{record['bootstrap_identified_fraction']:.0%} of successful "
                    "resamples"
                )

    sweep = []
    if args.sink_group:
        if args.sink_group not in names:
            raise SystemExit(
                f"error: --sink-group {args.sink_group!r} is not a declared group {names}"
            )
        sweep = sink_sweep(
            fit,
            args.sink_group,
            _parse_sweep(args.sink_rates),
            recombined_group=state_map.recombined_group,
        )

    out = prepare_output(args.out, overwrite=args.overwrite)

    write_csv(out / "rates.csv", RATE_HEADER, rate_rows(fit))
    write_csv(
        out / "kinetics_curves.csv",
        ["time_ns"]
        + [f"{n}_observed" for n in names]
        + [f"{n}_model" for n in names],
        np.column_stack([time_ns, observed, fit.model]).tolist(),
    )
    figures = plot_kinetics(fit, out / "kinetics", title=state_map.name)
    if sweep:
        write_csv(out / "sink_sweep.csv", SINK_HEADER, sink_rows(sweep))
        figures += plot_sink_sweep(sweep, out / "sink_sweep")

    payload = {
        "command": "kinetics",
        "environment": environment(),
        "inputs": fingerprint(paths),
        "state_map": {
            "name": state_map.name,
            "groups": state_map.groups,
            "recombined_group": state_map.recombined_group,
        },
        "scheme": args.scheme,
        "weighting": "1/SEM per point" if weights is not None else "unweighted",
        "weighting_notes": weight_notes,
        "conservation": population.conservation,
        "fit": fit.as_dict(),
        "bootstrap": {
            "intervals_per_ns": bootstrap,
            "diagnostics": bootstrap_diagnostics,
            "note": bootstrap_note,
        },
        "sink_sweep": (
            {
                "sink_group": args.sink_group,
                "points": [p.as_dict() for p in sweep],
                "note": (
                    "the escape channel is propagated as part of the augmented "
                    "system, so escaped population leaves the dynamics and cannot "
                    "return or recombine. k_escape is supplied by the user; the "
                    "interface calculation does not determine it, and the transfer "
                    "rates were fitted to data containing no extraction"
                ),
            }
            if sweep
            else None
        ),
        "figures": [str(path) for path in figures],
        "interpretation_limits": [
            "a Markovian rate matrix is assumed, not demonstrated; a good fit does "
            "not establish that the dynamics are Markovian",
            "rates marked identified: false were not determined by these population "
            "curves and must not be quoted; use the eigenvalue timescales instead",
            "the sink sweep is a counterfactual over an assumed escape rate, not a "
            "measured extraction efficiency",
        ],
    }
    _write_report(out, payload, args)

    print(f"{population.n_files} file(s), {fit.n_points} points in "
          f"[{fit.window[0]:.4g}, {fit.window[1]:.4g}] ns, scheme {args.scheme!r}")
    print(f"  R^2 over all groups = {fit.r_squared_total:.5f}, "
          f"Jacobian condition number = {fit.condition_number:.3g}")
    for estimate in fit.rates:
        flag = "" if estimate.identified else "   [UNIDENTIFIED]"
        error = (
            f" +- {estimate.stderr_per_ns:.4g}" if estimate.stderr_per_ns is not None else ""
        )
        print(
            f"  {estimate.name:<22s} {estimate.rate_per_ns:10.4g}{error} /ns"
            f"  (tau = {estimate.lifetime_ns:.4g} ns){flag}"
        )
    print("  eigenvalue timescales (ns): "
          + ", ".join(f"{t:.4g}" for t in fit.eigen_timescales_ns))
    if not bootstrap and any(e.stderr_per_ns is not None for e in fit.rates):
        print(
            f"  note: those +- values are a linearized LOWER bound (measured at up "
            f"to {ASYMPTOTIC_UNDERSTATEMENT_FACTOR:g}x too small because P(0) is read "
            "from a noisy sample). Pass --bootstrap for an interval that varies it."
        )
    for warning in fit.warnings:
        print(f"  warning: {warning}")
    if sweep:
        print(f"  sink on {args.sink_group}: collected at the end of the window ranges "
              f"{sweep[0].collected_final:.4f} -> {sweep[-1].collected_final:.4f}")
    print(f"written to {out}")
    return 0


# --------------------------------------------------------------------------
# vacf-spectra


def cmd_vacf_spectra(args: argparse.Namespace) -> int:
    paths = _expand(args.files)
    labels = _parse_labels(args.label)
    spectra = []
    for path in paths:
        label = labels.get(str(path)) or labels.get(path.name)
        spectra.append(load_spectrum(path, label=label))

    bands = _parse_bands(args.bands)
    analysis_range = _parse_range(args.range, "--range")
    payload_compare = compare(
        spectra,
        reference_label=args.reference,
        bands=bands,
        analysis_range=analysis_range,
        peak_rel_height=args.peak_rel_height,
    )

    out = prepare_output(args.out, overwrite=args.overwrite)
    write_csv(out / "systems.csv", SYSTEM_HEADER, system_rows(payload_compare))
    header, rows = band_rows(payload_compare)
    write_csv(out / "bands.csv", header, rows)
    write_csv(out / "peaks.csv", PEAK_HEADER, peak_rows(payload_compare))
    figures = plot_spectra(
        spectra,
        out / "spectra",
        xlim=_parse_range(args.xlim, "--xlim"),
        title=args.title,
        normalize=args.normalize_plot,
    )

    payload = {
        "command": "vacf-spectra",
        "environment": environment(),
        "inputs": fingerprint(paths),
        "bands_cm1": [list(band) for band in bands],
        "analysis_range_cm1": list(analysis_range),
        **payload_compare,
        "figures": [str(path) for path in figures],
        "interpretation_limits": [
            "band integrals describe the curve; they do not assign modes to "
            "specific motions",
            "raw integrals are comparable between systems only when the spectra "
            "share convention, normalization, trajectory length and atom count",
            "a spectrum read from a two-column file carries no record of its "
            "transform convention, window or smoothing",
        ],
    }
    _write_report(out, payload, args)

    print(f"{len(spectra)} spectra, shared grid: {payload_compare['shared_grid']}")
    for system in payload_compare["systems"]:
        low_band = next(
            (b for b in system["bands"] if b["low_cm1"] == 0.0), system["bands"][0]
        )
        print(
            f"  {system['label']}: centroid = {system['centroid_cm1']:.2f} cm^-1, "
            f"{low_band['low_cm1']:.0f}-{low_band['high_cm1']:.0f} cm^-1 holds "
            f"{100 * low_band['fraction_of_range']:.1f}% of the range integral, "
            f"{len(system['peaks'])} peaks"
        )
    print(f"written to {out}")
    return 0


# --------------------------------------------------------------------------
# vacf-trajectory


def cmd_vacf_trajectory(args: argparse.Namespace) -> int:
    paths = _expand(args.xdatcar)
    labels = _parse_labels(args.label)
    out = prepare_output(args.out, overwrite=args.overwrite)

    spectra = []
    systems = []
    for path in paths:
        label = labels.get(str(path)) or labels.get(path.name) or path.parent.name
        trajectory = read_xdatcar(path)
        vacf, spectrum = trajectory_spectrum(
            trajectory,
            dt_fs=args.dt_fs,
            max_lag=args.max_lag,
            mass_weight=args.mass_weight,
            unwrap=not args.no_unwrap,
            segment_length=args.segment_length,
            convention=args.convention,
            window=args.window,
            smooth_sigma=args.smooth_sigma,
            atom_reduction=args.atom_reduction,
            remove_com=not args.keep_com,
            label=label,
        )
        spectra.append(spectrum)

        write_csv(
            out / f"vacf_{label}.csv",
            ["lag_fs", "vacf", "vacf_normalized", "n_origins"],
            np.column_stack(
                [vacf.lags_fs, vacf.values, vacf.normalized, vacf.n_origins]
            ).tolist(),
        )
        np.savetxt(
            out / f"spectral_density_{label}.txt",
            np.column_stack([spectrum.frequency_cm1, spectrum.intensity]),
            header="Frequency(cm^-1)\tSpectral_Density",
            comments="",
        )
        plot_vacf(vacf, out / f"vacf_{label}", title=f"VACF: {label}")
        systems.append(
            {
                "label": label,
                "xdatcar": str(path),
                "frames": trajectory.nframes,
                "atoms": trajectory.natoms,
                "species": trajectory.species,
                "counts": trajectory.counts,
                "coordinates": "direct" if trajectory.direct else "cartesian",
                "variable_cell": trajectory.variable_cell,
                "vacf": vacf.diagnostics(),
                "spectrum": spectrum.meta,
            }
        )

    figures = plot_spectra(
        spectra,
        out / "spectra",
        xlim=_parse_range(args.xlim, "--xlim"),
        title=args.title,
        normalize=args.normalize_plot,
    )

    payload = {
        "command": "vacf-trajectory",
        "environment": environment(),
        "inputs": fingerprint(paths),
        "settings": {
            "dt_fs": args.dt_fs,
            "max_lag": args.max_lag,
            "segment_length": args.segment_length,
            "mass_weight": args.mass_weight,
            "minimum_image_unwrap": not args.no_unwrap,
            "convention": args.convention,
            "window": args.window,
            "smooth_sigma": args.smooth_sigma,
            "atom_reduction": args.atom_reduction,
            "center_of_mass_motion_removed": not args.keep_com,
        },
        "settings_note": (
            "velocities are Cartesian: fractional displacements are folded into "
            "the minimum image and multiplied by the cell before differencing. "
            "--no-unwrap reproduces a raw fractional difference and is only for "
            "comparison with older output."
        ),
        "systems": systems,
        "figures": [str(path) for path in figures],
    }
    _write_report(out, payload, args)

    for system in systems:
        diagnostics = system["vacf"]
        print(
            f"{system['label']}: {system['frames']} frames, {system['atoms']} atoms, "
            f"VACF(0) = {diagnostics['vacf_zero_lag']:.4g} "
            f"{diagnostics['vacf_zero_lag_unit']}, correlation time = "
            f"{diagnostics['correlation_time_fs']:.2f} fs"
        )
    print(f"written to {out}")
    return 0


# --------------------------------------------------------------------------


def cmd_compare_schemes(args):
    mapping = StateMap.from_json(args.config)
    if not mapping.complete_population or mapping.ungrouped_columns():
        raise ValueError("Scheme comparison needs a complete exhaustive group map")
    paths = _expand(args.files)
    population = load_population_set(paths, mapping)
    groups = group_series(population, mapping)
    time = population.time_ns
    mask = np.ones(len(time), dtype=bool)
    if args.fit_start_ns is not None:
        mask &= time >= args.fit_start_ns
    if args.fit_end_ns is not None:
        mask &= time <= args.fit_end_ns
    observed = np.column_stack([g.values for g in groups])[mask]
    schemes = json.loads(Path(args.schemes).read_text())
    payload, fits = compare_schemes(time[mask], observed, [g.name for g in groups], schemes, criterion=args.criterion)
    out = prepare_output(args.out, overwrite=args.overwrite)
    payload.update(command="compare-schemes", environment=environment(),
                   inputs=fingerprint(paths + [Path(args.schemes)]),
                   fits={name: fit.as_dict() for name, fit in fits.items()})
    header = ["candidate", "scheme", "status", "rank", "score", "delta", "aic", "aicc", "bic", "unidentified_rates", "rank_deficient", "numerically_exact", "error"]
    write_csv(out / "schemes.csv", header, [[row.get(key) for key in header] for row in payload["candidates"]])
    _write_report(out, payload, args)
    print(f"Lowest descriptive score: {payload['lowest_score_candidate']}; see model assumptions in report.json")
    return 0


def cmd_compare_runs(args):
    payload, time, curves = compare_runs(args.manifest, start_ns=args.start_ns, end_ns=args.end_ns)
    out = prepare_output(args.out, overwrite=args.overwrite)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = payload["shared_groups"]
    fig, axes = plt.subplots(len(names), 1, figsize=(8, 3 * len(names)), squeeze=False, constrained_layout=True)
    rows = []
    for i, name in enumerate(names):
        ax = axes[i, 0]
        for label, groups in curves.items():
            ax.plot(time, groups[name], label=label)
            rows.extend([label, name, float(t), float(p)] for t, p in zip(time, groups[name]))
        ax.set(xlabel="Time (ns)", ylabel=f"P({name})", ylim=(-.03, 1.03))
        ax.legend()
    for suffix in ["png", "pdf"]:
        fig.savefig(out / f"comparison.{suffix}", dpi=200)
    plt.close(fig)
    write_csv(out / "curves.csv", ["run", "group", "time_ns", "population"], rows)
    header = ["run", "reference", "group", "initial_difference", "final_difference", "max_abs_difference", "difference_integral_ns"]
    write_csv(out / "differences.csv", header, [[row[key] for key in header] for row in payload["differences"]])
    payload.update(command="compare-runs", environment=environment())
    _write_report(out, payload, args)
    print(f"Compared {len(curves)} runs on {len(time)} exact shared times; written to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="namd-analysis",
        description=(
            "Analyze saved CA-NAC / Hefei-NAMD output and VACF phonon spectra. "
            "Inputs are read, never modified."
        ),
    )
    parser.add_argument("--version", action="version", version=f"namd-analysis {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_output(target, default_required=True):
        target.add_argument("--out", required=default_required, help="new output directory")
        target.add_argument("--launcher-manifest", action="append", default=[],
                            help="additional upstream manifest JSON; repeat as needed")
        target.add_argument(
            "--overwrite",
            action="store_true",
            help="allow writing into a directory that already holds results",
        )

    inventory = sub.add_parser("inventory", help="list run directories and legacy fits")
    inventory.add_argument("root", help="directory holding unpacked campaigns")
    inventory.add_argument("--max-depth", type=int, default=4)
    add_output(inventory)
    inventory.set_defaults(func=cmd_inventory)

    audit = sub.add_parser("audit", help="check EIGTXT/NATXT and summarize couplings")
    audit.add_argument("directory", help="a single run directory")
    audit.add_argument(
        "--nac-unit",
        required=True,
        choices=NAC_UNITS,
        help="unit of NATXT, taken from the producing code and not from the magnitudes",
    )
    audit.add_argument("--dt-fs", type=float, default=None, help="overrides POTIM from inp")
    audit.add_argument("--threshold-mev", type=float, default=0.5)
    audit.add_argument("--max-pairs", type=int, default=None)
    audit.add_argument(
        "--nac-policy",
        default=None,
        help="JSON declaring the upstream NAC handling policy for this dataset "
        "(warning_threshold_eV, numerical_limit, reject_above_eV, "
        "action_above_limit); nothing is assumed without it",
    )
    audit.add_argument(
        "--warn-fraction",
        type=float,
        default=DEFAULT_WARNING_FRACTION,
        help="fraction of samples inside the 0.8 x hbar/dt region treated as "
        "unremarkable (default 0: any sample there is reported)",
    )
    add_output(audit)
    audit.set_defaults(func=cmd_audit)

    populations = sub.add_parser("populations", help="average SHPROP files and analyze groups")
    populations.add_argument("--files", nargs="+", required=True, help="paths or glob patterns")
    populations.add_argument("--config", required=True, help="state map JSON")
    populations.add_argument("--fit-group", default=None)
    populations.add_argument("--fit-start-ns", type=float, default=None)
    populations.add_argument("--fit-end-ns", type=float, default=None)
    populations.add_argument("--bootstrap", type=int, default=0, help="whole-file decay bootstrap (>=20)")
    populations.add_argument("--bootstrap-seed", type=int, default=0)
    add_output(populations)
    populations.set_defaults(func=cmd_populations)

    average = sub.add_parser(
        "average-shprop",
        help="build a canonical master SHPROP from the original SHPROP files",
    )
    average.add_argument("--files", nargs="+", required=True, help="paths or glob patterns")
    average.add_argument("--config", required=True, help="state map JSON")
    average.add_argument(
        "--average-extra-columns",
        action="store_true",
        help="average columns that are neither time nor population when the "
        "source files disagree on them; without this such a disagreement is an "
        "error rather than a silent copy of the first file",
    )
    add_output(average)
    average.set_defaults(func=cmd_average_shprop)

    kinetics = sub.add_parser(
        "kinetics", help="fit a multistate rate model to group populations"
    )
    kinetics.add_argument("--files", nargs="+", required=True, help="paths or glob patterns")
    kinetics.add_argument("--config", required=True, help="state map JSON (must be complete)")
    kinetics.add_argument(
        "--scheme",
        required=True,
        help='"dense", "sequential", "reversible", or an explicit comma-separated '
        'list such as "CBM->BCF,BCF->CBM,BCF->PCBM,PCBM->VBM"',
    )
    kinetics.add_argument("--fit-start-ns", type=float, default=None)
    kinetics.add_argument("--fit-end-ns", type=float, default=None)
    kinetics.add_argument(
        "--weight-by-sem",
        action="store_true",
        help="weight residuals by 1/SEM; needs more than one input file",
    )
    kinetics.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        metavar="N",
        help="resample whole input files N times for a percentile interval",
    )
    kinetics.add_argument("--bootstrap-seed", type=int, default=0)
    kinetics.add_argument(
        "--sink-group",
        default=None,
        help="add an absorbing extraction channel draining this group",
    )
    kinetics.add_argument(
        "--sink-rates",
        default="0.01:1000:25",
        help="LOW:HIGH:COUNT on a log grid, or a comma-separated list, in ns^-1",
    )
    add_output(kinetics)
    kinetics.set_defaults(func=cmd_kinetics)

    scheme_compare = sub.add_parser("compare-schemes", help="score declared kinetic graphs on common observations")
    scheme_compare.add_argument("--files", nargs="+", required=True)
    scheme_compare.add_argument("--config", required=True)
    scheme_compare.add_argument("--schemes", required=True, help="JSON object mapping candidate names to schemes")
    scheme_compare.add_argument("--criterion", choices=["aic", "aicc", "bic"], default="aicc")
    scheme_compare.add_argument("--fit-start-ns", type=float)
    scheme_compare.add_argument("--fit-end-ns", type=float)
    add_output(scheme_compare)
    scheme_compare.set_defaults(func=cmd_compare_schemes)

    run_compare = sub.add_parser("compare-runs", help="compare separately averaged systems or initial states")
    run_compare.add_argument("--manifest", required=True, help="JSON runs with labels, configs, and files")
    run_compare.add_argument("--start-ns", type=float)
    run_compare.add_argument("--end-ns", type=float)
    add_output(run_compare)
    run_compare.set_defaults(func=cmd_compare_runs)

    spectra = sub.add_parser(
        "vacf-spectra", help="describe and compare existing spectral density files"
    )
    spectra.add_argument("--files", nargs="+", required=True)
    spectra.add_argument("--label", nargs="*", default=[], help="FILE=NAME overrides")
    spectra.add_argument("--bands", default=None, help='e.g. "0:50,50:100,100:200"')
    spectra.add_argument("--range", default="0:800", help="analysis range LOW:HIGH in cm^-1")
    spectra.add_argument("--reference", default=None, help="label to difference against")
    spectra.add_argument("--peak-rel-height", type=float, default=0.1)
    spectra.add_argument("--xlim", default="0:150")
    spectra.add_argument("--title", default="Phonon spectral density")
    spectra.add_argument("--normalize-plot", action="store_true")
    add_output(spectra)
    spectra.set_defaults(func=cmd_vacf_spectra)

    trajectory = sub.add_parser(
        "vacf-trajectory", help="compute VACF and spectral density from XDATCAR"
    )
    trajectory.add_argument("--xdatcar", nargs="+", required=True)
    trajectory.add_argument("--label", nargs="*", default=[], help="FILE=NAME overrides")
    trajectory.add_argument("--dt-fs", type=float, required=True)
    trajectory.add_argument("--max-lag", type=int, default=5000)
    trajectory.add_argument("--segment-length", type=int, default=None)
    trajectory.add_argument("--mass-weight", action="store_true")
    trajectory.add_argument(
        "--no-unwrap",
        action="store_true",
        help="difference raw fractional coordinates (legacy behaviour, not physical)",
    )
    trajectory.add_argument("--convention", choices=CONVENTIONS, default="cosine")
    trajectory.add_argument("--window", choices=WINDOWS, default="hann")
    trajectory.add_argument("--smooth-sigma", type=float, default=0.0)
    trajectory.add_argument("--atom-reduction", choices=("mean", "sum"), default="mean")
    trajectory.add_argument(
        "--keep-com",
        action="store_true",
        help="keep rigid centre-of-mass translation, which adds spurious "
        "low-frequency intensity",
    )
    trajectory.add_argument("--xlim", default="0:800")
    trajectory.add_argument("--title", default="Phonon spectral density")
    trajectory.add_argument("--normalize-plot", action="store_true")
    add_output(trajectory)
    trajectory.set_defaults(func=cmd_vacf_trajectory)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
