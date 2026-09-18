"""CLI for ``namd-analysis character-ensemble``.

Two modes, because the production shape is a job array followed by a gather:

* **analyse** -- walk the SHPROP histories of ONE run, one at a time, and write
  a per-history row plus a per-run summary. Histories stream independently, so
  a job array can split them and the rows concatenate.
* **--combine** -- read several per-run summaries and emit the across-run
  comparison and the reviewer table. This is the only level at which a
  Campaign-level statement may be made.

Per-history rows are flushed as they are produced. A run of a hundred 910 MB
histories takes hours, and a crash at history 87 must not discard the first 86.
"""

from __future__ import annotations

import argparse
import csv
import glob as globlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .character import aligned_frames, resolve_namdtini
from .crossings import CrossingError, history_fragment_population, read_projection_character
from .ensemble import (
    EnsembleError,
    HistoryResult,
    compare_runs,
    ensemble_curve,
    history_header,
    reviewer_table,
    summarize_run,
)
from .io.hefei import shprop_structure
from .populations import StateMap
from .provenance import environment, fingerprint
from .report import prepare_output, write_csv, write_json


def _expand(patterns: Sequence[str]) -> List[Path]:
    found: List[Path] = []
    for pattern in patterns:
        matches = sorted(globlib.glob(str(pattern)))
        if not matches:
            candidate = Path(pattern)
            if not candidate.exists():
                raise CrossingError(f"no file matches {pattern!r}")
            matches = [str(candidate)]
        found.extend(Path(m) for m in matches)
    unique, seen = [], set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def analyse_history(
    path: Path,
    run: str,
    state_map: StateMap,
    character,
    bands: Sequence[int],
    frame_mode: str,
    cycle_length: Optional[int],
    episode: Optional[tuple],
    late_start_ns: float,
    chunk_rows: int,
    curve_stride: int,
) -> tuple:
    """One history, start to finish. Returns (HistoryResult, curves)."""
    record = shprop_structure(path)
    start, source = resolve_namdtini(record)
    frames = aligned_frames(
        {"NAMDTINI": start, "NSW": record.metadata.get("NSW")},
        record.n_rows, frame_mode, path, cycle_length=cycle_length,
    )
    split = history_fragment_population(
        path, state_map, character, bands, frames, chunk_rows=chunk_rows
    )
    P = split.total
    occ = split.occupation_redistribution
    chr_ = split.character_evolution
    time_ns = split.time_raw / state_map.to_ns
    groups = list(character.groups)

    residual = float(np.abs(np.diff(P, axis=0) - (occ + chr_)[1:]).max())

    late = time_ns >= float(late_start_ns)
    n_passes = int(np.ceil(record.n_rows / cycle_length)) if cycle_length else 1

    result = HistoryResult(
        run=run, history=path.name, namdtini=int(start),
        n_rows=int(record.n_rows), n_passes=n_passes,
        decomposition_residual=residual,
    )
    for gi, g in enumerate(groups):
        result.net_full[g] = float(P[-1, gi] - P[0, gi])
        result.occupation_redistribution[g] = float(occ[1:, gi].sum())
        result.character_evolution[g] = float(chr_[1:, gi].sum())
        if late.sum() >= 2:
            y = P[late, gi]
            result.net_late[g] = float(y[-1] - y[0])
            result.range_late[g] = float(y.max() - y.min())

    if episode and cycle_length:
        lo, hi = episode
        in_ep = (frames >= lo) & (frames <= hi)
        steps = np.flatnonzero(in_ep)
        if steps.size:
            pass_id = np.arange(record.n_rows) // cycle_length
            pids = pass_id[steps]
            npass = int(pids.max()) + 1
            for gi, g in enumerate(groups):
                o = np.bincount(pids, weights=occ[steps, gi], minlength=npass)
                c = np.bincount(pids, weights=chr_[steps, gi], minlength=npass)
                net = o + c
                # Per pass, never summed over passes: the passes are
                # re-traversals of one nuclear trajectory.
                result.episode_net_per_pass[g] = float(net.mean())
                result.episode_occupation_per_pass[g] = float(o.mean())
            result.episode_verdict = _verdict(result, groups)

    curves = {g: P[::curve_stride, gi].copy() for gi, g in enumerate(groups)}
    curves["_time_ns"] = time_ns[::curve_stride].copy()
    return result, curves


def _verdict(result: HistoryResult, groups: Sequence[str],
             donor: str = "BCF", acceptor: str = "PCBM",
             tolerance: float = 1.0e-5) -> str:
    """Name this history's episode response without claiming a mechanism."""
    if donor not in groups or acceptor not in groups:
        return "not_classified"
    a = result.episode_net_per_pass.get(acceptor, 0.0)
    d = result.episode_net_per_pass.get(donor, 0.0)
    if abs(a) <= tolerance and abs(d) <= tolerance:
        return "essentially_reversible"
    if a > tolerance and d < -tolerance:
        return "persistent_acceptor_gain"
    if d > tolerance and a < -tolerance:
        return "persistent_donor_gain"
    return "no_clear_direction"


def build_parser(prog: str = "namd-analysis character-ensemble") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Aggregate SHPROP histories at three levels: per history, per run, "
            "and across runs. Passes within a history and histories within a "
            "run are not independent samples, and the hierarchy is preserved."
        ),
    )
    parser.add_argument("--run-label", default=None, help="name of the run being analysed")
    parser.add_argument("--shprop", nargs="+", default=None, help="SHPROP histories or globs")
    parser.add_argument("--projection-character", default=None,
                        help="projection_character.csv for this run")
    parser.add_argument("--state-map", default=None, help="SHPROP state-map JSON")
    parser.add_argument("--frame-mode", choices=("dish-cyclic", "linear"), default=None)
    parser.add_argument("--cycle-length", type=int, default=None,
                        help="frames in the recycled nuclear trajectory")
    parser.add_argument("--episode-window", default=None,
                        help="FIRST:LAST frame of the crossing manifold")
    parser.add_argument("--late-window-start-ns", type=float, default=0.1,
                        help="start of the late window in ns (default 0.1 = 100 ps)")
    parser.add_argument("--shprop-chunk-rows", type=int, default=250_000)
    parser.add_argument("--curve-stride", type=int, default=1000,
                        help="thin the saved population curves by this factor")
    parser.add_argument("--combine", nargs="+", default=None,
                        help="per-run summary JSONs to compare across runs")
    parser.add_argument("--donor", default="BCF")
    parser.add_argument("--acceptor", default="PCBM")
    parser.add_argument("--out", required=True, help="new output directory")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        out = prepare_output(args.out, overwrite=args.overwrite)

        if args.combine:
            runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.combine]
            groups = sorted({
                g for r in runs
                for g in r.get("quantities", {}).get("net_full", {})
            })
            comparison = {
                "across_runs": {
                    q: compare_runs(runs, groups, quantity=q)
                    for q in ("net_full", "net_late", "episode_net_per_pass")
                },
                "reviewer_table_full_trajectory": reviewer_table(
                    runs, donor=args.donor, acceptor=args.acceptor,
                    quantity="net_full"),
                "reviewer_table_late_window": reviewer_table(
                    runs, donor=args.donor, acceptor=args.acceptor,
                    quantity="net_late"),
                "environment": environment(),
            }
            write_json(out / "across_runs.json", comparison)
            for name, table in (
                ("reviewer_table_full_trajectory.csv",
                 comparison["reviewer_table_full_trajectory"]),
                ("reviewer_table_late_window.csv",
                 comparison["reviewer_table_late_window"]),
            ):
                write_csv(
                    out / name, ["quantity", *table["runs"]],
                    [[row["row"], *[row["values"][r] for r in table["runs"]]]
                     for row in table["rows"]],
                )
            print(f"combined {len(runs)} run(s): {', '.join(r['run'] for r in runs)}")
            for row in comparison["reviewer_table_full_trajectory"]["rows"]:
                values = comparison["reviewer_table_full_trajectory"]["runs"]
                cells = "  ".join(
                    f"{row['values'][r]:>12}" if isinstance(row["values"][r], str)
                    else f"{row['values'][r]:>12.4f}" for r in values
                )
                print(f"  {row['row']:>34}: {cells}")
            print("\nruns are the reproducibility unit; read the spread, not the mean")
            print(f"written to {out}")
            return 0

        missing = [
            name for name, value in (
                ("--run-label", args.run_label), ("--shprop", args.shprop),
                ("--projection-character", args.projection_character),
                ("--state-map", args.state_map), ("--frame-mode", args.frame_mode),
            ) if not value
        ]
        if missing:
            print(f"error: analysis mode needs {', '.join(missing)}")
            return 2

        character = read_projection_character(args.projection_character)
        state_map = StateMap.from_json(args.state_map)
        bands = [int(b) for b in character.bands]
        groups = list(character.groups)
        episode = None
        if args.episode_window:
            lo, _, hi = args.episode_window.partition(":")
            episode = (int(lo), int(hi))

        paths = _expand(args.shprop)
        print(f"run {args.run_label}: {len(paths)} history/histories")

        rows_path = out / "per_history.csv"
        header = history_header(groups)
        results: List[HistoryResult] = []
        curves: Dict[str, List[np.ndarray]] = {g: [] for g in groups}
        time_axis = None

        # Flushed per history: hours of work must not be lost to a late crash.
        with rows_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for index, path in enumerate(paths, start=1):
                result, curve = analyse_history(
                    path, args.run_label, state_map, character, bands,
                    args.frame_mode, args.cycle_length, episode,
                    args.late_window_start_ns, args.shprop_chunk_rows,
                    args.curve_stride,
                )
                writer.writerow(result.as_row(groups))
                handle.flush()
                results.append(result)
                if time_axis is None:
                    time_axis = curve["_time_ns"]
                for g in groups:
                    curves[g].append(curve[g])
                print(f"  [{index}/{len(paths)}] {path.name}: "
                      f"NAMDTINI {result.namdtini}, "
                      + ", ".join(f"{g} {result.net_full.get(g, float('nan')):+.4f}"
                                  for g in groups)
                      + f", {result.episode_verdict}")

        summary = summarize_run(args.run_label, results, groups)
        summary["inputs"] = fingerprint(
            [Path(args.projection_character), Path(args.state_map)]
        )
        summary["environment"] = environment()
        summary["late_window_start_ns"] = args.late_window_start_ns
        summary["episode_window"] = list(episode) if episode else None
        write_json(out / "run_summary.json", summary)

        bands_out = {}
        for g in groups:
            try:
                bands_out[g] = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in ensemble_curve(curves[g]).items()
                }
            except EnsembleError as exc:
                bands_out[g] = {"error": str(exc)}
        write_json(out / "ensemble_curves.json", {
            "time_ns": time_axis.tolist() if time_axis is not None else [],
            "groups": bands_out,
            "note": (
                "quantiles across histories of one run, NOT a standard error: "
                "the histories share a nuclear trajectory and are not "
                "independent draws"
            ),
        })

        print(f"\nrun {args.run_label}: {len(results)} history/histories")
        print(f"  max decomposition residual {summary['max_decomposition_residual']:.3e}")
        for g in groups:
            block = summary["quantities"]["net_full"][g]
            print(f"  {g:>11}: median {block['median']:+.4f}  "
                  f"gain {block['n_gain']}/{block['n']}  "
                  f"loss {block['n_loss']}/{block['n']}")
        print("  episode verdicts:", summary["episode_verdicts"])
        print("  spread is between histories at fixed nuclei, not an uncertainty")
        print(f"written to {out}")
        return 0
    except (CrossingError, EnsembleError, ValueError, OSError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
