"""CLI for projection-weighted diagonal subsystem populations.

The populations written here are the diagonal contraction defined by
``character.POPULATION_DEFINITION``, not the exact subsystem populations of
the propagated state. Every output that carries a population carries that
qualification with it.
"""

from __future__ import annotations

import argparse
import glob as globlib
from pathlib import Path
from typing import List, Optional, Sequence

from .character import (
    DISCREPANCY_HEADER,
    POPULATION_DEFINITION,
    POPULATION_LABEL,
    SHPROP_IO_MODES,
    AtomGroupMap,
    CharacterError,
    character_populations,
    character_swap_rows,
    fixed_vs_projected_summary,
    plan_analysis,
    preflight_report,
    projection_table_rows,
    resolve_chunk_rows,
)
from .memory_budget import RETAIN_CHOICES, BudgetError, estimate_memory, human, parse_size
from .observables import LEGEND, OBSERVED
from .populations import StateMap
from .provenance import environment, fingerprint
from .report import prepare_output, write_csv, write_json
from .validation_summary import write as write_validation_summary

ALIGNMENT_KEYS = (
    "file",
    "path",
    "NAMDTINI",
    "NSW",
    "n_time_points",
    "header_cycle_length",
    "cycle_length_used",
    "cycle_length_source",
    "header_cycle_mismatch",
    "BMIN",
    "BMAX",
    "frame_mode",
    "first_projection_frame",
    "last_projection_frame",
    "first_five_frames",
    "last_five_frames",
    "unique_projection_frames_used",
    "wrap_count",
)

QUALITY_BAND_HEADER = [
    "band",
    "median_captured_projection",
    "minimum_captured_projection",
    "frame_of_minimum",
    "samples_below_threshold",
    "fraction_below_threshold",
    "most_common_character",
    "dominant_character_swaps",
    "adjacent_swaps",
    "changes_across_a_frame_gap",
    "cycle_wrap_swaps",
    "mixed_fraction",
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
    unique: List[Path] = []
    seen = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def build_parser(prog: str = "namd-analysis character-populations") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Combine each original SHPROP history with frame-dependent PROCAR "
            "subsystem character before ensemble averaging. Pass the ORIGINAL "
            "SHPROP files, never an averaged SHPROP.master: histories with "
            "different NAMDTINI select different PROCAR frames, and averaging "
            "first destroys that alignment."
        ),
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="original SHPROP files/globs (not SHPROP.master)",
    )
    parser.add_argument("--config", required=True, help="SHPROP state-map JSON")
    parser.add_argument(
        "--projection-manifest",
        required=True,
        help="JSON mapping MD frame numbers to PROCAR files",
    )
    parser.add_argument(
        "--atom-groups",
        required=True,
        help="JSON assigning one-based PROCAR ion indices to physical subsystems",
    )
    parser.add_argument(
        "--frame-mode",
        choices=("dish-cyclic", "linear"),
        required=True,
        help=(
            "how NAMD time points select electronic-structure frames; use dish-cyclic "
            "only for engines following RTTIME=mod(tion+NAMDTINI-1,period)"
        ),
    )
    parser.add_argument(
        "--dominance-threshold",
        type=float,
        default=0.6,
        help="maximum subsystem weight below which a frame/band is called mixed",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "check configuration and metadata only: no PROCAR projections are parsed "
            "and no populations are produced"
        ),
    )
    parser.add_argument(
        "--shprop-chunk-rows",
        type=int,
        default=None,
        help=(
            "rows of a SHPROP table to hold at once (default: chosen from file size). "
            "Chunking changes only residency, never the result"
        ),
    )
    parser.add_argument(
        "--shprop-io-mode",
        choices=SHPROP_IO_MODES,
        default="auto",
        help=(
            "auto picks a chunk size from the largest history; stream always chunks; "
            "memory reads each whole table in one chunk"
        ),
    )
    parser.add_argument(
        "--memory-budget",
        default=None,
        help=(
            "ceiling for the arrays this run will hold, e.g. 24G. Checked against an "
            "estimate made before any file is opened; a run that cannot fit is "
            "refused rather than started"
        ),
    )
    parser.add_argument(
        "--retain-per-file",
        choices=RETAIN_CHOICES,
        default="auto",
        help=(
            "keep the per-history projected populations. They are a diagnostic: the "
            "ensemble mean and standard error come from a running accumulator and "
            "never depend on them. auto decides from the budget"
        ),
    )
    parser.add_argument(
        "--accumulator-memmap-dir",
        default=None,
        help=(
            "spill the running ensemble mean/variance to memory-mapped files in this "
            "directory when they are large; without it they stay in RAM"
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="new output directory (optional with --preflight, which can print only)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


#: Above this many time points a figure is drawn from a min/max envelope
#: rather than from every sample.  A raster axis has a few thousand
#: distinguishable x positions, so plotting six million points costs gigabytes
#: of matplotlib path data to draw something no eye can resolve.
PLOT_ENVELOPE_THRESHOLD = 20_000

#: Buckets across the x axis when the envelope is used.
PLOT_ENVELOPE_BUCKETS = 4_000


def _envelope(time_ns, values, buckets: int):
    """Per-bucket min and max, so no excursion is hidden by decimation.

    Plain decimation (take every Nth sample) can step straight over a spike.
    Taking both extremes of each bucket cannot: whatever the series did inside
    a bucket, the band drawn for that bucket spans it.
    """
    import numpy as np

    n = len(time_ns)
    edges = np.linspace(0, n, buckets + 1, dtype=int)
    edges = np.unique(edges)
    centres = np.empty(len(edges) - 1, dtype=float)
    lows = np.empty(len(edges) - 1, dtype=float)
    highs = np.empty(len(edges) - 1, dtype=float)
    for i in range(len(edges) - 1):
        start, stop = edges[i], edges[i + 1]
        if stop <= start:
            stop = start + 1
        block = values[start:stop]
        centres[i] = float(time_ns[(start + stop - 1) // 2])
        lows[i] = float(block.min())
        highs[i] = float(block.max())
    return centres, lows, highs


def _plot(result, out: Path) -> List[str]:
    """Population figures, plus a fixed-vs-projected overlay per shared group.

    Large campaigns are drawn as a min/max envelope; the axis says so, because
    a reader must not take a decimated curve for every sample.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ntime = len(result.time_ns)
    decimated = ntime > PLOT_ENVELOPE_THRESHOLD
    outputs: List[str] = []

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for gi, group in enumerate(result.group_names):
        series = np.asarray(result.mean[:, gi])
        if decimated:
            x, low, high = _envelope(result.time_ns, series, PLOT_ENVELOPE_BUCKETS)
            ax.fill_between(x, low, high, alpha=0.85, label=group, linewidth=0)
        else:
            ax.plot(result.time_ns, series, label=group)
            if result.sem is not None:
                spread = np.asarray(result.sem[:, gi])
                ax.fill_between(
                    result.time_ns, series - spread, series + spread, alpha=0.2
                )
    ax.set(
        xlabel="Time (ns)",
        ylabel="Projection-weighted diagonal subsystem population",
        ylim=(-0.03, 1.03),
    )
    ax.set_title(
        "diagonal approximation: coherences are not recorded by SHPROP"
        + (
            f"  |  min/max envelope over {PLOT_ENVELOPE_BUCKETS} buckets of {ntime} samples"
            if decimated
            else ""
        ),
        fontsize=8,
        loc="left",
    )
    ax.legend()
    for suffix in ("png", "pdf"):
        path = out / f"character_populations.{suffix}"
        fig.savefig(path, dpi=200)
        outputs.append(str(path))
    plt.close(fig)

    # Fixed-column against projection-weighted, one panel per shared group.
    shared = [name for name in result.fixed_mean if name in result.group_names]
    if shared:
        lookup = {name: i for i, name in enumerate(result.group_names)}
        fig, axes = plt.subplots(
            len(shared), 1, figsize=(8, 2.6 * len(shared)), sharex=True,
            constrained_layout=True, squeeze=False,
        )
        for row, name in enumerate(shared):
            panel = axes[row][0]
            fixed = np.asarray(result.fixed_mean[name])
            projected = np.asarray(result.mean[:, lookup[name]])
            if decimated:
                x, low, high = _envelope(result.time_ns, fixed, PLOT_ENVELOPE_BUCKETS)
                panel.fill_between(x, low, high, alpha=0.45, label="fixed column", linewidth=0)
                x, low, high = _envelope(result.time_ns, projected, PLOT_ENVELOPE_BUCKETS)
                panel.fill_between(
                    x, low, high, alpha=0.45, label="projection-weighted", linewidth=0
                )
            else:
                panel.plot(result.time_ns, fixed, label="fixed column")
                panel.plot(result.time_ns, projected, label="projection-weighted")
                if result.sem is not None:
                    spread = np.asarray(result.sem[:, lookup[name]])
                    panel.fill_between(
                        result.time_ns, projected - spread, projected + spread, alpha=0.2
                    )
            panel.set(ylabel=name, ylim=(-0.03, 1.03))
            if row == 0:
                panel.legend(fontsize=8)
                panel.set_title(
                    "fixed column map vs projection-weighted diagonal subsystem "
                    "population; a difference is mislabelling, not a transfer rate",
                    fontsize=8,
                    loc="left",
                )
        axes[-1][0].set(xlabel="Time (ns)")
        for suffix in ("png", "pdf"):
            path = out / f"fixed_vs_projected.{suffix}"
            fig.savefig(path, dpi=200)
            outputs.append(str(path))
        plt.close(fig)
    return outputs


def _run_preflight(args, paths, state_map, atom_groups) -> int:
    report = preflight_report(
        paths,
        state_map,
        args.projection_manifest,
        atom_groups,
        args.frame_mode,
        memory_budget=parse_size(args.memory_budget) if args.memory_budget else None,
        retain_per_file=args.retain_per_file,
        memmap_dir=args.accumulator_memmap_dir,
    )
    payload = {
        "command": "character-populations --preflight",
        "environment": environment(),
        "preflight": report,
    }
    if args.out:
        out = prepare_output(args.out, overwrite=args.overwrite)
        write_json(out / "preflight.json", payload)

    print(f"preflight: {report['n_shprop_files']} SHPROP file(s), mode {report['frame_mode']}")
    if report.get("band_window"):
        window = report["band_window"]
        print(
            f"  basis: BMIN={window['BMIN']} BMAX={window['BMAX']} "
            f"({window['basis_size']} states), state map declares "
            f"{len(window['state_map_population_columns'])} population columns"
        )
    if report.get("distinct_namdtini") is not None:
        print(f"  distinct NAMDTINI: {report['distinct_namdtini']}")
        print(f"  SHPROP row counts: {report['row_counts']}")
    io_block = report.get("shprop_io")
    if io_block:
        megabytes = io_block["total_bytes"] / (1024 * 1024)
        print(
            f"  SHPROP input: {len(io_block['per_file'])} file(s), {megabytes:.1f} MiB total, "
            f"{io_block['rows_per_history']} rows each; analysis would read "
            f"{io_block['chunk_rows']} row(s) at a time ({io_block['reason']})"
        )
    cycle = report.get("cycle")
    if cycle:
        print(
            f"  cycle: manifest={cycle['explicit_manifest_cycle_length']}, "
            f"NSW-1={cycle['header_derived_periods_nsw_minus_1']}, "
            f"disagree={cycle['disagree']}"
        )
        if cycle.get("period_used") is not None:
            print(
                f"  cycle wrap: frame {cycle['period_used']} -> 1 crossed by "
                f"{cycle['histories_crossing_the_wrap']} of "
                f"{report['n_shprop_files']} history/histories"
            )
    frames = report.get("frames")
    if frames:
        print(
            f"  frames: manifest declares {frames['manifest_declared_frames']} "
            f"({frames['manifest_frame_min']}..{frames['manifest_frame_max']}); "
            f"{frames['frames_required_by_shprop']} required; "
            f"{frames['procars_skipped']} PROCARs will not be parsed"
        )
    if report.get("representative_procar"):
        structure = report["representative_procar"]
        print(
            f"  representative PROCAR (frame {structure['frame']}): "
            f"{structure['n_ions']} ions, {structure['n_bands']} bands, "
            f"{structure['n_kpoints']} k-point(s), spins "
            f"{structure['spin_components_seen'] or '[unlabelled]'}"
        )
    memory = report.get("memory")
    if memory and "estimated_total_human" in memory:
        print(
            f"  estimated memory: {memory['estimated_resident_human']} of arrays "
            f"(+{human(memory['assumed_overhead_bytes'])} assumed overhead) = "
            f"{memory['estimated_total_human']}"
            + (f", budget {memory['budget_human']}" if memory.get("budget_human") else "")
        )
        print(
            f"  plan: {memory['shprop_chunk_rows']} row chunks, "
            f"retain per-history results = {memory['retain_per_file']}, "
            f"memmap accumulators = {memory['spill_accumulators_to_memmap']}"
        )
    window = report.get("band_window")
    if window and window.get("band_numbers"):
        print(
            f"  bands: {window['band_numbers'][0]}..{window['band_numbers'][-1]} "
            f"({len(window['band_numbers'])} states), source {window['band_numbers_source']}"
        )
    provenance = report.get("provenance")
    if provenance:
        sources = sorted({row["source"] for row in provenance["NAMDTINI"]})
        print(f"  NAMDTINI provenance: {', '.join(sources)}")
        periods = sorted({str(row["source"]) for row in provenance["cycle_period"]})
        print(f"  cycle period provenance: {', '.join(periods)}")
    if report.get("atom_coverage"):
        coverage = report["atom_coverage"]
        print(
            f"  atom coverage: {coverage['assigned_ions']} of {coverage['procar_ions']} "
            f"ions assigned (complete_atoms={coverage['complete_atoms']})"
        )
    for warning in report.get("warnings", []):
        print(f"  warning: {warning}")
    for problem in report.get("problems", []):
        print(f"  PROBLEM: {problem}")
    if report["ok"]:
        print("preflight OK: nothing blocking was found")
    else:
        print(f"preflight found {len(report['problems'])} blocking problem(s)")
    if args.out:
        print(f"written to {args.out}")
    return 0 if report["ok"] else 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.preflight and not args.out:
        print("error: --out is required unless --preflight is given")
        return 2
    try:
        paths = _expand(args.files)
        state_map = StateMap.from_json(args.config)
        atom_groups = AtomGroupMap.from_json(args.atom_groups)
        if args.preflight:
            return _run_preflight(args, paths, state_map, atom_groups)

        plan = plan_analysis(paths, state_map, args.projection_manifest, args.frame_mode)
        chunk_rows, io_plan = resolve_chunk_rows(
            plan.shprop_structures, args.shprop_io_mode, args.shprop_chunk_rows
        )
        budget = parse_size(args.memory_budget) if args.memory_budget else None
        memory = estimate_memory(
            n_files=len(paths),
            n_time=plan.n_time[0],
            n_states=len(plan.bands),
            n_groups=len(atom_groups.names),
            n_fixed_groups=len(state_map.groups),
            n_frames=len(plan.required_frames),
            n_bands=len(plan.bands),
            n_columns=plan.shprop_structures[0].n_columns,
            chunk_rows=chunk_rows,
            budget=budget,
            retain_per_file=args.retain_per_file,
            memmap_dir=args.accumulator_memmap_dir,
        )
        for line in memory.decisions:
            print(f"memory: {line}")
        if not memory.ok:
            for problem in memory.problems:
                print(f"error: {problem}")
            return 2
        result = character_populations(
            paths,
            state_map,
            args.projection_manifest,
            atom_groups,
            args.frame_mode,
            plan=plan,
            chunk_rows=chunk_rows,
            memmap_dir=Path(args.accumulator_memmap_dir)
            if (args.accumulator_memmap_dir and memory.spill_accumulators)
            else None,
            keep_per_file=memory.retain_per_file,
        )
        swaps, swap_summary = character_swap_rows(
            result.projection, dominance_threshold=args.dominance_threshold
        )
        discrepancy = fixed_vs_projected_summary(result)
        dominance = result.projection.dominance_summary(args.dominance_threshold)
    except (CharacterError, BudgetError, ValueError, OSError) as exc:
        print(f"error: {exc}")
        return 2

    out = prepare_output(args.out, overwrite=args.overwrite)

    # Generators, not lists. write_csv consumes an iterable, and a campaign of
    # six million time points times three groups is eighteen million rows: as
    # Python lists that is gigabytes staged in memory purely to hand it to a
    # writer one row at a time.
    def _population_rows():
        for ti, time in enumerate(result.time_ns):
            for gi, group in enumerate(result.group_names):
                yield [
                    float(time),
                    group,
                    float(result.mean[ti, gi]),
                    None if result.sem is None else float(result.sem[ti, gi]),
                    OBSERVED,
                ]

    write_csv(
        out / "character_populations.csv",
        [
            "time_ns",
            "group",
            "projection_weighted_diagonal_population",
            "sem",
            "observable_class",
        ],
        _population_rows(),
    )

    projected_lookup = {name: i for i, name in enumerate(result.group_names)}

    def _comparison_rows():
        for group, fixed in result.fixed_mean.items():
            if group not in projected_lookup:
                continue
            projected = result.mean[:, projected_lookup[group]]
            for time, fixed_value, projected_value in zip(
                result.time_ns, fixed, projected
            ):
                yield [
                    float(time),
                    group,
                    float(fixed_value),
                    float(projected_value),
                    float(projected_value - fixed_value),
                ]

    write_csv(
        out / "fixed_vs_projected.csv",
        [
            "time_ns",
            "group",
            "fixed_column_population",
            "projection_weighted_diagonal_population",
            "difference",
        ],
        _comparison_rows(),
    )
    write_csv(
        out / "fixed_vs_projected_summary.csv", DISCREPANCY_HEADER, discrepancy["rows"]
    )
    write_csv(
        out / "character_swaps.csv",
        [
            "band",
            "frame_before",
            "frame_after",
            "dominant_before",
            "dominant_after",
            "confidence_before",
            "confidence_after",
            "frame_gap",
            "resolution",
        ],
        swaps,
    )
    quality = result.projection.quality_summary()
    dominance_by_band = {entry["band"]: entry for entry in dominance["per_band"]}
    write_csv(
        out / "projection_quality_by_band.csv",
        QUALITY_BAND_HEADER,
        [
            [
                entry["band"],
                entry["median_captured_projection"],
                entry["minimum_captured_projection"],
                entry["frame_of_minimum"],
                entry["samples_below_threshold"],
                entry["fraction_below_threshold"],
                dominance_by_band[entry["band"]]["most_common_character"],
                dominance_by_band[entry["band"]]["dominant_character_swaps"],
                dominance_by_band[entry["band"]]["adjacent_swaps"],
                dominance_by_band[entry["band"]]["changes_across_a_frame_gap"],
                dominance_by_band[entry["band"]]["cycle_wrap_swaps"],
                dominance_by_band[entry["band"]]["mixed_fraction"],
            ]
            for entry in quality["per_band"]
        ],
    )
    write_csv(
        out / "projection_character.csv",
        [
            "frame",
            "band",
            "group",
            "normalized_weight",
            "captured_projection",
            "total_projection",
            "dominant_group",
            "dominant_weight",
        ],
        projection_table_rows(result.projection),
    )
    write_csv(
        out / "shprop_alignment.csv",
        list(ALIGNMENT_KEYS),
        [
            [
                ";".join(str(v) for v in record[key])
                if isinstance(record.get(key), list)
                else record.get(key)
                for key in ALIGNMENT_KEYS
            ]
            for record in result.file_alignment
        ],
    )
    figures = _plot(result, out)

    input_paths = paths + [Path(args.config), Path(args.projection_manifest), Path(args.atom_groups)]
    input_paths += result.projection.source_paths
    payload = {
        "command": "character-populations",
        "environment": environment(),
        "inputs": fingerprint(input_paths),
        "frame_mode": args.frame_mode,
        "projection_cycle_length": result.projection.cycle_length,
        "cycle_period_used": result.projection.cycle_period,
        "state_map": state_map.as_dict(),
        "atom_groups": {
            "groups": atom_groups.groups,
            "complete_atoms": atom_groups.complete_atoms,
            "min_projection_weight": atom_groups.min_projection_weight,
        },
        "frame_consumption": plan.consumption(),
        "procar_parsing": result.projection.parse_stats,
        "band_provenance_conflicts": plan.band_provenance_conflicts,
        "shprop_io": {**io_plan, **result.io},
        "memory": memory.as_dict(),
        "projection_frames": [int(v) for v in result.projection.frames],
        "projection_bands": [int(v) for v in result.projection.bands],
        "projection_quality": quality,
        "dominance": dominance,
        "character_swaps": swap_summary,
        "shprop_alignment": result.file_alignment,
        "population_definition": POPULATION_DEFINITION,
        "projection_weighted_population_conservation": result.conservation,
        "cycle_wrap": swap_summary["cycle_wrap"],
        "fixed_vs_projected_summary": discrepancy,
        "shared_fixed_projection_groups": sorted(
            set(result.fixed_mean).intersection(result.group_names)
        ),
        "figures": figures,
        "observable_class": OBSERVED,
        "observable_class_legend": LEGEND,
        "interpretation_limits": [
            POPULATION_DEFINITION,
            "the reported population is the diagonal contraction of SHPROP "
            "populations with PROCAR subsystem weights: electronic coherences "
            "are absent from both input files and are therefore omitted from "
            "the result rather than estimated or bounded",
            "projection-weighted populations are derived directly from SHPROP "
            "populations and declared PROCAR subsystem projections; no kinetic "
            "model or nearest-energy band tracking is used",
            "dominant-character swap counts use one transition set: adjacent "
            "steps, changes across unexamined frame gaps and the cyclic "
            "period-to-first-frame step are counted separately, and the "
            "per-band counts sum to the campaign total",
            "each original SHPROP is projected before ensemble averaging because "
            "different NAMDTINI values select different electronic-structure frames",
            "for dish-cyclic alignment an explicit manifest cycle_length overrides "
            "SHPROP NSW-1 and any mismatch is retained in the alignment report",
            "PROCAR weights are normalized across the declared physical subsystems; "
            "captured_projection is reported so weak PAW-sphere projection can be audited",
            "a change of dominant adiabatic-state character is not itself a surface hop, "
            "and neither is a population change a transfer rate or an extraction",
        ],
    }
    write_json(out / "report.json", payload)
    summary_path = write_validation_summary(out / "validation_summary.md", payload)

    consumption = plan.consumption()
    print(
        f"{len(paths)} SHPROP file(s), {len(result.projection.frames)} projection frame(s) "
        f"parsed of {consumption['manifest_declared_frames']} declared, "
        f"{len(result.projection.bands)} band(s)"
    )
    for conflict in plan.band_provenance_conflicts:
        print(f"warning: {conflict}")
    print(f"reporting the {POPULATION_LABEL} (coherences are not in the inputs)")
    print(
        f"estimated peak {human(memory.resident_bytes)} of arrays plus assumed overhead"
        + (f", budget {human(memory.budget_bytes)}" if memory.budget_bytes else "")
    )
    print(
        f"SHPROP read in {result.io['chunks_per_history']} chunk(s) of "
        f"{result.io['shprop_chunk_rows']} row(s) per history "
        f"(mode {io_plan['requested_mode']}); no whole-campaign stack was built"
    )
    print(
        f"groups: {', '.join(result.group_names)}; dominant-character swaps: "
        f"{swap_summary['dominant_character_swaps']} "
        f"({swap_summary['adjacent_swaps']} adjacent, "
        f"{swap_summary['changes_across_a_frame_gap']} across a frame gap, "
        f"{swap_summary['cycle_wrap_swaps']} at the cycle wrap)"
    )
    wrap = swap_summary["cycle_wrap"]
    if args.frame_mode == "dish-cyclic" or wrap["period"] is not None:
        state = "examined" if wrap["examined"] else f"excluded ({wrap['state']})"
        print(f"cycle wrap frame {wrap['frame_before']} -> {wrap['frame_after']}: {state}")
    print(
        f"captured projection median={quality['median_captured_projection']:.3g} "
        f"(p5={quality['p5_captured_projection']:.3g}, "
        f"min={quality['minimum_captured_projection']:.3g}); "
        f"{quality['samples_below_threshold']} frame/band samples below "
        f"{quality['quality_threshold']:.3g}"
    )
    largest = discrepancy.get("largest_disagreement")
    if largest:
        print(
            f"largest fixed-vs-projected disagreement: {largest['group']} "
            f"{largest['max_abs_difference']:.4g} at {largest['time_of_max_ns']:.4g} ns "
            f"(RMS {largest['rms_difference']:.4g})"
        )
    if any(record.get("header_cycle_mismatch") for record in result.file_alignment):
        print(
            "note: projection-manifest cycle_length differs from SHPROP NSW-1; "
            "the explicit manifest period was used and the mismatch was recorded"
        )
    print(f"summary: {summary_path}")
    print(f"written to {out}")
    # Every output is on disk; the running accumulators can go, taking any
    # memory-mapped spill files with them.
    result.release()
    return 0
