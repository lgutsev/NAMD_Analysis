"""CLI for projection-weighted physical-state populations."""

from __future__ import annotations

import argparse
import glob as globlib
from pathlib import Path
from typing import List, Optional, Sequence

from .character import (
    DISCREPANCY_HEADER,
    AtomGroupMap,
    CharacterError,
    character_populations,
    character_swap_rows,
    fixed_vs_projected_summary,
    plan_analysis,
    preflight_report,
    projection_table_rows,
)
from .observables import LEGEND, OBSERVED
from .populations import StateMap
from .provenance import environment, fingerprint
from .report import prepare_output, write_csv, write_json

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
        "--out",
        default=None,
        help="new output directory (optional with --preflight, which can print only)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _plot(result, out: Path) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for gi, group in enumerate(result.group_names):
        ax.plot(result.time_ns, result.mean[:, gi], label=group)
        if result.sem is not None:
            low = result.mean[:, gi] - result.sem[:, gi]
            high = result.mean[:, gi] + result.sem[:, gi]
            ax.fill_between(result.time_ns, low, high, alpha=0.2)
    ax.set(xlabel="Time (ns)", ylabel="Projection-weighted population", ylim=(-0.03, 1.03))
    ax.legend()
    outputs = []
    for suffix in ("png", "pdf"):
        path = out / f"character_populations.{suffix}"
        fig.savefig(path, dpi=200)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def _run_preflight(args, paths, state_map, atom_groups) -> int:
    report = preflight_report(
        paths, state_map, args.projection_manifest, atom_groups, args.frame_mode
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
    cycle = report.get("cycle")
    if cycle:
        print(
            f"  cycle: manifest={cycle['explicit_manifest_cycle_length']}, "
            f"NSW-1={cycle['header_derived_periods_nsw_minus_1']}, "
            f"disagree={cycle['disagree']}"
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
        result = character_populations(
            paths,
            state_map,
            args.projection_manifest,
            atom_groups,
            args.frame_mode,
            plan=plan,
        )
        swaps, swap_summary = character_swap_rows(
            result.projection, dominance_threshold=args.dominance_threshold
        )
        discrepancy = fixed_vs_projected_summary(result)
        dominance = result.projection.dominance_summary(args.dominance_threshold)
    except (CharacterError, ValueError, OSError) as exc:
        print(f"error: {exc}")
        return 2

    out = prepare_output(args.out, overwrite=args.overwrite)
    rows = []
    for ti, time in enumerate(result.time_ns):
        for gi, group in enumerate(result.group_names):
            rows.append(
                [
                    float(time),
                    group,
                    float(result.mean[ti, gi]),
                    None if result.sem is None else float(result.sem[ti, gi]),
                    OBSERVED,
                ]
            )
    write_csv(
        out / "character_populations.csv",
        ["time_ns", "group", "population", "sem", "observable_class"],
        rows,
    )

    comparison_rows = []
    projected_lookup = {name: i for i, name in enumerate(result.group_names)}
    for group, fixed in result.fixed_mean.items():
        if group not in projected_lookup:
            continue
        gi = projected_lookup[group]
        projected = result.mean[:, gi]
        for time, fixed_value, projected_value in zip(result.time_ns, fixed, projected):
            comparison_rows.append(
                [
                    float(time),
                    group,
                    float(fixed_value),
                    float(projected_value),
                    float(projected_value - fixed_value),
                ]
            )
    write_csv(
        out / "fixed_vs_projected.csv",
        ["time_ns", "group", "fixed_population", "projected_population", "difference"],
        comparison_rows,
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
        list(projection_table_rows(result.projection)),
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
        "state_map": state_map.as_dict(),
        "atom_groups": {
            "groups": atom_groups.groups,
            "complete_atoms": atom_groups.complete_atoms,
            "min_projection_weight": atom_groups.min_projection_weight,
        },
        "frame_consumption": plan.consumption(),
        "projection_frames": [int(v) for v in result.projection.frames],
        "projection_bands": [int(v) for v in result.projection.bands],
        "projection_quality": quality,
        "dominance": dominance,
        "character_swaps": swap_summary,
        "shprop_alignment": result.file_alignment,
        "physical_population_conservation": result.conservation,
        "fixed_vs_projected_summary": discrepancy,
        "shared_fixed_projection_groups": sorted(
            set(result.fixed_mean).intersection(result.group_names)
        ),
        "figures": figures,
        "observable_class": OBSERVED,
        "observable_class_legend": LEGEND,
        "interpretation_limits": [
            "projection-weighted populations are derived directly from SHPROP "
            "populations and declared PROCAR subsystem projections; no kinetic "
            "model or nearest-energy band tracking is used",
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

    consumption = plan.consumption()
    print(
        f"{len(paths)} SHPROP file(s), {len(result.projection.frames)} projection frame(s) "
        f"parsed of {consumption['manifest_declared_frames']} declared, "
        f"{len(result.projection.bands)} band(s)"
    )
    print(
        f"projection-weighted groups: {', '.join(result.group_names)}; "
        f"dominant-character swaps: {swap_summary['dominant_character_swaps']}"
    )
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
    print(f"written to {out}")
    return 0
