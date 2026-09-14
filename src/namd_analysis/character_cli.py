"""CLI for projection-weighted physical-state populations."""

from __future__ import annotations

import argparse
import glob as globlib
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .character import (
    AtomGroupMap,
    CharacterError,
    character_populations,
    character_swap_rows,
    projection_table_rows,
)
from .observables import LEGEND, OBSERVED
from .populations import StateMap
from .provenance import environment, fingerprint
from .report import prepare_output, write_csv, write_json


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="namd-analysis character-populations",
        description=(
            "Combine each original SHPROP history with frame-dependent PROCAR "
            "subsystem character before ensemble averaging."
        ),
    )
    parser.add_argument("--files", nargs="+", required=True, help="original SHPROP files/globs")
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
            "only for engines following RTTIME=mod(tion+NAMDTINI-1,NSW-1)"
        ),
    )
    parser.add_argument(
        "--dominance-threshold",
        type=float,
        default=0.6,
        help="maximum subsystem weight below which a frame/band is called mixed",
    )
    parser.add_argument("--out", required=True, help="new output directory")
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = _expand(args.files)
        state_map = StateMap.from_json(args.config)
        atom_groups = AtomGroupMap.from_json(args.atom_groups)
        result = character_populations(
            paths,
            state_map,
            args.projection_manifest,
            atom_groups,
            args.frame_mode,
        )
        swaps, swap_summary = character_swap_rows(
            result.projection, dominance_threshold=args.dominance_threshold
        )
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
        [
            "path",
            "NAMDTINI",
            "NSW",
            "BMIN",
            "BMAX",
            "frame_mode",
            "first_projection_frame",
            "last_projection_frame",
            "unique_projection_frames_used",
        ],
        [
            [record.get(key) for key in (
                "path",
                "NAMDTINI",
                "NSW",
                "BMIN",
                "BMAX",
                "frame_mode",
                "first_projection_frame",
                "last_projection_frame",
                "unique_projection_frames_used",
            )]
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
        "state_map": state_map.as_dict(),
        "atom_groups": {
            "groups": atom_groups.groups,
            "complete_atoms": atom_groups.complete_atoms,
            "min_projection_weight": atom_groups.min_projection_weight,
        },
        "projection_frames": [int(v) for v in result.projection.frames],
        "projection_bands": [int(v) for v in result.projection.bands],
        "projection_quality": result.projection.quality_summary(),
        "character_swaps": swap_summary,
        "shprop_alignment": result.file_alignment,
        "physical_population_conservation": result.conservation,
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
            "PROCAR weights are normalized across the declared physical subsystems; "
            "captured_projection is reported so weak PAW-sphere projection can be audited",
            "a change of dominant adiabatic-state character is not itself a surface hop",
        ],
    }
    write_json(out / "report.json", payload)

    quality = payload["projection_quality"]
    print(
        f"{len(paths)} SHPROP file(s), {len(result.projection.frames)} projection frame(s), "
        f"{len(result.projection.bands)} band(s)"
    )
    print(
        f"projection-weighted groups: {', '.join(result.group_names)}; "
        f"dominant-character swaps: {swap_summary['dominant_character_swaps']}"
    )
    print(
        f"captured projection median={quality['median_captured_projection']:.3g}; "
        f"{quality['samples_below_threshold']} frame/band samples below "
        f"{quality['quality_threshold']:.3g}"
    )
    print(f"written to {out}")
    return 0
