"""A compact, human-readable account of one character run.

``report.json`` holds everything; this holds what a reader needs in order to
decide whether to believe the numbers, in the order they would ask.  It is
meant to be pasted into project notes and later rewritten as SI text, so it
states provenance and limitations as prominently as results.

Nothing here computes anything.  Every value is read from the run that already
happened, so the summary cannot disagree with the report beside it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .character import POPULATION_DEFINITION
from .memory_budget import human


def _human(value: Optional[int]) -> Optional[str]:
    """Bytes as a size, or None when the report did not record one."""
    return None if value is None else human(int(value))


def _table(header: List[str], rows: List[List[Any]]) -> List[str]:
    if not rows:
        return ["_none_", ""]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    lines.append("")
    return lines


def _fmt(value: Any, places: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{places}g}"
    return str(value)


def render(payload: Dict[str, Any]) -> str:
    """Build the summary text from a finished ``report.json`` payload."""
    out: List[str] = []
    env = payload.get("environment", {})
    out += [
        "# Character analysis validation summary",
        "",
        f"- package version: `{env.get('version', 'unknown')}`",
        f"- python: `{env.get('python', 'unknown')}`",
        f"- command: `{payload.get('command', 'character-populations')}`",
        f"- frame mode: `{payload.get('frame_mode')}`",
        "",
        "## What the numbers are",
        "",
        POPULATION_DEFINITION,
        "",
    ]

    # --- provenance --------------------------------------------------------
    out += ["## Provenance", ""]
    alignment = payload.get("shprop_alignment", []) or []
    if alignment:
        first = alignment[0]
        out += [
            f"**Basis.** VASP bands `{first.get('band_numbers')}`, "
            f"source `{first.get('band_numbers_source')}`. "
            "BMIN/BMAX are properties of the NAMD input and need not appear in a "
            "SHPROP table, so the basis is never inferred from one.",
            "",
        ]
        out += _table(
            ["history", "NAMDTINI", "origin source", "cycle period", "period source", "disagrees with NSW-1"],
            [
                [
                    r.get("file"),
                    r.get("NAMDTINI"),
                    r.get("NAMDTINI_source"),
                    r.get("cycle_length_used"),
                    r.get("cycle_length_source"),
                    r.get("header_cycle_mismatch"),
                ]
                for r in alignment
            ],
        )

    # --- atom mapping ------------------------------------------------------
    groups = (payload.get("atom_groups") or {}).get("groups") or {}
    if groups:
        out += ["## Atom mapping", ""]
        out += _table(
            ["subsystem", "ions"],
            [[name, len(ions)] for name, ions in groups.items()],
        )
        out += [
            "Subsystem names and atom boundaries are declarations, never inferred "
            "from element symbols or contiguity.",
            "",
        ]

    # --- frame alignment ---------------------------------------------------
    consumption = payload.get("frame_consumption") or {}
    parsing = payload.get("procar_parsing") or {}
    if consumption or parsing:
        out += ["## Frame alignment and PROCAR parsing", ""]
        out += _table(
            ["quantity", "value"],
            [
                ["manifest declared frames", consumption.get("manifest_declared_frames")],
                ["frames required by SHPROP", consumption.get("frames_required_by_shprop")],
                ["PROCARs parsed", parsing.get("frames_parsed", consumption.get("procars_parsed"))],
                ["PROCARs skipped", parsing.get("frames_skipped", consumption.get("procars_skipped"))],
                ["unique files parsed", parsing.get("unique_files_parsed")],
                ["bytes read", parsing.get("bytes_read")],
            ],
        )

    # --- memory ------------------------------------------------------------
    memory = payload.get("memory") or {}
    io_block = payload.get("shprop_io") or {}
    if memory or io_block:
        out += ["## Memory mode", ""]
        out += _table(
            ["quantity", "value"],
            [
                ["estimated resident arrays", memory.get("estimated_resident_human")],
                ["assumed overhead", _human(memory.get("assumed_overhead_bytes"))],
                ["estimated total", memory.get("estimated_total_human")],
                ["budget", memory.get("budget_human")],
                ["SHPROP chunk rows", io_block.get("chunk_rows")],
                ["chunks per history", io_block.get("chunks_per_history")],
                ["per-history results retained", memory.get("retain_per_file")],
                ["accumulators memory-mapped", memory.get("spill_accumulators_to_memmap")],
            ],
        )

    # --- projection quality ------------------------------------------------
    quality = payload.get("projection_quality") or {}
    if quality:
        out += ["## Projection quality", ""]
        out += _table(
            ["statistic", "captured projection"],
            [
                ["minimum", _fmt(quality.get("minimum_captured_projection"))],
                ["1st percentile", _fmt(quality.get("p1_captured_projection"))],
                ["5th percentile", _fmt(quality.get("p5_captured_projection"))],
                ["median", _fmt(quality.get("median_captured_projection"))],
                ["95th percentile", _fmt(quality.get("p95_captured_projection"))],
                ["maximum", _fmt(quality.get("maximum_captured_projection"))],
                ["samples below threshold", quality.get("samples_below_threshold")],
                ["fraction below threshold", _fmt(quality.get("fraction_below_threshold"))],
            ],
        )
        per_band = quality.get("per_band") or []
        if per_band:
            out += ["Per band:", ""]
            out += _table(
                ["band", "median capture", "minimum capture", "frame of minimum", "below threshold"],
                [
                    [
                        b.get("band"),
                        _fmt(b.get("median_captured_projection")),
                        _fmt(b.get("minimum_captured_projection")),
                        b.get("frame_of_minimum"),
                        b.get("samples_below_threshold"),
                    ]
                    for b in per_band
                ],
            )
        worst = quality.get("worst_frame_band_samples") or []
        if worst:
            out += ["Worst individual frame/band samples:", ""]
            out += _table(
                ["frame", "band", "captured", "total"],
                [
                    [w.get("frame"), w.get("band"), _fmt(w.get("captured_projection")), _fmt(w.get("total_projection"))]
                    for w in worst[:10]
                ],
            )
        out += [
            "A low captured projection means the band lies largely outside every "
            "declared PAW sphere, so its normalized character is poorly determined. "
            "**Nothing is discarded, repaired or reweighted on this basis** -- it is "
            "reported so the reader can decide.",
            "",
        ]

    # --- conservation ------------------------------------------------------
    conservation = payload.get("projection_weighted_population_conservation") or {}
    if conservation:
        out += ["## Population conservation", ""]
        out += _table(
            ["quantity", "value"],
            [
                ["minimum total", _fmt(conservation.get("min"), 8)],
                ["maximum total", _fmt(conservation.get("max"), 8)],
                ["mean total", _fmt(conservation.get("mean"), 8)],
                ["max deviation from one", _fmt(conservation.get("max_abs_deviation_from_one"))],
            ],
        )
        out += [
            "Conservation is checked on every row of every chunk before projection, "
            "and on the ensemble mean after it. The tolerance is the one used "
            "everywhere else in the package and is not relaxed for large inputs.",
            "",
        ]

    # --- fixed vs dynamic --------------------------------------------------
    discrepancy = payload.get("fixed_vs_projected_summary") or {}
    per_group = discrepancy.get("per_group") or []
    if per_group:
        out += ["## Fixed column map vs projection-weighted character", ""]
        out += _table(
            [
                "group",
                "max abs difference",
                "time of max (ns)",
                "RMS difference",
                "integrated abs difference (ns)",
                "mean signed difference",
                "fixed at max",
                "dynamic at max",
            ],
            [
                [
                    g.get("group"),
                    _fmt(g.get("max_abs_difference")),
                    _fmt(g.get("time_of_max_ns")),
                    _fmt(g.get("rms_difference")),
                    _fmt(g.get("integrated_abs_difference_ns")),
                    _fmt(g.get("mean_signed_difference")),
                    _fmt(g.get("fixed_at_max")),
                    _fmt(g.get("projected_at_max")),
                ]
                for g in per_group
            ],
        )
        only_fixed = discrepancy.get("groups_only_in_fixed_map") or []
        only_projected = discrepancy.get("groups_only_in_projection") or []
        if only_fixed or only_projected:
            out += [
                f"Groups only in the fixed map: `{only_fixed}`. "
                f"Groups only in the projection: `{only_projected}`. "
                "These have no counterpart to compare against and are listed rather "
                "than dropped.",
                "",
            ]
        out += [
            "A difference means the fixed labelling was wrong somewhere. It is not a "
            "flux, a transfer rate or an extraction.",
            "",
        ]

    # --- swaps -------------------------------------------------------------
    swaps = payload.get("character_swaps") or {}
    if swaps:
        out += ["## Character swaps", ""]
        out += _table(
            ["quantity", "value"],
            [
                ["dominant-character swaps", swaps.get("dominant_character_swaps")],
                ["adjacent (fully resolved)", swaps.get("adjacent_swaps")],
                ["across an unexamined frame gap", swaps.get("changes_across_a_frame_gap")],
                ["at the cyclic wrap", swaps.get("cycle_wrap_swaps")],
                ["MD frames skipped between examined frames", swaps.get("md_frames_skipped_between_examined_frames")],
                ["mixed frame/band samples", swaps.get("mixed_frame_band_samples")],
            ],
        )
        wrap = swaps.get("cycle_wrap") or {}
        out += [
            f"Cycle wrap: {wrap.get('state', 'unknown')}. {wrap.get('note', '')}",
            "",
            "A dominant-character swap is **not** a surface hop. It is a change in "
            "the chemical character of an adiabatic eigenstate.",
            "",
        ]

    # --- limitations -------------------------------------------------------
    out += ["## Limitations", ""]
    limits = payload.get("interpretation_limits") or []
    for item in limits:
        out.append(f"- {item}")
    out += [
        "- preflight checks structure, not every value; the analysis validates "
        "every row of every chunk",
        "- a plot of a long campaign is drawn as a min/max envelope, which bounds "
        "every excursion but is not every sample",
        "",
    ]
    return "\n".join(out).rstrip() + "\n"


def write(path, payload: Dict[str, Any]) -> Optional[Path]:
    """Render and write the summary beside the rest of the outputs."""
    path = Path(path)
    path.write_text(render(payload), encoding="utf-8")
    return path
