"""``crossing_summary.md`` -- adiabatic character events, stated for a referee.

The whole value of this file is in what it refuses to say. A dominant-character
swap is reported as a swap; it becomes transfer only where the fragment
population moved with it, and where it did not, the summary says so.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _table(header: List[str], rows: List[List[Any]]) -> List[str]:
    if not rows:
        return ["_none_", ""]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        lines.append(
            "| " + " | ".join("—" if v is None else str(v) for v in row) + " |"
        )
    lines.append("")
    return lines


def _per_history_section(payload: Dict[str, Any]) -> List[str]:
    """What each history did, before anything was summed.

    A referee asking "is this transfer?" is asking about a trajectory, not
    about an average, so the per-history counts come before the totals.
    """
    aggregate = payload.get("per_history")
    if not aggregate:
        return []

    out: List[str] = [
        "## Per history",
        "",
        f"{aggregate['n_histories']} SHPROP histories, starting at "
        f"`NAMDTINI` {aggregate['distinct_namdtini']}. Each was classified on "
        "its **own** resolved frame mapping, carrying its **own** "
        "projection-weighted fragment population "
        "`P_g(t) = sum_i P_i(t) w_ig[f(t)]`.",
        "",
    ]
    if len(aggregate["distinct_namdtini"]) > 1:
        out += [
            "> The starts differ, so the histories occupy **different frames at "
            "the same row**, and a cyclic mapping wraps them. There is no "
            "ensemble population at a frame to classify against: one history "
            "revisits a frame many times, at a different population each time. "
            "**No population was correlated by row number across histories, and "
            "nothing was averaged before classification** — two histories can "
            "exchange character in opposite directions at the same frame, and "
            "their mean would show nothing.",
            "",
        ]
    out += _table(
        ["history", "NAMDTINI", "source", "steps", "first frame", "last frame",
         "distinct frames", "events"],
        [
            [r["file"], r["NAMDTINI"], r["NAMDTINI_source"], r["n_time_points"],
             r["first_frame"], r["last_frame"], r["distinct_frames_visited"],
             r["n_events"]]
            for r in aggregate["per_history"]
        ],
    )
    labels = sorted(
        {label for r in aggregate["per_history"] for label in r["by_classification"]}
    )
    if labels:
        out += ["Classifications per history, and only then the sum:", ""]
        out += _table(
            ["classification", *[r["file"] for r in aggregate["per_history"]], "total"],
            [
                [
                    label,
                    *[r["by_classification"].get(label, 0)
                      for r in aggregate["per_history"]],
                    aggregate["totals_by_classification"].get(label, 0),
                ]
                for label in labels
            ],
        )

    comparison = payload.get("early_vs_late_events")
    if comparison:
        out += [
            "### Early against late",
            "",
            f"`{comparison['early_window']}` against `{comparison['late_window']}`, "
            "on the same per-history classifications.",
            "",
        ]
        out += _table(
            ["classification", "early", "late", "early − late"],
            [[r["classification"], r["early"], r["late"], r["difference"]]
             for r in comparison["rows"]],
        )
        out += [
            f"**{comparison['early_total']}** early against "
            f"**{comparison['late_total']}** late. "
            + comparison["note"][:1].upper() + comparison["note"][1:] + ".",
            "",
        ]
    elif aggregate.get("totals_by_window"):
        out += [
            "No late window was supplied, so no early-vs-late comparison was "
            "made. No manuscript fit window is encoded in this repository, and "
            "one is not invented here.",
            "",
        ]
    return out


def render(payload: Dict[str, Any], events: Sequence[Any]) -> str:
    counts = Counter(e.classification for e in events)
    swaps = [e for e in events if e.character_swap]
    small_gap_swaps = [e for e in swaps if e.small_gap]
    strong_nac_swaps = [e for e in swaps if e.strong_nac]
    # Only a swap whose population moved *the way the swap did* supports a
    # transfer reading.  A magnitude alone is co-occurrence.
    moved = [
        e for e in swaps
        if e.classification == "character_swap_with_fragment_population_change"
    ]
    unrelated = [
        e for e in swaps
        if e.classification == "character_swap_with_unrelated_population_change"
    ]
    undetermined = [
        e for e in swaps
        if e.classification == "character_swap_with_undetermined_direction"
    ]

    out: List[str] = [
        "# Adiabatic character events",
        "",
        "Gap, nonadiabatic coupling and frame-resolved fragment character, "
        "synchronized on the resolved MD frame.",
        "",
        "## What these events are, and are not",
        "",
        "| term | meaning |",
        "| --- | --- |",
        "| adiabatic state population | `P_i = rho_ii` — which eigenstate is occupied |",
        "| adiabatic character exchange | the fragment composition of a **fixed band index** changing |",
        "| nonadiabatic hop | population moving between adiabatic states; lives in SHPROP, not in a PROCAR |",
        "| fragment population | `sum_i P_i w_ig` — closer to a diabatic reading, but **not** a diabatization |",
        "| true diabatic transformation | **not performed anywhere in this package** |",
        "",
        "> Staying on one adiabatic state through an avoided crossing **changes** "
        "the fragment identity: no hop, but the charge moved. Hopping between two "
        "adiabatic states at a crossing can **preserve** the fragment identity: a "
        "hop, but the charge did not move. **A character swap is not a surface "
        "hop, and neither is automatically charge transfer.**",
        "",
        "## Counts",
        "",
    ]
    out += _table(
        ["quantity", "count"],
        [
            ["frames examined", len(set(e.frame for e in events)) or 0],
            ["flagged events", payload.get("n_events", len(events))],
            ["dominant-character exchanges", len(swaps)],
            ["…that coincide with a small gap", len(small_gap_swaps)],
            ["…that coincide with a strong NAC", len(strong_nac_swaps)],
            ["…where the population moved the same way the swap did", len(moved)],
            ["…where it moved, but not that way", len(unrelated)],
            ["…where it moved and the swap named no single direction", len(undetermined)],
        ],
    )
    out += ["By classification:", ""]
    out += _table(
        ["classification", "count"],
        [[name, count] for name, count in counts.most_common()],
    )

    if swaps and not moved and not (unrelated or undetermined):
        out += [
            "**No character exchange in this run was accompanied by a change in "
            "projection-weighted fragment population.** The band labels moved; the "
            "charge did not follow. These are not charge-transfer events, and must "
            "not be described as such.",
            "",
        ]
    elif swaps and not moved:
        out += [
            "**No character exchange in this run was accompanied by a fragment "
            "population change in the direction the swap implies.** Population "
            "did move at some of these steps, but not in a way that supports "
            "calling any of them transfer. None may be described as a "
            "charge-transfer event.",
            "",
        ]
    elif moved:
        out += [
            f"**{len(moved)} character exchange(s) were accompanied by a fragment "
            "population change in the corresponding direction** — the fragment the "
            "dominance moved to gained what the one it left lost. Those are the "
            f"ones that support a transfer reading; the remaining "
            f"{len(swaps) - len(moved)} are not.",
            "",
        ]
        if unrelated:
            out += [
                f"**{len(unrelated)}** had the population move at the same step "
                "but *not* in the direction the swap implies. That is "
                "co-occurrence, and it must not be reported as transfer.",
                "",
            ]
        if undetermined:
            out += [
                f"**{len(undetermined)}** had the population move where the swap "
                "named no single direction — the two bands exchanged character, "
                "so movement either way would match one of them. From a total "
                "fragment population that cannot be decided, and it is left "
                "undecided rather than counted as transfer.",
                "",
            ]

    out += _per_history_section(payload)

    thresholds = payload.get("thresholds", {})
    out += ["## Thresholds", "", "Every cutoff that turned a metric into an event:", ""]
    out += _table(
        ["threshold", "value"],
        [[k, v] for k, v in (thresholds.get("values") or {}).items()],
    )
    sweep = payload.get("sensitivity", {})
    if sweep.get("rows"):
        out += ["Event counts when every threshold is scaled:", ""]
        out += _table(
            ["factor", "total events"],
            [[r["factor"], r["total_events"]] for r in sweep["rows"]],
        )
        spread = sweep.get("relative_spread")
        out += [
            f"Relative spread {spread:.2f}. "
            + (
                "The counts move substantially across this range, so they are "
                "counts the threshold chose as much as the trajectory. Use the raw "
                "metrics in `crossing_metrics.csv` rather than these totals."
                if spread is not None and spread > 0.3
                else "The counts are stable across this range."
            ),
            "",
        ]

    nac = payload.get("nac_interpretation", {})
    if nac.get("max_abs_nac_ev") is not None:
        out += ["## Coupling magnitudes", ""]
        rows = [["largest |NAC|", f"{nac['max_abs_nac_ev']:.4g} eV"]]
        if nac.get("hbar_over_dt_ev"):
            rows.append(["ħ/dt", f"{nac['hbar_over_dt_ev']:.4g} eV"])
            rows.append(["fraction of ħ/dt", f"{nac.get('fraction_of_hbar_over_dt', 0):.3g}"])
        out += _table(["quantity", "value"], rows)
        out += [nac.get("interpretation", ""), ""]
        if nac.get("repeated_magnitude_note"):
            out += [nac["repeated_magnitude_note"] + ".", ""]

    if swaps:
        out += ["## The exchanges themselves", ""]
        out += _table(
            ["frame", "bands", "gap (eV)", "|NAC| (eV)", "band i before → after",
             "classification"],
            [
                [
                    e.frame, f"{e.band_i}/{e.band_j}",
                    None if e.gap_ev != e.gap_ev else round(e.gap_ev, 5),
                    None if e.nac_ev is None else round(e.nac_ev, 5),
                    f"{e.dominant_i_before} → {e.dominant_i_after}",
                    e.classification,
                ]
                for e in swaps[:40]
            ],
        )
        if len(swaps) > 40:
            out += [f"_{len(swaps) - 40} further exchanges in `crossing_events.csv`._", ""]

    out += [
        "## Limitations",
        "",
        "- Synchronization is on the resolved MD frame. A correlation by SHPROP row "
        "index would be wrong wherever histories start at different `NAMDTINI` or "
        "the trajectory wraps cyclically, and would still produce plausible numbers.",
        "- The fragment weights are projection-weighted diagonal quantities: SHPROP "
        "records no coherences and a PROCAR no cross-band projections, so the "
        "off-diagonal terms of a true diabatic picture are absent from the inputs.",
        "- A coupling approaching ħ/dt is numerically pathological, not a giant "
        "physical matrix element, and a magnitude repeated exactly is an upstream "
        "policy rather than dynamics.",
        "- Event counts depend on the thresholds above; the raw metrics are exported "
        "so any other cutoff can be applied without re-running the analysis.",
        "",
    ]
    return "\n".join(out).rstrip() + "\n"


def write(path, payload: Dict[str, Any], events: Sequence[Any]) -> Optional[Path]:
    path = Path(path)
    path.write_text(render(payload, events), encoding="utf-8")
    return path
