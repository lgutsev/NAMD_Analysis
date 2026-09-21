"""``crossing_summary.md`` -- adiabatic character events, stated for a referee.

The whole value of this file is in what it refuses to say, and there are three
states it must keep apart rather than two.

**Not evaluated.** A configuration-level scan reads
``projection_character.csv``, EIGTXT and NATXT and has **no SHPROP populations
at all**. Such a run cannot find that nothing moved; it can only report that
the question was never asked.

**Evaluated, and ``P_g`` did not move.** A measured absence, about the
projected occupied density and about nothing else.

**Evaluated, and ``P_g`` moved.** The occupied density shifted between
fragments. This holds *whichever* term of the symmetric split carries the
change: an occupied adiabatic state turning from BCF-like to PCBM-like at fixed
population has moved its density between the fragments in real space, and
``P_g`` sums over the whole basis so no relabelling can move it. **A
character-driven change is never reported here as "no charge moved."** What the
split cannot do is name the microscopic process, and it is described as a
description rather than as a branching fraction.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .crossings import NO_PROJECTED_CHANGE, PROJECTED_CHANGE


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

    changed = aggregate["totals_by_classification"].get(PROJECTED_CHANGE, 0)
    unchanged = aggregate["totals_by_classification"].get(NO_PROJECTED_CHANGE, 0)
    descriptors = aggregate.get("totals_by_decomposition_descriptor") or {}
    if changed or unchanged:
        out += [
            f"Across the histories, **{changed}** character exchange(s) moved the "
            f"projected fragment population and **{unchanged}** did not. Where "
            "`P_g` moved, the occupied density shifted between fragments — "
            "`P_g` sums over every band of the basis, so no re-ordering of "
            "labels can move it.",
            "",
        ]
    if descriptors:
        out += ["How the symmetric split *describes* the changes:", ""]
        out += _table(
            ["description", "count"],
            sorted(descriptors.items(), key=lambda kv: -kv[1]),
        )
        if descriptors.get("character_dominated"):
            out += [
                f"> **{descriptors['character_dominated']}** are carried mainly "
                "by character evolution: an occupied state's own composition "
                "changed while its population did not. That still **moves its "
                "density between the fragments in real space** — it is the "
                "adiabatic passage named at the top of this file, and it **must "
                "not be reported as \"no charge moved\"** or as a relabelling. "
                "What it does not establish is a nonadiabatic *hop*.",
                "",
            ]
        out += [
            "> The split is **exact bookkeeping and one of infinitely many exact "
            "splits**. These are descriptions of that convention, not physical "
            "branching fractions and not mechanisms, and no hop record is an "
            "input here.",
            "",
        ]

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
    # Three cases, on one observable. Whether P_g moved is the whole test; the
    # symmetric split only describes a change already established.
    not_evaluated = [e for e in swaps if not e.population_evaluated]
    evaluated = [e for e in swaps if e.population_evaluated]
    moved = [e for e in swaps if e.classification == PROJECTED_CHANGE]
    unchanged = [e for e in swaps if e.classification == NO_PROJECTED_CHANGE]
    by_descriptor = Counter(
        e.decomposition_descriptor for e in moved if e.decomposition_descriptor
    )
    direction_matches = sum(
        1 for e in moved if e.swap_direction_matches_projected_change is True
    )
    direction_differs = sum(
        1 for e in moved if e.swap_direction_matches_projected_change is False
    )

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
            ["…whose projected fragment population was **not evaluated**",
             len(not_evaluated)],
            ["…whose projected fragment population **was** evaluated", len(evaluated)],
            ["…where `P_g` moved", len(moved)],
            ["…where `P_g` was evaluated and did not move", len(unchanged)],
        ],
    )
    out += ["By classification:", ""]
    out += _table(
        ["classification", "count"],
        [[name, count] for name, count in counts.most_common()],
    )

    if not_evaluated:
        # Said first, and said before any sentence about what did or did not
        # move, so it cannot be read as a qualifier on a finding of absence.
        out += [
            f"**Projected fragment population was NOT EVALUATED for "
            f"{len(not_evaluated)} of the {len(swaps)} character exchange(s) "
            "below.**"
            + (
                " No SHPROP population was supplied with this scan, so for those "
                "exchanges the analysis has no information about fragment "
                "population at all."
                if not evaluated
                else " No fragment population was available at those steps."
            ),
            "",
            "> This is **not** a finding that no charge moved. A "
            "configuration-level crossing scan reads `projection_character.csv`, "
            "`EIGTXT` and `NATXT`; none of them carries a SHPROP population, so "
            "the question was never asked. Those exchanges are classified "
            "`character_swap_population_not_evaluated`, which must not be "
            "reported as, summarized as, or counted with "
            "`character_swap_without_projected_fragment_change`. **Neither a "
            "change nor its absence may be claimed for them.** Rerun with "
            "`--shprop` and `--state-map` to evaluate `P_g` per history.",
            "",
        ]

    if evaluated:
        out += [
            "### What the projected fragment population did",
            "",
            "`P_g = sum_i P_i w_ig` sums over **every** band of the basis, so no "
            "re-ordering of band labels can move it. A change in it is a change "
            "in where the occupied density sits.",
            "",
        ]

    if evaluated and not moved:
        out += [
            f"**Of the {len(evaluated)} character exchange(s) whose projected "
            "fragment population WAS evaluated, none moved `P_g` beyond "
            "tolerance.** The projected occupied density stayed where it was "
            "across these steps. That is an observation about `P_g`; it is not "
            "by itself a statement about nonadiabatic hops, for which no hop "
            "record is an input here.",
            "",
        ]
    elif moved:
        out += [
            f"**{len(moved)} character exchange(s) moved the projected fragment "
            f"population**, and {len(unchanged)} did not. Where `P_g` moved, the "
            "occupied density shifted between fragments. **This holds whichever "
            "bookkeeping term carries the change.**",
            "",
        ]
        if by_descriptor:
            out += ["How the symmetric split *describes* those changes:", ""]
            out += _table(
                ["description", "count"],
                [[name, count] for name, count in by_descriptor.most_common()],
            )
        if by_descriptor.get("character_dominated"):
            out += [
                f"> **{by_descriptor['character_dominated']}** of them are "
                "carried mainly by character evolution: the occupied state's own "
                "composition changed while its population did not. **That is not "
                "a relabelling and it must not be reported as \"no charge "
                "moved\".** An occupied adiabatic state turning from BCF-like to "
                "PCBM-like moves its density between the fragments in real space "
                "— the adiabatic passage named at the top of this file. What it "
                "does not establish is a nonadiabatic *hop*.",
                "",
            ]
        out += [
            "> The split into `occupation_redistribution` and "
            "`character_evolution` is **exact bookkeeping and one of infinitely "
            "many exact splits**. The descriptions above are descriptions of "
            "that convention, not physical branching fractions, and neither term "
            "names a microscopic mechanism. No surface-hopping record is an "
            "input here, so no statement about hops, hop counts or hopping rates "
            "follows from any of them.",
            "",
        ]
        if direction_matches or direction_differs:
            out += [
                f"Of the {len(moved)}, **{direction_matches}** moved `P_g` in the "
                f"direction the dominance swap points and **{direction_differs}** "
                "did not"
                + (
                    "; the remaining "
                    f"{len(moved) - direction_matches - direction_differs} name no "
                    "single direction — the usual two-state exchange, where "
                    "movement either way would match one band"
                    if len(moved) > direction_matches + direction_differs
                    else ""
                )
                + ". This is a description of each step, not evidence that the "
                "swap caused the change or that it did not — a swap and a "
                "population change can co-occur without either driving the other.",
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
             "ΔP_g", "described as", "classification"],
            [
                [
                    e.frame, f"{e.band_i}/{e.band_j}",
                    None if e.gap_ev != e.gap_ev else round(e.gap_ev, 5),
                    None if e.nac_ev is None else round(e.nac_ev, 5),
                    f"{e.dominant_i_before} → {e.dominant_i_after}",
                    (
                        "**not evaluated**" if not e.population_evaluated
                        else f"{e.fragment_population_change:.5g}"
                        + (f" ({e.dominant_fragment})" if e.dominant_fragment else "")
                    ),
                    e.decomposition_descriptor,
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
        "- Where no SHPROP population was supplied, `P_g` was **not evaluated**, "
        "and no statement about it — in either direction — is made for those "
        "events. Absence of evidence was not recorded as evidence of absence.",
        "- `occupation_redistribution` and `character_evolution` are an **exact "
        "bookkeeping split**, one of infinitely many, and the "
        "`decomposition_descriptor` column describes that convention. Neither "
        "term is a physical branching fraction or a mechanism, and a change "
        "described as `character_dominated` moved the projected occupied density "
        "exactly as much as an `occupation_dominated` one did.",
        "- No hop record is an input. Nothing here counts hops, and no statement "
        "about hopping rates follows from any classification or descriptor above.",
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
