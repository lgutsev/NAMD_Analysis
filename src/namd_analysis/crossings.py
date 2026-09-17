"""Adiabatic gap, nonadiabatic coupling and fragment character, synchronized.

Five things are routinely conflated, and this module keeps them apart because
the manuscript's argument depends on the distinction:

**adiabatic state population**
    :math:`P_i(t) = \\rho_{ii}(t)`, what SHPROP records. Which eigenstate the
    carrier occupies.

**adiabatic-state character exchange**
    the fragment composition :math:`w_{ig}` of a *fixed* band index changing
    with time. The band did not move; the orbital it labels did.

**nonadiabatic hop between adiabatic states**
    population moving from :math:`i` to :math:`j`. Lives in the SHPROP
    populations, not in a PROCAR.

**fragment population**
    :math:`\\sum_i P_i w_{ig}`, which is what ``character-populations``
    reports. Closer to a diabatic reading than a band-resolved one.

**a true diabatic transformation**
    a unitary that removes the derivative coupling. This package does **not**
    perform one, and nothing here should be called diabatic without that
    qualification.

The two failure modes worth stating plainly, because they run opposite ways:

* Staying on one adiabatic state through an avoided crossing **changes** the
  fragment identity. No hop occurred; the physical charge moved.
* Hopping between two adiabatic states at a crossing can **preserve** the
  fragment identity. A hop occurred; the physical charge did not move.

So a character swap is not a surface hop, and a surface hop is not charge
transfer. An event here is only called transfer when the fragment population
actually changes in the corresponding direction.

Two-state picture behind all of it.  Near an avoided crossing between diabatic
fragment states :math:`|A\\rangle` and :math:`|B\\rangle`,

.. math::

    |\\phi_1\\rangle = \\cos\\theta\\,|A\\rangle + \\sin\\theta\\,|B\\rangle \\\\
    |\\phi_2\\rangle = -\\sin\\theta\\,|A\\rangle + \\cos\\theta\\,|B\\rangle

with :math:`\\tan 2\\theta = 2V/(E_A - E_B)`.  Far from the crossing
:math:`\\theta \\to 0` and the adiabatic states are the fragment states; at the
crossing :math:`\\theta \\to \\pi/4` and each is an even mixture.  A PROCAR
projection onto the fragment's atoms measures approximately
:math:`\\cos^2\\theta` and :math:`\\sin^2\\theta`, so the character curves this
module reads track :math:`\\theta(t)` sweeping through the crossing.  That is
why a fixed band label obscures physical transfer: the label is constant while
:math:`\\theta` is not.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .units import hbar_over_dt_ev

#: Event-flag defaults.  Every one is configurable and every one is reported in
#: provenance: a cutoff that decides whether something is called an event must
#: never be implicit.
DEFAULT_THRESHOLDS = {
    #: eV.  Below this the two adiabatic states are "close".
    "gap_ev": 0.05,
    #: eV.  Above this the coupling is "enhanced" relative to the trajectory.
    "nac_ev": 0.01,
    #: Absolute change in normalized fragment weight that counts as exchange.
    "character_change": 0.25,
    #: Weight above which a band is called dominated by one fragment.
    "dominance": 0.6,
}

#: Multipliers applied to every threshold for the sensitivity sweep.  Event
#: counts that move sharply across these are counts the threshold chose.
SENSITIVITY_FACTORS = (0.5, 0.75, 1.0, 1.5, 2.0)


class CrossingError(ValueError):
    """Raised when a crossing analysis would have to assume something."""


# --------------------------------------------------------------------------
# Inputs, joined on the resolved frame -- never on a row number
# --------------------------------------------------------------------------


@dataclass
class CharacterSeries:
    """Fragment weights per (frame, band), read from ``projection_character.csv``.

    Frames are the join key throughout.  A SHPROP row index is meaningless
    across histories with different ``NAMDTINI`` and a cyclic mapping: row 500
    of one history and row 500 of another are different MD frames.  The frame
    number is what the alignment already resolved, so it is what everything is
    correlated on.
    """

    frames: np.ndarray
    bands: np.ndarray
    groups: List[str]
    weights: np.ndarray  # (nframe, nband, ngroup)

    def frame_index(self) -> Dict[int, int]:
        return {int(f): i for i, f in enumerate(self.frames)}

    def band_index(self) -> Dict[int, int]:
        return {int(b): i for i, b in enumerate(self.bands)}


def read_projection_character(path) -> CharacterSeries:
    """Load the long-form character table the character run already wrote.

    Reusing that output is the point: the PROCARs were parsed once, and this
    analysis must not open a hundred gigabytes again to ask a different
    question of the same numbers.
    """
    path = Path(path)
    rows: Dict[Tuple[int, int], Dict[str, float]] = {}
    groups: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"frame", "band", "group", "normalized_weight"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise CrossingError(
                f"{path}: projection character table lacks {sorted(missing)}. "
                "Point this at the projection_character.csv a character run wrote."
            )
        for record in reader:
            frame = int(float(record["frame"]))
            band = int(float(record["band"]))
            group = record["group"]
            if group not in groups:
                groups.append(group)
            rows.setdefault((frame, band), {})[group] = float(record["normalized_weight"])
    if not rows:
        raise CrossingError(f"{path}: no rows")

    frames = sorted({frame for frame, _ in rows})
    bands = sorted({band for _, band in rows})
    weights = np.full((len(frames), len(bands), len(groups)), np.nan)
    frame_at = {f: i for i, f in enumerate(frames)}
    band_at = {b: i for i, b in enumerate(bands)}
    for (frame, band), by_group in rows.items():
        for gi, group in enumerate(groups):
            if group in by_group:
                weights[frame_at[frame], band_at[band], gi] = by_group[group]
    if np.isnan(weights).any():
        missing_count = int(np.isnan(weights).sum())
        raise CrossingError(
            f"{path}: {missing_count} (frame, band, group) cell(s) are absent. "
            "The table must be rectangular; nothing here fills a gap."
        )
    return CharacterSeries(
        frames=np.asarray(frames, dtype=int),
        bands=np.asarray(bands, dtype=int),
        groups=groups,
        weights=weights,
    )


@dataclass
class CouplingSeries:
    """Adiabatic gaps and |NAC| per frame for one basis."""

    frames: np.ndarray
    energies: np.ndarray  # (nframe, nstate), eV
    nac: Optional[np.ndarray]  # (nframe, nstate, nstate), eV
    nac_unit: str
    dt_fs: Optional[float]
    ceiling_ev: Optional[float]

    def gap(self, i: int, j: int) -> np.ndarray:
        return np.abs(self.energies[:, i] - self.energies[:, j])


def load_couplings(
    eigtxt,
    natxt=None,
    frames: Optional[Sequence[int]] = None,
    nac_unit: str = "eV",
    dt_fs: Optional[float] = None,
) -> CouplingSeries:
    """Read EIGTXT and optional NATXT, indexed by MD frame.

    ``frames`` names the MD frame each row corresponds to.  Without it the rows
    are assumed to be frames ``1..nframes`` in order, which is the Hefei-NAMD
    convention; that assumption is recorded rather than hidden.
    """
    from .io.hefei import read_eigtxt, read_natxt
    from .units import nac_to_mev

    energies = read_eigtxt(eigtxt)
    nframes, nstates = energies.shape
    if frames is None:
        frame_numbers = np.arange(1, nframes + 1, dtype=int)
    else:
        frame_numbers = np.asarray(list(frames), dtype=int)
        if frame_numbers.size != nframes:
            raise CrossingError(
                f"{eigtxt}: {nframes} rows but {frame_numbers.size} frame numbers "
                "were supplied; they must correspond one to one"
            )

    coupling = None
    if natxt is not None:
        raw = read_natxt(natxt, nstates=nstates)
        if raw.shape[0] != nframes:
            raise CrossingError(
                f"{natxt}: {raw.shape[0]} frames against {nframes} in {eigtxt}; "
                "the two must describe the same trajectory"
            )
        coupling = np.abs(nac_to_mev(np.abs(raw), nac_unit, dt_fs)) / 1000.0

    ceiling = hbar_over_dt_ev(dt_fs) if dt_fs else None
    return CouplingSeries(
        frames=frame_numbers,
        energies=energies,
        nac=coupling,
        nac_unit=nac_unit,
        dt_fs=dt_fs,
        ceiling_ev=ceiling,
    )


def nac_ceiling_note(series: CouplingSeries) -> Dict[str, Any]:
    """The established reading of couplings near hbar/dt, preserved.

    A magnitude shared exactly by many samples did not come out of the
    dynamics: some upstream step put it there, and repeated values near
    0.6 eV are consistent with an intentionally imposed NAC safety ceiling.
    Couplings approaching hbar/dt are treated as numerically pathological --
    the finite-difference evaluation has broken down over one step -- and never
    as a measurement of a giant physical matrix element.
    """
    if series.nac is None:
        return {"status": "no_nac_supplied"}
    finite = series.nac[np.isfinite(series.nac)]
    if finite.size == 0:
        return {"status": "no_finite_nac"}
    maximum = float(np.max(finite))
    payload: Dict[str, Any] = {
        "max_abs_nac_ev": maximum,
        "hbar_over_dt_ev": series.ceiling_ev,
        "interpretation": (
            "a coupling approaching hbar/dt is numerically pathological -- the "
            "finite-difference evaluation of the NAC has broken down over one "
            "step -- and is NOT a measurement of an arbitrarily large physical "
            "matrix element. Nothing here is filtered, rescaled or rejected on "
            "account of it"
        ),
    }
    if series.ceiling_ev:
        payload["fraction_of_hbar_over_dt"] = maximum / series.ceiling_ev
    # An exactly repeated magnitude is an upstream policy, not a coincidence.
    values, counts = np.unique(np.round(finite[finite > 0], 12), return_counts=True)
    if counts.size and counts.max() > max(10, 0.001 * finite.size):
        repeated = float(values[np.argmax(counts)])
        payload["repeated_magnitude_ev"] = repeated
        payload["repeated_magnitude_count"] = int(counts.max())
        payload["repeated_magnitude_note"] = (
            f"{int(counts.max())} samples share the magnitude {repeated:.6g} eV "
            "exactly. A value repeated exactly did not come out of the dynamics; "
            "some upstream step put it there. This is consistent with an "
            "intentionally imposed NAC safety ceiling and is NOT described as "
            "accidental clipping or as a defect"
        )
    return payload


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------


@dataclass
class CrossingEvent:
    """One frame where a pair of states was close, coupled, or exchanged character."""

    frame: int
    band_i: int
    band_j: int
    gap_ev: float
    nac_ev: Optional[float]
    character_change_i: float
    character_change_j: float
    dominant_i_before: str
    dominant_i_after: str
    dominant_j_before: str
    dominant_j_after: str
    small_gap: bool
    strong_nac: bool
    character_swap: bool
    fragment_population_change: Optional[float]
    classification: str
    note: str
    #: The exact split of ``fragment_population_change`` into occupation moving
    #: at fixed character, and character moving at fixed occupation. Only a
    #: single history carries the per-band populations this needs, so both are
    #: ``None`` on the ensemble path. See :class:`HistoryPopulation`.
    population_driven_change: Optional[float] = None
    character_driven_change: Optional[float] = None

    def as_row(self) -> List[Any]:
        return [
            self.frame, self.band_i, self.band_j, self.gap_ev, self.nac_ev,
            self.character_change_i, self.character_change_j,
            self.dominant_i_before, self.dominant_i_after,
            self.dominant_j_before, self.dominant_j_after,
            self.small_gap, self.strong_nac, self.character_swap,
            self.fragment_population_change,
            self.population_driven_change, self.character_driven_change,
            self.classification,
        ]


EVENT_HEADER = [
    "frame", "band_i", "band_j", "gap_ev", "nac_ev",
    "character_change_i", "character_change_j",
    "dominant_i_before", "dominant_i_after",
    "dominant_j_before", "dominant_j_after",
    "small_gap", "strong_nac", "character_swap",
    "fragment_population_change",
    "population_driven_change", "character_driven_change",
    "classification",
]


def _swap_moves(
    dom_i_before: str, dom_i_after: str, dom_j_before: str, dom_j_after: str
) -> List[Tuple[str, str]]:
    """The ``(from, to)`` fragment moves a dominance swap implies."""
    moves = []
    if dom_i_before != dom_i_after:
        moves.append((dom_i_before, dom_i_after))
    if dom_j_before != dom_j_after:
        moves.append((dom_j_before, dom_j_after))
    return moves


CONFOUND_NOTE = (
    "a change in total fragment population cannot say whether charge moved: "
    "P_g = sum_i P_i w_ig moves when the *weights* move, so a band-index swap "
    "shifts it mechanically, at fixed occupation. Only the population-driven "
    "part of the change can be tested against the swap direction, and that "
    "needs the per-band populations, which only a single SHPROP history has"
)


def _direction_agrees(
    moves: Sequence[Tuple[str, str]],
    delta: Optional[Dict[str, float]],
    tolerance: float,
) -> Optional[bool]:
    """Did *occupation* move the way the dominance swap did?

    ``delta`` must be the **population-driven** part of the change,
    ``sum_i [P_i(t) - P_i(t-1)] w_ig[f(t)]``, not the total.  The total is
    confounded: see :data:`CONFOUND_NOTE`.

    Agreement means the fragment the dominance moved *to* gained occupation
    while the one it left lost it.  A magnitude alone cannot say this -- a
    population can move at the same step as a swap and have nothing to do with
    it, and calling that transfer would read co-occurrence as cause.

    ``None`` where the question is not well posed, and the caller must not read
    that as either answer:

    * no population-driven change was supplied (the caller has only a total);
    * the character changed without any band's dominant fragment moving, so
      there is no swap direction at all;
    * the swap names **no single direction** -- the usual two-state avoided
      crossing, where one band goes BCF->PCBM while the other goes PCBM->BCF.
      The pair exchanged character, so occupation moving either way would
      "agree" with one of the two bands.  That case cannot be decided, and is
      reported as undecided rather than resolved in favour of transfer.
    """
    if delta is None or not moves:
        return None
    sources = {source for source, _ in moves}
    targets = {target for _, target in moves}
    if len(targets) != 1 or (sources & targets):
        return None
    target = next(iter(targets))
    if delta.get(target, 0.0) <= tolerance:
        return False
    return all(delta.get(source, 0.0) < -tolerance for source in sources)


def _classify(
    small_gap: bool,
    strong_nac: bool,
    character_swap: bool,
    population_change: Optional[float],
    population_tolerance: float,
    direction_agrees: Optional[bool] = None,
    population_driven_change: Optional[float] = None,
) -> Tuple[str, str]:
    """Name what happened, and refuse to name what the data do not show.

    ``charge transfer`` is reserved for the case where the *fragment*
    population actually moved.  A character swap on its own is a relabelling of
    which orbital a band index points at; a population redistribution on its
    own moved occupation between adiabatic states that may share a fragment.
    """
    moved = (
        population_change is not None
        and abs(population_change) > population_tolerance
    )
    parts = []
    if small_gap:
        parts.append("small gap")
    if strong_nac:
        parts.append("strong NAC")
    if character_swap:
        parts.append("character swap")
    if moved:
        parts.append("fragment population change")

    if character_swap and not moved:
        label = "character_swap_without_fragment_transfer"
        note = (
            "the dominant fragment of a band index changed, but the "
            "projection-weighted fragment population did not follow. The band "
            "label now points at a different orbital; no charge moved between "
            "fragments. This is NOT a charge-transfer event"
        )
    elif (
        character_swap
        and moved
        and population_driven_change is not None
        and abs(population_driven_change) <= population_tolerance
    ):
        # The total moved, but every bit of it was the weights moving.
        label = "character_swap_without_fragment_transfer"
        note = (
            "the dominant fragment of a band index changed and the total "
            "fragment population moved with it, but the occupation did not: "
            "the whole change is character-driven, sum_i P_i dw_ig at fixed "
            "P_i. The band label now points at a different orbital; no charge "
            "moved between fragments. This is NOT a charge-transfer event"
        )
    elif character_swap and moved and direction_agrees is False:
        label = "character_swap_with_unrelated_population_change"
        note = (
            "the character changed and the fragment population moved, but NOT "
            "in the direction the dominance swap implies: the fragment the "
            "dominance moved to did not gain what the one it left lost. The two "
            "co-occurred at this step, which is not evidence of transfer "
            "between the fragments the swap names"
        )
    elif character_swap and moved and direction_agrees is None:
        label = "character_swap_with_undetermined_direction"
        note = (
            "the character changed and the fragment population moved, but the "
            "direction could not be tested: either no band's dominant fragment "
            "moved, or the two bands exchanged character so movement either "
            "way would match one of them, or only a total population was "
            "available -- " + CONFOUND_NOTE + ". This is co-occurrence; it is "
            "not by itself transfer"
        )
    elif character_swap and moved:
        label = "character_swap_with_fragment_population_change"
        note = (
            "the band's dominant fragment changed AND the fragment population "
            "moved in the corresponding direction -- the fragment the dominance "
            "moved to gained population, and the one it left lost it. This is "
            "the combination that supports calling it transfer; the character "
            "alone would not"
        )
    elif moved and not character_swap:
        label = "fragment_population_change_without_character_swap"
        note = (
            "occupation moved between fragments while every band kept its "
            "dominant character. Population was redistributed among adiabatic "
            "states whose characters did not change"
        )
    elif small_gap and strong_nac:
        label = "close_and_coupled_without_character_swap"
        note = (
            "the states were close and strongly coupled, but neither the "
            "character nor the fragment population changed at this frame. A "
            "coupling is an opportunity, not an event"
        )
    else:
        label = "flagged_metric_only"
        note = (
            "one or more raw metrics crossed its threshold without any change "
            "of character or fragment population"
        )
    if parts:
        note = f"{' + '.join(parts)}: {note}"
    return label, note


def detect_events(
    character: CharacterSeries,
    couplings: Optional[CouplingSeries],
    band_pairs: Sequence[Tuple[int, int]],
    thresholds: Optional[Dict[str, float]] = None,
    fragment_population: Optional[Dict[int, Dict[str, float]]] = None,
    population_tolerance: float = 1.0e-3,
) -> List[CrossingEvent]:
    """Frames where a pair of bands was close, coupled, or exchanged character.

    Every raw metric is carried on the event alongside the boolean flag it
    produced, so a reader can re-threshold without re-running anything.
    """
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    band_at = character.band_index()
    events: List[CrossingEvent] = []

    gap_lookup: Dict[int, Dict[Tuple[int, int], float]] = {}
    nac_lookup: Dict[int, Dict[Tuple[int, int], float]] = {}
    if couplings is not None:
        state_of = {int(b): k for k, b in enumerate(character.bands)}
        frame_row = {int(f): r for r, f in enumerate(couplings.frames)}
        for frame in character.frames:
            row = frame_row.get(int(frame))
            if row is None:
                continue
            gap_lookup[int(frame)] = {}
            nac_lookup[int(frame)] = {}
            for bi, bj in band_pairs:
                i, j = state_of.get(bi), state_of.get(bj)
                if i is None or j is None:
                    continue
                if i >= couplings.energies.shape[1] or j >= couplings.energies.shape[1]:
                    continue
                gap_lookup[int(frame)][(bi, bj)] = float(
                    abs(couplings.energies[row, i] - couplings.energies[row, j])
                )
                if couplings.nac is not None:
                    nac_lookup[int(frame)][(bi, bj)] = float(couplings.nac[row, i, j])

    for index in range(1, len(character.frames)):
        frame = int(character.frames[index])
        previous = int(character.frames[index - 1])
        for bi, bj in band_pairs:
            if bi not in band_at or bj not in band_at:
                continue
            wi_before = character.weights[index - 1, band_at[bi]]
            wi_after = character.weights[index, band_at[bi]]
            wj_before = character.weights[index - 1, band_at[bj]]
            wj_after = character.weights[index, band_at[bj]]

            change_i = float(np.max(np.abs(wi_after - wi_before)))
            change_j = float(np.max(np.abs(wj_after - wj_before)))
            dom_i_before = character.groups[int(np.argmax(wi_before))]
            dom_i_after = character.groups[int(np.argmax(wi_after))]
            dom_j_before = character.groups[int(np.argmax(wj_before))]
            dom_j_after = character.groups[int(np.argmax(wj_after))]

            swap = (dom_i_before != dom_i_after) or (dom_j_before != dom_j_after)
            exchanged = max(change_i, change_j) >= thresholds["character_change"]

            gap = gap_lookup.get(frame, {}).get((bi, bj))
            nac = nac_lookup.get(frame, {}).get((bi, bj))
            small_gap = gap is not None and gap <= thresholds["gap_ev"]
            strong_nac = nac is not None and nac >= thresholds["nac_ev"]

            if not (swap or exchanged or small_gap or strong_nac):
                continue

            change = None
            before = after = None
            if fragment_population is not None:
                before = fragment_population.get(previous)
                after = fragment_population.get(frame)
                if before and after:
                    change = float(
                        max(
                            abs(after.get(g, 0.0) - before.get(g, 0.0))
                            for g in character.groups
                        )
                    )

            # A caller of detect_events supplies only a *total* fragment
            # population, which cannot be split into its population-driven and
            # character-driven parts, so no direction is claimed here. The
            # per-history path, which has the per-band populations, does the
            # real test.
            label, note = _classify(
                small_gap, strong_nac, swap or exchanged, change, population_tolerance
            )
            events.append(
                CrossingEvent(
                    frame=frame,
                    band_i=bi,
                    band_j=bj,
                    gap_ev=float(gap) if gap is not None else float("nan"),
                    nac_ev=float(nac) if nac is not None else None,
                    character_change_i=change_i,
                    character_change_j=change_j,
                    dominant_i_before=dom_i_before,
                    dominant_i_after=dom_i_after,
                    dominant_j_before=dom_j_before,
                    dominant_j_after=dom_j_after,
                    small_gap=small_gap,
                    strong_nac=strong_nac,
                    character_swap=bool(swap or exchanged),
                    fragment_population_change=change,
                    classification=label,
                    note=note,
                )
            )
    return events


def sensitivity(
    character: CharacterSeries,
    couplings: Optional[CouplingSeries],
    band_pairs: Sequence[Tuple[int, int]],
    thresholds: Optional[Dict[str, float]] = None,
    factors: Sequence[float] = SENSITIVITY_FACTORS,
    fragment_population: Optional[Dict[int, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """How the event counts move when every threshold is scaled.

    A count that changes by an order of magnitude across a factor of two in the
    cutoff is a count the cutoff chose, not one the trajectory produced.
    """
    base = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    rows = []
    for factor in factors:
        scaled = {key: value * factor for key, value in base.items()}
        events = detect_events(
            character, couplings, band_pairs, scaled, fragment_population
        )
        counts: Dict[str, int] = {}
        for event in events:
            counts[event.classification] = counts.get(event.classification, 0) + 1
        rows.append(
            {
                "factor": factor,
                "thresholds": scaled,
                "total_events": len(events),
                "by_classification": counts,
            }
        )
    totals = [row["total_events"] for row in rows]
    spread = (max(totals) - min(totals)) / max(1, max(totals))
    return {
        "base_thresholds": base,
        "factors": list(factors),
        "rows": rows,
        "relative_spread": float(spread),
        "note": (
            "event counts across scaled thresholds. A count that moves sharply "
            "here is a count the threshold chose; the raw gap, |NAC| and "
            "character-change metrics are in the event table so any other cutoff "
            "can be applied without re-running the analysis"
        ),
    }


# --------------------------------------------------------------------------
# Per history, along its own trajectory
# --------------------------------------------------------------------------
#
# When histories start at different NAMDTINI they visit different frames at the
# same row, so there is no ensemble "population at frame f": under a cyclic
# mapping one history revisits a frame many times, at different populations
# each time. What *is* well defined is one history's own trajectory -- at step
# t it occupies frame f_r(t) with population P^(r)(t) -- so events are detected
# along that, per history, and only then aggregated.
#
# Averaging first would destroy exactly the thing being measured: two histories
# can exchange character in opposite directions at the same frame, and their
# mean shows nothing.


@dataclass
class HistoryPopulation:
    """One history's fragment population, and what moved it.

    Each step's change splits exactly, ``dP_g = dP_g^pop + dP_g^char`` with

        dP_g^pop  = sum_i [P_i(t) - P_i(t-1)] w_ig[f(t)]
        dP_g^char = sum_i P_i(t-1) [w_ig[f(t)] - w_ig[f(t-1)]]

    The first is occupation moving between states at fixed character; the
    second is the character moving under fixed occupation.  **A band-index swap
    produces the second mechanically**, which is why the total on its own
    cannot say whether charge moved -- and why the direction test runs on
    ``population_driven`` alone.  Row 0 has no previous step and is zero in
    both.
    """

    total: np.ndarray  # (ntime, ngroup)
    population_driven: np.ndarray  # (ntime, ngroup)
    character_driven: np.ndarray  # (ntime, ngroup)
    time_raw: np.ndarray  # (ntime,), the file's own time column


@dataclass
class HistoryEvents:
    """Events found along one history, with the window each fell in."""

    path: Path
    namdtini: int
    namdtini_source: str
    n_time: int
    frames: np.ndarray
    events: List[CrossingEvent]
    window_of_event: List[str]
    fragment_population: np.ndarray  # (ntime, ngroup)
    time_ns: np.ndarray

    def counts(self, window: Optional[str] = None) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for event, which in zip(self.events, self.window_of_event):
            if window is not None and which != window:
                continue
            out[event.classification] = out.get(event.classification, 0) + 1
        return out


def history_fragment_population(
    shprop_path,
    state_map,
    character: CharacterSeries,
    bands: Sequence[int],
    frames: np.ndarray,
    chunk_rows: int = 100_000,
) -> "HistoryPopulation":
    """``P_g(t) = sum_i P_i(t) w_ig[f(t)]`` for one history, streamed.

    The same diagonal contraction ``character-populations`` performs, but kept
    per history rather than averaged, because an event has to be classified on
    the history that produced it.

    Also returns the exact split of each step's change into its
    population-driven and character-driven parts (see
    :class:`HistoryPopulation`), and this history's **own** time column, read
    from the file.  The time axis is not reconstructed from the first and last
    rows: that would assume a uniform grid, and an event would then be assigned
    to a window on the strength of the assumption rather than the data.
    """
    from .io.hefei import iter_shprop_chunks

    band_at = character.band_index()
    missing = [b for b in bands if b not in band_at]
    if missing:
        raise CrossingError(
            f"the character table has no bands {missing}; it must cover the whole "
            "SHPROP basis for a per-history population"
        )
    selected = character.weights[:, [band_at[b] for b in bands], :]
    frame_at = character.frame_index()
    columns = np.asarray(state_map.population_columns, dtype=int)

    out = np.empty((frames.size, len(character.groups)), dtype=float)
    driven_pop = np.zeros_like(out)
    driven_char = np.zeros_like(out)
    times = np.empty(frames.size, dtype=float)
    previous_pops = previous_weights = None
    rows_seen = 0
    for offset, chunk in iter_shprop_chunks(Path(shprop_path), chunk_rows):
        rows = chunk.shape[0]
        if rows_seen + rows > frames.size:
            raise CrossingError(
                f"{shprop_path}: more rows than the {frames.size} the alignment "
                "resolved; the file changed under the analysis"
            )
        slice_frames = frames[offset : offset + rows]
        unknown = sorted({int(f) for f in slice_frames} - set(frame_at))
        if unknown:
            raise CrossingError(
                f"{shprop_path}: rows {offset}..{offset + rows - 1} visit frames "
                f"{unknown[:10]} that the character table does not cover"
            )
        indices = np.asarray([frame_at[int(f)] for f in slice_frames], dtype=int)
        pops = chunk[:, columns]
        weights = selected[indices]
        out[offset : offset + rows] = np.einsum("ts,tsg->tg", pops, weights)
        times[offset : offset + rows] = chunk[:, state_map.time_column]

        # Split each step's change exactly. The first row of the file has no
        # previous step, so it is compared with itself and both parts are zero;
        # at a chunk join the carried-over row makes the split identical to
        # what a single-chunk read would give.
        if previous_pops is None:
            shifted_pops = np.concatenate([pops[:1], pops[:-1]])
            shifted_weights = np.concatenate([weights[:1], weights[:-1]])
        else:
            shifted_pops = np.concatenate([previous_pops[None, :], pops[:-1]])
            shifted_weights = np.concatenate([previous_weights[None, :, :], weights[:-1]])
        driven_pop[offset : offset + rows] = np.einsum(
            "ts,tsg->tg", pops - shifted_pops, weights
        )
        driven_char[offset : offset + rows] = np.einsum(
            "ts,tsg->tg", shifted_pops, weights - shifted_weights
        )
        previous_pops = pops[-1].copy()
        previous_weights = weights[-1].copy()
        rows_seen += rows
    if rows_seen != frames.size:
        raise CrossingError(
            f"{shprop_path}: {rows_seen} rows against {frames.size} resolved frames"
        )
    return HistoryPopulation(
        total=out,
        population_driven=driven_pop,
        character_driven=driven_char,
        time_raw=times,
    )


def detect_history_events(
    shprop_path,
    state_map,
    character: CharacterSeries,
    couplings: Optional[CouplingSeries],
    band_pairs: Sequence[Tuple[int, int]],
    frames: np.ndarray,
    namdtini: int,
    namdtini_source: str,
    thresholds: Optional[Dict[str, float]] = None,
    population_tolerance: float = 1.0e-3,
    windows: Optional[Sequence[Any]] = None,
    chunk_rows: int = 100_000,
) -> HistoryEvents:
    """Walk one history's trajectory and classify what happens along it.

    Consecutive entries are consecutive *time steps of this history*, so the
    frames they occupy are whatever the resolved mapping says -- adjacent,
    wrapped, or repeated. The population difference is between the two steps,
    not between two frames of some ensemble average.

    The time each event is stamped with is this history's own time column, so a
    window assignment never depends on a reconstructed grid.
    """
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    bands = [int(b) for b in character.bands]
    split = history_fragment_population(
        shprop_path, state_map, character, bands, frames, chunk_rows=chunk_rows
    )
    population = split.total
    time_ns = split.time_raw / state_map.to_ns

    band_at = character.band_index()
    frame_at = character.frame_index()
    state_of = {int(b): k for k, b in enumerate(character.bands)}
    coupling_row = (
        {int(f): r for r, f in enumerate(couplings.frames)} if couplings is not None else {}
    )

    events: List[CrossingEvent] = []
    window_of: List[str] = []
    for step in range(1, frames.size):
        before_frame, after_frame = int(frames[step - 1]), int(frames[step])
        if before_frame == after_frame:
            # The trajectory stayed on one electronic frame: the character
            # cannot have changed, whatever the population did.
            continue
        bi_index_before = frame_at[before_frame]
        bi_index_after = frame_at[after_frame]
        change = float(np.max(np.abs(population[step] - population[step - 1])))

        for bi, bj in band_pairs:
            if bi not in band_at or bj not in band_at:
                continue
            wi_before = character.weights[bi_index_before, band_at[bi]]
            wi_after = character.weights[bi_index_after, band_at[bi]]
            wj_before = character.weights[bi_index_before, band_at[bj]]
            wj_after = character.weights[bi_index_after, band_at[bj]]

            change_i = float(np.max(np.abs(wi_after - wi_before)))
            change_j = float(np.max(np.abs(wj_after - wj_before)))
            dom_i_before = character.groups[int(np.argmax(wi_before))]
            dom_i_after = character.groups[int(np.argmax(wi_after))]
            dom_j_before = character.groups[int(np.argmax(wj_before))]
            dom_j_after = character.groups[int(np.argmax(wj_after))]
            swap = (dom_i_before != dom_i_after) or (dom_j_before != dom_j_after)
            exchanged = max(change_i, change_j) >= thresholds["character_change"]

            gap = nac = None
            row = coupling_row.get(after_frame)
            if row is not None and couplings is not None:
                i, j = state_of.get(bi), state_of.get(bj)
                if i is not None and j is not None and max(i, j) < couplings.energies.shape[1]:
                    gap = float(abs(couplings.energies[row, i] - couplings.energies[row, j]))
                    if couplings.nac is not None:
                        nac = float(couplings.nac[row, i, j])
            small_gap = gap is not None and gap <= thresholds["gap_ev"]
            strong_nac = nac is not None and nac >= thresholds["nac_ev"]

            if not (swap or exchanged or small_gap or strong_nac):
                continue
            label, note = _classify(
                small_gap,
                strong_nac,
                swap or exchanged,
                change,
                population_tolerance,
                _direction_agrees(
                    _swap_moves(
                        dom_i_before, dom_i_after, dom_j_before, dom_j_after
                    ),
                    dict(zip(character.groups, split.population_driven[step])),
                    population_tolerance,
                ),
                float(np.max(np.abs(split.population_driven[step]))),
            )
            events.append(
                CrossingEvent(
                    population_driven_change=float(
                        np.max(np.abs(split.population_driven[step]))
                    ),
                    character_driven_change=float(
                        np.max(np.abs(split.character_driven[step]))
                    ),
                    frame=after_frame,
                    band_i=bi,
                    band_j=bj,
                    gap_ev=float(gap) if gap is not None else float("nan"),
                    nac_ev=nac,
                    character_change_i=change_i,
                    character_change_j=change_j,
                    dominant_i_before=dom_i_before,
                    dominant_i_after=dom_i_after,
                    dominant_j_before=dom_j_before,
                    dominant_j_after=dom_j_after,
                    small_gap=small_gap,
                    strong_nac=strong_nac,
                    character_swap=bool(swap or exchanged),
                    fragment_population_change=change,
                    classification=label,
                    note=note,
                )
            )
            window_of.append(_window_name(time_ns[step], windows))

    return HistoryEvents(
        path=Path(shprop_path),
        namdtini=int(namdtini),
        namdtini_source=namdtini_source,
        n_time=int(frames.size),
        frames=frames,
        events=events,
        window_of_event=window_of,
        fragment_population=population,
        time_ns=time_ns,
    )


def _window_name(time_value: float, windows: Optional[Sequence[Any]]) -> str:
    if not windows:
        return "all"
    for window in windows:
        low = -np.inf if window.start_ns is None else window.start_ns
        high = np.inf if window.end_ns is None else window.end_ns
        if low <= time_value <= high:
            return window.name
    return "outside"


def aggregate_histories(
    histories: Sequence[HistoryEvents], window_names: Sequence[str] = ()
) -> Dict[str, Any]:
    """Combine per-history classifications -- after classification, never before.

    The per-history counts are kept alongside the totals: two histories can
    exchange character in opposite directions at the same frame, and a number
    that only ever appears summed would hide that.
    """
    per_history = [
        {
            "file": h.path.name,
            "NAMDTINI": h.namdtini,
            "NAMDTINI_source": h.namdtini_source,
            "n_time_points": h.n_time,
            "distinct_frames_visited": int(np.unique(h.frames).size),
            "first_frame": int(h.frames[0]),
            "last_frame": int(h.frames[-1]),
            "n_events": len(h.events),
            "by_classification": h.counts(),
            "by_window": {name: h.counts(name) for name in window_names} if window_names else {},
        }
        for h in histories
    ]
    totals: Dict[str, int] = {}
    for record in per_history:
        for label, count in record["by_classification"].items():
            totals[label] = totals.get(label, 0) + count

    by_window: Dict[str, Dict[str, int]] = {}
    for name in window_names:
        merged: Dict[str, int] = {}
        for history in histories:
            for label, count in history.counts(name).items():
                merged[label] = merged.get(label, 0) + count
        by_window[name] = merged

    starts = sorted({h.namdtini for h in histories})
    return {
        "n_histories": len(histories),
        "distinct_namdtini": starts,
        "per_history": per_history,
        "totals_by_classification": totals,
        "totals_by_window": by_window,
        "note": (
            "every event was classified on the history that produced it, using "
            "that history's own resolved frame mapping and its own "
            "projection-weighted fragment population, and only then summed. "
            "Histories starting at different NAMDTINI visit different frames at "
            "the same row, so no population was ever correlated by row number "
            "across histories, and nothing was averaged before classification"
        ),
        "why_not_averaged": (
            "two histories can exchange character in opposite directions at the "
            "same frame; their mean shows nothing. The per-history counts are "
            "kept so that cancellation is visible rather than silent"
        ),
    }


def compare_event_windows(
    aggregate: Dict[str, Any], early_name: str, late_name: str
) -> Dict[str, Any]:
    """Early against late, on the same per-history classifications."""
    by_window = aggregate.get("totals_by_window", {})
    early = by_window.get(early_name, {})
    late = by_window.get(late_name, {})
    labels = sorted(set(early) | set(late))
    rows = [
        {
            "classification": label,
            "early": early.get(label, 0),
            "late": late.get(label, 0),
            "difference": early.get(label, 0) - late.get(label, 0),
        }
        for label in labels
    ]
    early_total = sum(early.values())
    late_total = sum(late.values())
    return {
        "early_window": early_name,
        "late_window": late_name,
        "rows": rows,
        "early_total": early_total,
        "late_total": late_total,
        "note": (
            "raw event counts per window. The windows are of different length, so "
            "a larger count in one is not by itself a higher rate; divide by the "
            "window duration before comparing, and remember these are flagged "
            "metric crossings rather than measured transitions"
        ),
    }
