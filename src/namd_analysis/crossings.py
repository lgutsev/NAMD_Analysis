"""Adiabatic gap, nonadiabatic coupling and fragment character, synchronized.

Five things are routinely conflated, and this module keeps them apart because
the manuscript's argument depends on the distinction:

**adiabatic state population**
    :math:`P_i(t) = \\rho_{ii}(t)`, what SHPROP records. Which eigenstate the
    carrier occupies.

**adiabatic-state character exchange**
    the fragment composition :math:`w_{ig}` of a *fixed* band index changing
    with time. The band index did not move; the orbital it labels did. Note
    what this is **not**: it is not a bookkeeping relabelling with nothing
    behind it. If that band is occupied, the change can correspond to a spatial
    redistribution of its density between the fragments.

**nonadiabatic hop between adiabatic states**
    population moving from :math:`i` to :math:`j`. Lives in the SHPROP
    populations, not in a PROCAR.

**projection-weighted diagonal fragment population**
    :math:`P_g = \\sum_i P_i w_{ig}`, which is what ``character-populations``
    reports. Closer to a diabatic reading than a band-resolved one, but
    **diagonal**: SHPROP records no coherences and a PROCAR no cross-band
    projections, so it is not exact fragment charge.

**a true diabatic transformation**
    a unitary that removes the derivative coupling. This package does **not**
    perform one, and nothing here should be called diabatic without that
    qualification.

The two failure modes worth stating plainly, because they run opposite ways:

* Staying on one adiabatic state through an avoided crossing **changes** the
  fragment identity. No hop occurred, and the occupied density can nonetheless
  have redistributed between the fragments.
* Hopping between two adiabatic states at a crossing can **preserve** the
  fragment identity. A hop occurred, and the occupied density need not have
  redistributed at all.

So a character swap is not a surface hop, and a surface hop is not charge
transfer.

What an event *is* classified on is the projection-weighted **diagonal**
fragment population :math:`P_g = \\sum_i P_i w_{ig}` -- whether it moved, and
nothing else. That quantity sums over the whole SHPROP basis, so no
re-ordering of band labels can move it, and a change in it is therefore not a
labelling artifact. **A change carried by character evolution is not "no charge
moved"**: an occupied state whose composition turns can correspond to a spatial
redistribution of its density, which is the first bullet above. Equally,
neither term of the split may be equated with charge motion, and :math:`P_g` is
not exact fragment charge -- it is diagonal, the coherences are absent from the
inputs, and projection weight outside the declared fragments is unassigned. The
split describes a change; it does not decide whether one happened, and it never
names a mechanism.

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
why a fixed band label obscures such a redistribution: the label is constant
while :math:`\\theta` is not.
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

#: The classification given to a character swap when **no** fragment population
#: was supplied.  ``fragment_population_change is None`` means *not evaluated*,
#: never *evaluated and found to be zero*, and the two must not share a label:
#: a configuration-level scan (``projection_character.csv`` + EIGTXT + NATXT,
#: with no SHPROP) cannot say anything at all about fragment population, and a
#: report that said "nothing moved" from such a run would be claiming an
#: absence it never measured.
NOT_EVALUATED = "character_swap_population_not_evaluated"

#: A character swap where ``P_g`` **was** evaluated and did not move beyond
#: tolerance.  A measured absence, and a statement about ``P_g`` alone.
NO_PROJECTED_CHANGE = "character_swap_without_projected_fragment_change"

#: A character swap where ``P_g`` **was** evaluated and did move.  The
#: projection-weighted diagonal fragment population changed; which microscopic
#: process produced it is not resolved here, and no bookkeeping term of the
#: split may be read as deciding that.
PROJECTED_CHANGE = "character_swap_with_projected_fragment_change"

#: Share of the absolute movement one bookkeeping term must carry before the
#: event is *described* as dominated by it.  Matches
#: :data:`namd_analysis.episodes.DOMINANCE_SPLIT`, so a frame and an episode
#: are described on the same convention.
DOMINANCE_SHARE = 0.7

NOT_EVALUATED_NOTE = (
    "no fragment population was supplied with this scan, so whether the "
    "projected fragment population moved was NOT EVALUATED. This is not a "
    "finding of zero movement: a configuration-level crossing scan reads "
    "projection_character.csv, EIGTXT and NATXT, none of which carries a SHPROP "
    "population, and the question cannot be asked of them. Rerun with --shprop "
    "and --state-map to evaluate it"
)

#: The **one** place the step decomposition is written down, so a report, a
#: docstring and the code cannot drift apart.  This is the symmetric midpoint
#: form, which is what :func:`history_fragment_population` computes.  An
#: endpoint-biased form is equally exact in the sum but apportions up to half a
#: step's movement differently between the two terms, and a document quoting it
#: while the code uses this one would misdescribe every number in the split.
DECOMPOSITION_IDENTITY = (
    "dP_g = sum_i [P_i(t) - P_i(t-1)] * 0.5*(w_ig[f(t)] + w_ig[f(t-1)]) "
    "+ sum_i 0.5*(P_i(t) + P_i(t-1)) * (w_ig[f(t)] - w_ig[f(t-1)])"
)

#: What ``P_g`` is, what a change in it does and does not establish, and why
#: neither term of the split may be dismissed as bookkeeping about labels.
#:
#: The error this note exists to prevent: reading ``character_evolution`` as
#: "only a relabelling, nothing moved".  It is not.  An occupied adiabatic
#: state whose own composition turns from BCF-like to PCBM-like can correspond
#: to a spatial redistribution of the occupied density -- the adiabatic passage
#: the two-state picture at the top of this module describes.
#:
#: The opposite error is equally available, and this note does not commit it:
#: ``P_g`` is a projection-weighted **diagonal** quantity, so neither term may
#: be equated with charge motion outright.
PROJECTION_NOTE = (
    "P_g = sum_i P_i w_ig is the projection-weighted DIAGONAL fragment "
    "population: the SHPROP populations contracted with the PROCAR fragment "
    "weights, summed over EVERY band of the basis. Because the sum runs over "
    "the whole basis, a re-ordering of band labels leaves it invariant, so a "
    "change in it is not a labelling artifact. In particular character_evolution "
    "is NOT mere relabelling: an occupied state whose own composition changes "
    "can correspond to a spatial redistribution of the occupied density, and "
    "describing such a change as 'no charge moved' is wrong. Neither term of "
    "the split may be equated with charge motion either, and P_g is NOT exact "
    "fragment charge: SHPROP records no coherences and a PROCAR carries no "
    "cross-band projections, so the off-diagonal terms of Tr[rho P_g] are "
    "absent from the inputs, and projection weight falling outside the declared "
    "fragments is unassigned. The split itself is one of infinitely many exact "
    "splits, a bookkeeping convention rather than a branching fraction, and no "
    "surface-hopping record is an input here, so neither term names a "
    "microscopic process"
)


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
    #: ``sum_g W_ig`` before normalization, when the table carries it.
    #: ``w_ig * captured`` recovers the raw PAW-sphere weight, which is
    #: what a normalization-artifact check needs.
    captured: Optional[np.ndarray] = None  # (nframe, nband)
    #: The sum over every PROCAR ion, when the table carries it.  Equal to
    #: ``captured`` for a complete atom partition; ``1 - total`` is the band
    #: weight outside every ion's projection.
    total: Optional[np.ndarray] = None  # (nframe, nband)

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
    captured_rows: Dict[Tuple[int, int], float] = {}
    total_rows: Dict[Tuple[int, int], float] = {}
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
            if record.get("captured_projection") not in (None, ""):
                captured_rows[(frame, band)] = float(record["captured_projection"])
            if record.get("total_projection") not in (None, ""):
                total_rows[(frame, band)] = float(record["total_projection"])
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
    captured = None
    if len(captured_rows) == len(rows):
        captured = np.empty((len(frames), len(bands)), dtype=float)
        for (frame, band), value in captured_rows.items():
            captured[frame_at[frame], band_at[band]] = value
    total = None
    if len(total_rows) == len(rows):
        total = np.empty((len(frames), len(bands)), dtype=float)
        for (frame, band), value in total_rows.items():
            total[frame_at[frame], band_at[band]] = value

    return CharacterSeries(
        frames=np.asarray(frames, dtype=int),
        bands=np.asarray(bands, dtype=int),
        groups=groups,
        weights=weights,
        captured=captured,
        total=total,
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
    #: ``None`` means the fragment population was **not evaluated** for this
    #: event, which is a different statement from a change of zero. Read
    #: alongside :attr:`population_evaluated`, which says so explicitly rather
    #: than leaving it to a reader of a blank CSV cell.
    fragment_population_change: Optional[float]
    classification: str
    note: str
    #: The exact split of this step's change in ``P_g`` for
    #: :attr:`dominant_fragment`, into occupation moving at fixed character and
    #: character moving at fixed occupation. Signed, and they sum to that
    #: fragment's signed change. Only a single history carries the per-band
    #: populations this needs, so both are ``None`` on the ensemble path. See
    #: :class:`HistoryPopulation`.
    #:
    #: **Neither is a mechanism.** Both move the projected occupied density,
    #: and a large ``character_evolution`` does not mean nothing moved: see
    #: :data:`PROJECTION_NOTE`. They describe the change; they do not explain
    #: it, and they are never used to decide :attr:`classification`.
    occupation_redistribution: Optional[float] = None
    character_evolution: Optional[float] = None
    #: The fragment whose ``|dP_g|`` was largest at this step, and therefore
    #: the fragment the three numbers above refer to.
    dominant_fragment: Optional[str] = None
    #: ``occupation_dominated`` / ``character_dominated`` / ``mixed`` /
    #: ``no_movement``, from the share of absolute movement each term carries.
    #: A **description** of the bookkeeping split, not a physical branching
    #: fraction and not a mechanism. ``None`` where the split is unavailable.
    decomposition_descriptor: Optional[str] = None
    #: Did ``P_g`` move the way the dominance swap points -- the fragment the
    #: dominance moved to gaining what the one it left lost? Tested on the
    #: **projected population itself**, never on one term of the split.
    #: ``None`` where the question is not well posed.
    swap_direction_matches_projected_change: Optional[bool] = None

    @property
    def population_evaluated(self) -> bool:
        """Was fragment population available to be tested at this event?

        A blank ``projected_fragment_population_change`` cell in a CSV is
        ambiguous to the eye, so this travels beside it as its own column:
        ``False`` means the question was never asked, not that the answer was
        zero.
        """
        return self.fragment_population_change is not None

    def as_row(self) -> List[Any]:
        return [
            self.frame, self.band_i, self.band_j, self.gap_ev, self.nac_ev,
            self.character_change_i, self.character_change_j,
            self.dominant_i_before, self.dominant_i_after,
            self.dominant_j_before, self.dominant_j_after,
            self.small_gap, self.strong_nac, self.character_swap,
            self.population_evaluated,
            self.fragment_population_change,
            self.dominant_fragment,
            self.occupation_redistribution, self.character_evolution,
            self.decomposition_descriptor,
            self.swap_direction_matches_projected_change,
            self.classification,
        ]


EVENT_HEADER = [
    "frame", "band_i", "band_j", "gap_ev", "nac_ev",
    "character_change_i", "character_change_j",
    "dominant_i_before", "dominant_i_after",
    "dominant_j_before", "dominant_j_after",
    "small_gap", "strong_nac", "character_swap",
    "population_evaluated",
    "projected_fragment_population_change",
    "dominant_fragment",
    "occupation_redistribution", "character_evolution",
    "decomposition_descriptor",
    "swap_direction_matches_projected_change",
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


def _direction_agrees(
    moves: Sequence[Tuple[str, str]],
    delta: Optional[Dict[str, float]],
    tolerance: float,
) -> Optional[bool]:
    """Did the **projected fragment population** move the way the swap points?

    ``delta`` is the per-fragment change in ``P_g`` itself -- the observable --
    and never one term of the symmetric split.  Testing the occupation term
    alone would ask whether *occupation* followed the swap and then report the
    answer as though it were about charge, which it is not: a change carried
    entirely by ``character_evolution`` moves the occupied density just as
    surely.  See :data:`PROJECTION_NOTE`.

    Agreement means the fragment the dominance moved *to* gained projected
    population while the one it left lost it.

    This is a **descriptor**, not a classification.  A swap and a population
    change can co-occur without one causing the other, so agreement here is not
    evidence of a mechanism and disagreement is not evidence against one; the
    classification rests on ``|dP_g|`` alone.

    ``None`` where the question is not well posed, and the caller must not read
    that as either answer:

    * no per-fragment change was supplied;
    * the character changed without any band's dominant fragment moving, so
      there is no swap direction at all;
    * the swap names **no single direction** -- the usual two-state avoided
      crossing, where one band goes BCF->PCBM while the other goes PCBM->BCF.
      The pair exchanged character, so movement either way would "agree" with
      one of the two bands.  That case cannot be decided, and is left undecided.
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


def decomposition_descriptor(
    occupation: Optional[float],
    character: Optional[float],
    tolerance: float,
    dominance: float = DOMINANCE_SHARE,
) -> Optional[str]:
    """Which bookkeeping term carries the movement -- a description, not a cause.

    ``occupation_dominated``, ``character_dominated``, ``mixed``, or
    ``no_movement``; ``None`` when the split was not available.

    This **describes** an observed change in ``P_g``; it never decides whether
    one occurred, and it is not a branching fraction of the dynamics.  An event
    described as ``character_dominated`` changed ``P_g`` by exactly as much as
    an ``occupation_dominated`` one did: the share is a share of the
    accounting, not a scale of how much charge moved.
    """
    if occupation is None or character is None:
        return None
    total = abs(occupation) + abs(character)
    if total <= tolerance:
        return "no_movement"
    share = abs(occupation) / total
    if share >= dominance:
        return "occupation_dominated"
    if share <= 1.0 - dominance:
        return "character_dominated"
    return "mixed"


def _classify(
    small_gap: bool,
    strong_nac: bool,
    character_swap: bool,
    population_change: Optional[float],
    population_tolerance: float,
    direction_agrees: Optional[bool] = None,
    descriptor: Optional[str] = None,
) -> Tuple[str, str]:
    """Name what happened, and refuse to name what the data do not show.

    The classification rests on **one observable**: the projection-weighted
    fragment population ``P_g = sum_i P_i w_ig``, and whether it moved beyond
    tolerance.  Three cases for a character swap, and no others:

    ``P_g`` not supplied
        :data:`NOT_EVALUATED` -- the question was never asked.
    ``|dP_g| <= tolerance``
        :data:`NO_PROJECTED_CHANGE` -- a measured absence, about ``P_g``.
    ``|dP_g| > tolerance``
        :data:`PROJECTED_CHANGE` -- the projected occupied density moved.

    **The symmetric split never decides this.** An earlier version called a
    change "no fragment transfer" whenever ``occupation_redistribution`` was
    ~zero, on the reasoning that a character-driven change is a mere
    relabelling.  That reasoning is wrong, and it contradicted this module's
    own opening paragraph: an occupied adiabatic state turning from BCF-like to
    PCBM-like at fixed ``P_i`` can correspond to a spatial redistribution of its
    density between the fragments, with no band-index population hop anywhere --
    the adiabatic passage the two-state picture describes.  ``P_g`` also sums
    over the whole SHPROP basis, so a relabelling cannot move it at all.  What
    the classification does **not** claim is that either term of the split is
    charge motion, or that ``P_g`` is exact fragment charge; see
    :data:`PROJECTION_NOTE`.

    ``descriptor`` and ``direction_agrees`` therefore only *describe* a change
    the classification has already established on ``|dP_g|``.

    ``population_change is None`` is **not evaluated**, and is kept apart from
    an evaluated zero.  A configuration-level scan has no SHPROP populations at
    all, so it cannot find that nothing moved -- it can only report that the
    question was not asked.  Collapsing the two would turn a missing input into
    a scientific claim.
    """
    evaluated = population_change is not None
    moved = evaluated and abs(population_change) > population_tolerance
    parts = []
    if small_gap:
        parts.append("small gap")
    if strong_nac:
        parts.append("strong NAC")
    if character_swap:
        parts.append("character swap")
    if moved:
        parts.append("projected fragment population change")
    if not evaluated:
        parts.append("projected fragment population not evaluated")

    if character_swap and not evaluated:
        label = NOT_EVALUATED
        note = (
            "the dominant fragment of a band index changed. Whether the "
            "projection-weighted diagonal fragment population moved with it is UNKNOWN: "
            + NOT_EVALUATED_NOTE
            + ". Nothing here supports saying charge moved, and nothing here "
            "rules it out"
        )
    elif character_swap and not moved:
        label = NO_PROJECTED_CHANGE
        note = (
            "the dominant fragment of a band index changed, and the "
            "projection-weighted diagonal fragment population P_g did not move beyond "
            "tolerance. Because P_g sums over every band of the basis, this is "
            "a measured statement about where the occupied density sits, not "
            "about labels. It says the projected density stayed put across this "
            "step; it is not by itself a statement about nonadiabatic hops, for "
            "which no hop record is an input"
        )
    elif character_swap and moved:
        label = PROJECTED_CHANGE
        note = (
            "the dominant fragment of a band index changed AND the "
            "projection-weighted diagonal fragment population moved with it. "
            + PROJECTION_NOTE
        )
        if descriptor == "character_dominated":
            note += (
                ". Here the movement is carried mainly by character evolution: "
                "the occupied state's own composition changed while its "
                "population did not. That is an adiabatic passage and can "
                "correspond to a spatial redistribution of the occupied "
                "density -- it MUST NOT be reported as 'no charge moved' or as "
                "a relabelling. What it does not establish is a nonadiabatic "
                "hop, and no hop record is an input"
            )
        elif descriptor == "occupation_dominated":
            note += (
                ". Here the movement is carried mainly by occupation moving "
                "between states of differing character. That is a description "
                "of the split, not evidence that a hop occurred: no hop record "
                "is an input"
            )
        elif descriptor == "mixed":
            note += (
                ". Here neither term of the split carries the movement on its "
                "own, which is a description of the bookkeeping and not a "
                "statement that two mechanisms ran in that proportion"
            )
        if direction_agrees is True:
            note += (
                ". The projected population moved in the direction the "
                "dominance swap points -- the fragment it moved to gained what "
                "the one it left lost. The two agreeing is a description of "
                "this step, not evidence that one caused the other"
            )
        elif direction_agrees is False:
            note += (
                ". The projected population did NOT move in the direction the "
                "dominance swap points. The change is real either way; it "
                "simply does not line up with the fragments the swap names"
            )
    elif moved and not character_swap:
        label = "fragment_population_change_without_character_swap"
        note = (
            "the projection-weighted diagonal fragment population moved while every "
            "band kept its dominant character, so the movement is occupation "
            "redistributing among adiabatic states whose characters did not "
            "change. " + PROJECTION_NOTE
        )
    elif small_gap and strong_nac:
        label = "close_and_coupled_without_character_swap"
        note = (
            "the states were close and strongly coupled, but the character did "
            "not change at this frame. A coupling is an opportunity, not an "
            "event"
        ) + (
            ", and the projected fragment population did not move either"
            if evaluated
            else ". " + NOT_EVALUATED_NOTE
        )
    else:
        label = "flagged_metric_only"
        note = (
            "one or more raw metrics crossed its threshold without any change "
            "of character"
        ) + (
            " or of projected fragment population"
            if evaluated
            else ". " + NOT_EVALUATED_NOTE
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
            dominant_group = None
            delta: Optional[Dict[str, float]] = None
            if fragment_population is not None:
                before = fragment_population.get(previous)
                after = fragment_population.get(frame)
                if before and after:
                    delta = {
                        g: float(after.get(g, 0.0) - before.get(g, 0.0))
                        for g in character.groups
                    }
                    dominant_group = max(delta, key=lambda g: abs(delta[g]))
                    change = float(abs(delta[dominant_group]))

            # This path has only the contracted P_g, not the per-band
            # populations, so the split is unavailable and no descriptor is
            # produced. That costs nothing the classification needs: it rests
            # on |dP_g| alone, which is exactly what is in hand here.
            label, note = _classify(
                small_gap, strong_nac, swap or exchanged, change,
                population_tolerance,
                _direction_agrees(
                    _swap_moves(
                        dom_i_before, dom_i_after, dom_j_before, dom_j_after
                    ),
                    delta,
                    population_tolerance,
                ),
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
                    dominant_fragment=dominant_group,
                    swap_direction_matches_projected_change=_direction_agrees(
                        _swap_moves(
                            dom_i_before, dom_i_after, dom_j_before, dom_j_after
                        ),
                        delta,
                        population_tolerance,
                    ),
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
    """One history's projection-weighted diagonal fragment population, and its split.

    Each step's change splits exactly, ``dP_g = dP_g^pop + dP_g^char``, using
    the **symmetric midpoint** form -- this is the form the code implements,
    and :data:`DECOMPOSITION_IDENTITY` is the single place it is written down:

        dP_g^pop  = sum_i [P_i(t) - P_i(t-1)] * 0.5*(w_ig[f(t)] + w_ig[f(t-1)])
        dP_g^char = sum_i 0.5*(P_i(t) + P_i(t-1)) * (w_ig[f(t)] - w_ig[f(t-1)])

    An endpoint-biased form -- ``dP_g^pop = sum_i dP_i w_ig[f(t)]`` with
    ``dP_g^char = sum_i P_i(t-1) dw_ig`` -- is equally exact in the sum but
    assigns up to half a step's movement differently between the two terms, so
    the symmetric one is used and neither endpoint is privileged.

    The first term is occupation moving between states at fixed character; the
    second is an occupied state's own character evolving at fixed occupation.
    The second is **not** a relabelling -- ``total`` is summed over the whole
    basis, so re-ordering band labels leaves it invariant, and a state whose
    composition changes can correspond to a spatial redistribution of the
    occupied density.  See :data:`PROJECTION_NOTE` for what ``total`` is and is
    not.

    The split is exact, and it is **one of infinitely many exact splits**. It
    describes how a change in ``total`` is accounted for; it is not a branching
    fraction, it names no mechanism, and nothing decides whether a change
    occurred except ``total`` itself.  Row 0 has no previous step and is zero in
    both.
    """

    total: np.ndarray  # (ntime, ngroup)
    occupation_redistribution: np.ndarray  # (ntime, ngroup)
    character_evolution: np.ndarray  # (ntime, ngroup)
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

    def descriptor_counts(self, window: Optional[str] = None) -> Dict[str, int]:
        """How the split *describes* the events that moved ``P_g``.

        Counted only over :data:`PROJECTED_CHANGE` events, because a descriptor
        describes a change and there is nothing to describe otherwise. These
        are descriptions of a bookkeeping convention, never mechanisms: see
        :data:`PROJECTION_NOTE`.
        """
        out: Dict[str, int] = {}
        for event, which in zip(self.events, self.window_of_event):
            if window is not None and which != window:
                continue
            if event.classification != PROJECTED_CHANGE:
                continue
            if event.decomposition_descriptor:
                out[event.decomposition_descriptor] = (
                    out.get(event.decomposition_descriptor, 0) + 1
                )
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
        # Symmetric midpoint split. An endpoint-biased form is equally exact
        # in the sum but assigns up to ~0.5 of a population differently between
        # the two terms, so the symmetric one is used: neither endpoint is
        # privileged.
        driven_pop[offset : offset + rows] = np.einsum(
            "ts,tsg->tg", pops - shifted_pops, 0.5 * (weights + shifted_weights)
        )
        driven_char[offset : offset + rows] = np.einsum(
            "ts,tsg->tg", 0.5 * (pops + shifted_pops), weights - shifted_weights
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
        occupation_redistribution=driven_pop,
        character_evolution=driven_char,
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
        # The observable the classification rests on: the signed per-fragment
        # change in P_g, and the fragment that moved furthest. The split terms
        # are read for the SAME fragment, so the three numbers on the event
        # refer to one quantity and dP = occupation + character holds on it.
        step_delta = population[step] - population[step - 1]
        dominant_index = int(np.argmax(np.abs(step_delta)))
        dominant_group = character.groups[dominant_index]
        change = float(abs(step_delta[dominant_index]))
        delta_by_group = dict(zip(character.groups, (float(x) for x in step_delta)))
        occupation_term = float(split.occupation_redistribution[step, dominant_index])
        character_term = float(split.character_evolution[step, dominant_index])
        # A description of the change, never a test of whether one happened.
        descriptor = decomposition_descriptor(
            occupation_term, character_term, population_tolerance
        )

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
            # The direction test runs on dP_g, the observable -- not on the
            # occupation term, which would ask about occupation and answer as
            # though the question had been about charge.
            agrees = _direction_agrees(
                _swap_moves(dom_i_before, dom_i_after, dom_j_before, dom_j_after),
                delta_by_group,
                population_tolerance,
            )
            label, note = _classify(
                small_gap,
                strong_nac,
                swap or exchanged,
                change,
                population_tolerance,
                agrees,
                descriptor,
            )
            events.append(
                CrossingEvent(
                    occupation_redistribution=occupation_term,
                    character_evolution=character_term,
                    dominant_fragment=dominant_group,
                    decomposition_descriptor=descriptor,
                    swap_direction_matches_projected_change=agrees,
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
            "by_decomposition_descriptor": h.descriptor_counts(),
            "by_window": {name: h.counts(name) for name in window_names} if window_names else {},
        }
        for h in histories
    ]
    totals: Dict[str, int] = {}
    for record in per_history:
        for label, count in record["by_classification"].items():
            totals[label] = totals.get(label, 0) + count

    descriptors: Dict[str, int] = {}
    for record in per_history:
        for label, count in record["by_decomposition_descriptor"].items():
            descriptors[label] = descriptors.get(label, 0) + count

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
        "totals_by_decomposition_descriptor": descriptors,
        "decomposition_descriptor_note": (
            "these describe how the exact symmetric split apportions changes "
            "that P_g already established. They are descriptions of a "
            "bookkeeping convention, NOT physical branching fractions and NOT "
            "mechanisms, and an event described as character_dominated changed "
            "P_g by exactly as much as an occupation_dominated one did -- the "
            "share is a share of the accounting, not a scale of how much charge "
            "moved. " + PROJECTION_NOTE
        ),
        "totals_by_window": by_window,
        "note": (
            "every event was classified on the history that produced it, using "
            "that history's own resolved frame mapping and its own "
            "projection-weighted diagonal fragment population, and only then summed. "
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
