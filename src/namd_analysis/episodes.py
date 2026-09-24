"""Crossing episodes: what a BCF/PCBM exchange did to the fragment population.

A single step is the wrong unit for asking whether charge moved.  A passage
through an avoided crossing occupies several frames -- character mixes, is
exchanged, and separates again -- so the question "did the population follow?"
has to be asked over the **episode**, not over one finite difference.

Every quantity here is one of three kinds, and the summary keeps them apart:

* **observed** -- read from SHPROP populations and PROCAR weights;
* **inferred** -- the split of a population change into its two exact terms,
  which is arithmetic on observed quantities but is a *decomposition*, not a
  measurement of a mechanism;
* **not available** -- anything needing the hop record.  Surface-hopping
  trajectories record which state each trajectory occupied; that file is not
  an input here, so no statement about hops, hop counts or hopping rates can
  be made, and ``occupation_redistribution`` must never be read as "hopping".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "EpisodeError",
    "DOMINANCE_SPLIT",
    "NET_TRANSFER_TOLERANCE",
    "CrossingEpisode",
    "find_episodes",
    "classify_episode",
    "local_crossing_table",
    "isolation_check",
    "raw_weight_check",
    "fixed_vs_dynamic_late",
    "manifold_table",
    "band_participation",
    "alignment_scan",
]

#: An episode is called dominated by one term when it carries at least this
#: share of the total absolute movement.  Between the two bounds it is mixed.
DOMINANCE_SPLIT = 0.7

#: Net fragment change below this is "no net transfer" rather than a direction.
NET_TRANSFER_TOLERANCE = 1.0e-3

NO_HOP_RECORD = (
    "no hop record is an input to this analysis. Surface-hopping trajectories "
    "record which adiabatic state each trajectory occupied at each step; "
    "without that file, occupation_redistribution is a redistribution of "
    "SHPROP populations and must NOT be read as a hop count, a hopping rate, "
    "or evidence that hops occurred"
)


class EpisodeError(ValueError):
    """Raised when an episode cannot be formed from the inputs given."""


@dataclass
class CrossingEpisode:
    """One contiguous passage in which two bands exchange fragment character."""

    band_i: int
    band_j: int
    first_frame: int
    last_frame: int
    #: Index range on the supplied frame axis, half-open.
    start: int
    stop: int
    min_gap_ev: float
    max_abs_nac_ev: Optional[float]
    #: Dominant fragment of each band at the first and last frame.
    dominant_i_before: str = ""
    dominant_i_after: str = ""
    dominant_j_before: str = ""
    dominant_j_after: str = ""
    #: Summed over the episode, per fragment.  None without SHPROP.
    net_change: Optional[Dict[str, float]] = None
    occupation_redistribution: Optional[Dict[str, float]] = None
    character_evolution: Optional[Dict[str, float]] = None
    #: Occupation of the two crossing bands at the episode edges.
    occupancy_before: Optional[Dict[int, float]] = None
    occupancy_after: Optional[Dict[int, float]] = None
    classification: str = "not_classified"
    direction: str = "not_determined"
    note: str = ""

    def as_row(self) -> List[Any]:
        def g(mapping, key):
            return "" if mapping is None else mapping.get(key, "")

        return [
            self.band_i, self.band_j, self.first_frame, self.last_frame,
            self.min_gap_ev, self.max_abs_nac_ev,
            self.dominant_i_before, self.dominant_i_after,
            self.dominant_j_before, self.dominant_j_after,
            g(self.net_change, "BCF"), g(self.net_change, "PCBM"),
            g(self.occupation_redistribution, "BCF"),
            g(self.occupation_redistribution, "PCBM"),
            g(self.character_evolution, "BCF"),
            g(self.character_evolution, "PCBM"),
            self.direction, self.classification,
        ]


EPISODE_HEADER = [
    "band_i", "band_j", "first_frame", "last_frame",
    "min_gap_ev", "max_abs_nac_ev",
    "dominant_i_before", "dominant_i_after",
    "dominant_j_before", "dominant_j_after",
    "net_BCF", "net_PCBM",
    "occupation_redistribution_BCF", "occupation_redistribution_PCBM",
    "character_evolution_BCF", "character_evolution_PCBM",
    "direction", "classification",
]


def _dominant(character, frame_index: int, band_index: int) -> str:
    return character.groups[int(np.argmax(character.weights[frame_index, band_index]))]


def find_episodes(
    character,
    couplings,
    band_i: int,
    band_j: int,
    donor: str = "BCF",
    acceptor: str = "PCBM",
    pad: int = 3,
) -> List[CrossingEpisode]:
    """Contiguous frame windows where ``band_i`` and ``band_j`` swap dominance.

    Adjacent swaps are merged into one episode: a passage that exchanges
    character and exchanges back is *one* encounter with the crossing, not two
    independent events, and counting it twice would inflate every rate derived
    from it.
    """
    band_at = character.band_index()
    for band in (band_i, band_j):
        if band not in band_at:
            raise EpisodeError(f"band {band} is not in the character table")
    bi, bj = band_at[band_i], band_at[band_j]
    nframes = character.frames.size

    dom_i = np.array([_dominant(character, f, bi) for f in range(nframes)])
    dom_j = np.array([_dominant(character, f, bj) for f in range(nframes)])
    involved = {donor, acceptor}
    swap_at = np.array(
        [
            (dom_i[k] != dom_i[k - 1] or dom_j[k] != dom_j[k - 1])
            and (
                {dom_i[k], dom_i[k - 1]} <= involved
                or {dom_j[k], dom_j[k - 1]} <= involved
            )
            for k in range(1, nframes)
        ]
    )
    marks = np.flatnonzero(swap_at) + 1
    if marks.size == 0:
        return []

    groups: List[List[int]] = [[int(marks[0])]]
    for mark in marks[1:]:
        if mark - groups[-1][-1] <= 2 * pad:
            groups[-1].append(int(mark))
        else:
            groups.append([int(mark)])

    gap = None
    nac = None
    if couplings is not None:
        row_of = {int(f): r for r, f in enumerate(couplings.frames)}
        state_of = {int(b): k for k, b in enumerate(character.bands)}
        si, sj = state_of.get(band_i), state_of.get(band_j)
        if si is not None and sj is not None:
            gap = {}
            nac = {} if couplings.nac is not None else None
            for frame, row in row_of.items():
                if max(si, sj) < couplings.energies.shape[1]:
                    gap[frame] = abs(
                        couplings.energies[row, si] - couplings.energies[row, sj]
                    )
                    if nac is not None:
                        nac[frame] = abs(couplings.nac[row, si, sj])

    episodes = []
    for group in groups:
        start = max(0, group[0] - pad)
        stop = min(nframes, group[-1] + pad + 1)
        frames = [int(character.frames[k]) for k in range(start, stop)]
        gaps = [gap[f] for f in frames if gap and f in gap] if gap else []
        nacs = [nac[f] for f in frames if nac and f in nac] if nac else []
        episodes.append(
            CrossingEpisode(
                band_i=band_i,
                band_j=band_j,
                first_frame=frames[0],
                last_frame=frames[-1],
                start=start,
                stop=stop,
                min_gap_ev=float(min(gaps)) if gaps else float("nan"),
                max_abs_nac_ev=float(max(nacs)) if nacs else None,
                dominant_i_before=dom_i[start],
                dominant_i_after=dom_i[stop - 1],
                dominant_j_before=dom_j[start],
                dominant_j_after=dom_j[stop - 1],
            )
        )
    return episodes


def classify_episode(
    episode: CrossingEpisode,
    donor: str = "BCF",
    acceptor: str = "PCBM",
    dominance_split: float = DOMINANCE_SPLIT,
    net_tolerance: float = NET_TRANSFER_TOLERANCE,
) -> CrossingEpisode:
    """Name what moved the acceptor population over the episode, and how.

    Requires the decomposition to have been filled in; without SHPROP the
    episode is left ``not_classified`` rather than guessed.
    """
    if episode.net_change is None:
        episode.classification = "not_classified"
        episode.direction = "not_determined"
        episode.note = (
            "no SHPROP populations were supplied, so the fragment population "
            "over this episode is unknown. The character exchange is observed; "
            "whether charge followed it is not"
        )
        return episode

    net_a = float(episode.net_change.get(acceptor, 0.0))
    net_d = float(episode.net_change.get(donor, 0.0))
    occ = float((episode.occupation_redistribution or {}).get(acceptor, 0.0))
    chr_ = float((episode.character_evolution or {}).get(acceptor, 0.0))

    if abs(net_a) <= net_tolerance and abs(net_d) <= net_tolerance:
        episode.direction = "no_net_transfer"
        episode.classification = "no_net_fragment_transfer"
        episode.note = (
            f"the character was exchanged but neither {donor} nor {acceptor} "
            f"ended the episode with a net projected population change above "
            f"{net_tolerance:g}. The projected occupied density ended the "
            "episode where it began. That is a statement about the NET over "
            "the episode: the density may have moved and returned within it"
        )
        return episode

    if net_a > net_tolerance and net_d < -net_tolerance:
        episode.direction = f"{donor}->{acceptor}"
    elif net_a < -net_tolerance and net_d > net_tolerance:
        episode.direction = f"{acceptor}->{donor}"
    else:
        episode.direction = "not_between_these_fragments"

    total = abs(occ) + abs(chr_)
    if total <= 0.0:
        share = 0.0
    else:
        share = abs(occ) / total

    if share >= dominance_split:
        episode.classification = "occupation_redistribution_dominated"
        episode.note = (
            f"{100*share:.0f}% of the absolute {acceptor} movement is "
            "occupation redistribution: population moved between states whose "
            f"character was comparatively fixed. {NO_HOP_RECORD}"
        )
    elif share <= 1.0 - dominance_split:
        episode.classification = "character_following_transfer"
        episode.note = (
            f"{100*(1-share):.0f}% of the absolute {acceptor} movement is "
            "character evolution: the states' own composition changed while "
            "their occupation was comparatively fixed. This is the adiabatic, "
            "character-following case -- charge moved because the state the "
            "population sat on became a different fragment"
        )
    else:
        episode.classification = "mixed"
        episode.note = (
            f"occupation redistribution carries {100*share:.0f}% and character "
            f"evolution {100*(1-share):.0f}% of the absolute {acceptor} "
            "movement; neither dominates"
        )
    return episode


def local_crossing_table(
    character,
    couplings,
    band_i: int,
    band_j: int,
    first_frame: int,
    last_frame: int,
    neighbours: Sequence[int] = (),
    donor: str = "BCF",
    acceptor: str = "PCBM",
) -> Tuple[List[str], List[List[Any]]]:
    """Frame-by-frame energies, gap and character across a known crossing.

    Whatever needs SHPROP is left out here; this is the part that can be built
    from the character table and the energies alone.
    """
    band_at = character.band_index()
    frame_at = character.frame_index()
    state_of = {int(b): k for k, b in enumerate(character.bands)}
    row_of = (
        {int(f): r for r, f in enumerate(couplings.frames)} if couplings is not None else {}
    )

    bands = [int(b) for b in neighbours] + [band_i, band_j]
    header = ["frame"]
    header += [f"E_{b}_eV" for b in sorted(set(bands))]
    header += [
        f"gap_{band_i}_{band_j}_eV", f"abs_nac_{band_i}_{band_j}_eV",
        f"{band_i}_{donor}", f"{band_i}_{acceptor}",
        f"{band_j}_{donor}", f"{band_j}_{acceptor}",
        f"{band_i}_captured", f"{band_j}_captured",
        f"{band_i}_raw_{donor}", f"{band_i}_raw_{acceptor}",
        f"{band_j}_raw_{donor}", f"{band_j}_raw_{acceptor}",
    ]

    gi, gj = character.groups.index(donor), character.groups.index(acceptor)
    rows: List[List[Any]] = []
    for frame in range(int(first_frame), int(last_frame) + 1):
        if frame not in frame_at:
            continue
        k = frame_at[frame]
        row: List[Any] = [frame]
        for b in sorted(set(bands)):
            s = state_of.get(b)
            r = row_of.get(frame)
            row.append(
                float(couplings.energies[r, s])
                if (couplings is not None and s is not None and r is not None
                    and s < couplings.energies.shape[1])
                else ""
            )
        si, sj = state_of.get(band_i), state_of.get(band_j)
        r = row_of.get(frame)
        if couplings is not None and None not in (si, sj) and r is not None:
            row.append(float(abs(couplings.energies[r, si] - couplings.energies[r, sj])))
            row.append(
                float(abs(couplings.nac[r, si, sj])) if couplings.nac is not None else ""
            )
        else:
            row += ["", ""]

        wi = character.weights[k, band_at[band_i]]
        wj = character.weights[k, band_at[band_j]]
        ci = float(character.captured[k, band_at[band_i]]) if character.captured is not None else float("nan")
        cj = float(character.captured[k, band_at[band_j]]) if character.captured is not None else float("nan")
        row += [float(wi[gi]), float(wi[gj]), float(wj[gi]), float(wj[gj]), ci, cj]
        # Raw, unnormalized PAW-sphere weight: w_ig * sum_g W_ig.
        row += [
            float(wi[gi]) * ci, float(wi[gj]) * ci,
            float(wj[gi]) * cj, float(wj[gj]) * cj,
        ]
        rows.append(row)
    return header, rows


def isolation_check(
    couplings,
    character,
    band_i: int,
    band_j: int,
    neighbours: Sequence[int],
    first_frame: int,
    last_frame: int,
) -> Dict[str, Any]:
    """Is the ``band_i``/``band_j`` crossing separated from its neighbours?

    A two-state reading of an avoided crossing is only justified if no third
    state comes as close as the pair does to each other.  This reports the
    ratio; it does not decide for the reader.
    """
    state_of = {int(b): k for k, b in enumerate(character.bands)}
    row_of = {int(f): r for r, f in enumerate(couplings.frames)}
    rows = [row_of[f] for f in range(int(first_frame), int(last_frame) + 1) if f in row_of]
    if not rows:
        raise EpisodeError("no coupling rows in the requested window")
    si, sj = state_of[band_i], state_of[band_j]
    pair_gap = np.abs(couplings.energies[rows, si] - couplings.energies[rows, sj])

    others = {}
    for nb in neighbours:
        sn = state_of.get(int(nb))
        if sn is None:
            continue
        d = np.minimum(
            np.abs(couplings.energies[rows, sn] - couplings.energies[rows, si]),
            np.abs(couplings.energies[rows, sn] - couplings.energies[rows, sj]),
        )
        others[int(nb)] = {
            "min_separation_ev": float(d.min()),
            "median_separation_ev": float(np.median(d)),
            "ratio_to_pair_min_gap": float(d.min() / pair_gap.min())
            if pair_gap.min() > 0 else float("inf"),
        }
    closest = min((v["min_separation_ev"] for v in others.values()), default=float("inf"))
    return {
        "band_i": band_i,
        "band_j": band_j,
        "window": [int(first_frame), int(last_frame)],
        "pair_min_gap_ev": float(pair_gap.min()),
        "pair_median_gap_ev": float(np.median(pair_gap)),
        "neighbours": others,
        "closest_neighbour_separation_ev": float(closest),
        "isolation_ratio": float(closest / pair_gap.min()) if pair_gap.min() > 0 else float("inf"),
        "note": (
            "isolation_ratio is the closest approach of any neighbouring state "
            "to the pair, divided by the pair's own minimum gap. A large ratio "
            "means the pair is far closer to each other than to anything else "
            "over this window, which is what a two-state reading assumes. It is "
            "reported rather than thresholded"
        ),
    }


def raw_weight_check(
    character,
    band_i: int,
    band_j: int,
    first_frame: int,
    last_frame: int,
    donor: str = "BCF",
    acceptor: str = "PCBM",
) -> Dict[str, Any]:
    """Does the exchange survive without the ``w_ig`` normalization?

    ``w_ig = W_ig / sum_g W_ig`` divides away whatever fell outside the
    declared spheres.  If the captured fraction moved sharply across the
    crossing, a swap in ``w`` could in principle be produced by the
    denominator rather than by the numerator.  This compares the normalized
    exchange against the raw ``W_ig``.
    """
    if character.captured is None:
        raise EpisodeError("the character table carries no captured_projection")
    band_at = character.band_index()
    frame_at = character.frame_index()
    gi, gj = character.groups.index(donor), character.groups.index(acceptor)
    frames = [f for f in range(int(first_frame), int(last_frame) + 1) if f in frame_at]
    rows = [frame_at[f] for f in frames]

    out: Dict[str, Any] = {"window": [int(first_frame), int(last_frame)], "bands": {}}
    for band in (band_i, band_j):
        b = band_at[band]
        w = character.weights[rows, b]
        cap = character.captured[rows, b]
        raw = w * cap[:, None]

        # A passage that exchanges character and exchanges back has the same
        # dominance at both ends of the window, so comparing endpoints finds
        # nothing. The question is whether the sign of (donor - acceptor)
        # reverses ANYWHERE inside it.
        def _swaps(series):
            diff = series[:, gi] - series[:, gj]
            sign = np.sign(diff)
            nonzero = sign[sign != 0]
            return (
                bool(nonzero.size and np.any(nonzero != nonzero[0])),
                int(np.argmax(np.abs(diff - diff[0]))),
            )

        norm_swap, norm_at = _swaps(w)
        raw_swap, raw_at = _swaps(raw)
        out["bands"][int(band)] = {
            "normalized_swaps": norm_swap,
            "raw_swaps": raw_swap,
            "agree": norm_swap == raw_swap,
            "frame_of_largest_normalized_excursion": int(frames[norm_at]),
            "frame_of_largest_raw_excursion": int(frames[raw_at]),
            "captured_min": float(cap.min()),
            "captured_max": float(cap.max()),
            "captured_relative_range": float(
                (cap.max() - cap.min()) / cap.mean()) if cap.mean() else float("nan"),
            f"raw_{donor}_first": float(raw[0, gi]),
            f"raw_{donor}_at_excursion": float(raw[raw_at, gi]),
            f"raw_{acceptor}_first": float(raw[0, gj]),
            f"raw_{acceptor}_at_excursion": float(raw[raw_at, gj]),
            f"normalized_{donor}_at_excursion": float(w[norm_at, gi]),
            f"normalized_{acceptor}_at_excursion": float(w[norm_at, gj]),
        }
    out["all_agree"] = all(v["agree"] for v in out["bands"].values())
    out["note"] = (
        "the exchange is confirmed against the raw PAW-sphere weights W_ig as "
        "well as the normalized w_ig. Where both swap, the normalization did "
        "not create the exchange. captured_relative_range says how much the "
        "denominator moved over the window: a small range means it could not "
        "have produced the swap on its own"
    )
    return out


def fixed_vs_dynamic_late(
    time_ns: np.ndarray,
    fixed: Dict[str, np.ndarray],
    dynamic: Dict[str, np.ndarray],
    start_ns: float = 0.1,
    end_ns: Optional[float] = None,
) -> Dict[str, Any]:
    """Does a 'flat population' conclusion survive the dynamic projection?

    ``fixed`` is the fixed-column reading, ``dynamic`` the projection-weighted
    one, both keyed by fragment.  Flatness is reported for each, so a
    conclusion drawn from one can be checked against the other.
    """
    time_ns = np.asarray(time_ns, dtype=float)
    mask = time_ns >= float(start_ns)
    if end_ns is not None:
        mask &= time_ns <= float(end_ns)
    if mask.sum() < 2:
        raise EpisodeError(
            f"the window [{start_ns}, {end_ns}] ns holds {int(mask.sum())} samples"
        )
    t = time_ns[mask]
    out: Dict[str, Any] = {
        "window_ns": [float(start_ns), float(end_ns) if end_ns is not None else float(t[-1])],
        "n_samples": int(mask.sum()),
        "groups": {},
    }
    for name in sorted(set(fixed) | set(dynamic)):
        entry: Dict[str, Any] = {}
        for label, series in (("fixed", fixed.get(name)), ("dynamic", dynamic.get(name))):
            if series is None:
                entry[label] = None
                continue
            y = np.asarray(series, dtype=float)[mask]
            slope = float(np.polyfit(t, y, 1)[0]) if t.size > 1 else float("nan")
            entry[label] = {
                "initial": float(y[0]),
                "final": float(y[-1]),
                "net_change": float(y[-1] - y[0]),
                "min": float(y.min()),
                "max": float(y.max()),
                "range": float(y.max() - y.min()),
                "slope_per_ns": slope,
                "mean": float(y.mean()),
            }
        if entry.get("fixed") and entry.get("dynamic"):
            entry["net_change_difference"] = (
                entry["dynamic"]["net_change"] - entry["fixed"]["net_change"]
            )
            entry["range_ratio_dynamic_over_fixed"] = (
                entry["dynamic"]["range"] / entry["fixed"]["range"]
                if entry["fixed"]["range"] > 0 else float("inf")
            )
        out["groups"][name] = entry
    out["note"] = (
        "a population that is flat under the fixed column map is not "
        "necessarily flat under the projection, because the fixed map assigns "
        "each column to a fragment for the whole run while the projection lets "
        "the assignment move with the geometry. Both readings are reported; "
        "neither is subtracted from the other"
    )
    return out


# ---------------------------------------------------------------------------
# Multistate crossing manifold
# ---------------------------------------------------------------------------
#
# A region where three or more states approach each other is not a sequence of
# independent two-state avoided crossings, and describing it as one discards
# the thing that makes it interesting.  These report the whole manifold: every
# state, every pairwise gap, both weight conventions, and the couplings, so a
# reader can see which states are actually in play.


def manifold_table(
    character,
    couplings,
    bands: Sequence[int],
    first_frame: int,
    last_frame: int,
    occupations: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[List[str], List[List[Any]]]:
    """Every state, gap, weight and coupling over a crossing region.

    ``occupations`` maps band number to a per-frame SHPROP occupation, when the
    histories are available.  Without it those columns are empty rather than
    filled with a stand-in.
    """
    bands = [int(b) for b in bands]
    band_at = character.band_index()
    frame_at = character.frame_index()
    state_of = {int(b): k for k, b in enumerate(character.bands)}
    row_of = (
        {int(f): r for r, f in enumerate(couplings.frames)} if couplings is not None else {}
    )
    missing = [b for b in bands if b not in band_at]
    if missing:
        raise EpisodeError(f"bands {missing} are not in the character table")

    pairs = [(a, b) for i, a in enumerate(bands) for b in bands[i + 1:]]
    header = ["frame"]
    header += [f"E_{b}_eV" for b in bands]
    header += [f"gap_{a}_{b}_eV" for a, b in pairs]
    header += [f"abs_nac_{a}_{b}_eV" for a, b in pairs]
    for b in bands:
        header += [f"{b}_{g}" for g in character.groups]
    for b in bands:
        header += [f"{b}_raw_{g}" for g in character.groups]
    header += [f"{b}_captured" for b in bands]
    header += [f"{b}_dominant" for b in bands]
    if occupations is not None:
        header += [f"P_{b}" for b in bands]

    def energy(row, state):
        ok = (
            row is not None and state is not None and couplings is not None
            and state < couplings.energies.shape[1]
        )
        return float(couplings.energies[row, state]) if ok else ""

    rows: List[List[Any]] = []
    for frame in range(int(first_frame), int(last_frame) + 1):
        if frame not in frame_at:
            continue
        k = frame_at[frame]
        r = row_of.get(frame)
        row: List[Any] = [frame]
        for b in bands:
            row.append(energy(r, state_of.get(b)))
        for a, b in pairs:
            ea, eb = energy(r, state_of.get(a)), energy(r, state_of.get(b))
            row.append(abs(ea - eb) if ea != "" and eb != "" else "")
        for a, b in pairs:
            sa, sb = state_of.get(a), state_of.get(b)
            ok = (
                r is not None and couplings is not None and couplings.nac is not None
                and None not in (sa, sb) and max(sa, sb) < couplings.nac.shape[1]
            )
            row.append(float(abs(couplings.nac[r, sa, sb])) if ok else "")
        for b in bands:
            row += [float(v) for v in character.weights[k, band_at[b]]]
        for b in bands:
            cap = (
                float(character.captured[k, band_at[b]])
                if character.captured is not None else float("nan")
            )
            row += [float(v) * cap for v in character.weights[k, band_at[b]]]
        for b in bands:
            row.append(
                float(character.captured[k, band_at[b]])
                if character.captured is not None else ""
            )
        for b in bands:
            row.append(_dominant(character, k, band_at[b]))
        if occupations is not None:
            for b in bands:
                series = occupations.get(int(b))
                row.append(float(series[k]) if series is not None else "")
        rows.append(row)
    return header, rows


def band_participation(
    character,
    couplings,
    band: int,
    pair: Tuple[int, int],
    first_frame: int,
    last_frame: int,
    donor: str = "BCF",
    acceptor: str = "PCBM",
    occupation: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Does ``band`` materially participate in the ``pair``'s transfer pathway?

    Participation needs two things at once, and the distinction matters: a
    state can be energetically in the thick of a manifold and still be
    irrelevant to *fragment* transfer, if its own character never moves between
    the two fragments in question.  Both are reported.
    """
    band_at = character.band_index()
    frame_at = character.frame_index()
    state_of = {int(b): k for k, b in enumerate(character.bands)}
    frames = [f for f in range(int(first_frame), int(last_frame) + 1) if f in frame_at]
    if not frames:
        raise EpisodeError("no frames in the requested window")
    rows = [frame_at[f] for f in frames]

    gi, gj = character.groups.index(donor), character.groups.index(acceptor)
    w = character.weights[rows, band_at[int(band)]]
    dominant = [_dominant(character, k, band_at[int(band)]) for k in rows]

    energetic: Dict[str, Any] = {}
    if couplings is not None:
        crow = {int(f): r for r, f in enumerate(couplings.frames)}
        cr = [crow[f] for f in frames if f in crow]
        s = state_of.get(int(band))
        sa, sb = state_of.get(int(pair[0])), state_of.get(int(pair[1]))
        if cr and None not in (s, sa, sb):
            sep = np.minimum(
                np.abs(couplings.energies[cr, s] - couplings.energies[cr, sa]),
                np.abs(couplings.energies[cr, s] - couplings.energies[cr, sb]),
            )
            pair_gap = np.abs(couplings.energies[cr, sa] - couplings.energies[cr, sb])
            energetic = {
                "min_separation_from_pair_ev": float(sep.min()),
                "median_separation_from_pair_ev": float(np.median(sep)),
                "pair_min_gap_ev": float(pair_gap.min()),
                "comes_closer_than_the_pair_does": bool(sep.min() < pair_gap.min()),
                "frame_of_closest_approach": int(frames[int(np.argmin(sep))]),
            }

    changes_fragment = bool(len(set(dominant)) > 1)
    between_the_two = bool(set(dominant) <= {donor, acceptor} and changes_fragment)

    out: Dict[str, Any] = {
        "band": int(band),
        "pair": [int(pair[0]), int(pair[1])],
        "window": [int(first_frame), int(last_frame)],
        "energetic": energetic,
        "character": {
            "dominant_fragments_seen": sorted(set(dominant)),
            "mean_weight": {
                g: float(w[:, n].mean()) for n, g in enumerate(character.groups)
            },
            f"{donor}_range": float(w[:, gi].max() - w[:, gi].min()),
            f"{acceptor}_range": float(w[:, gj].max() - w[:, gj].min()),
            "changes_dominant_fragment": changes_fragment,
            "exchanges_between_donor_and_acceptor": between_the_two,
        },
        "occupation": (
            {
                "mean": float(np.mean(occupation)),
                "max": float(np.max(occupation)),
                "net_change": float(occupation[-1] - occupation[0]),
            }
            if occupation is not None else None
        ),
    }

    near = bool(energetic.get("comes_closer_than_the_pair_does", False))
    if between_the_two:
        out["verdict"] = "participates_in_fragment_transfer"
        out["why"] = (
            f"band {band} itself exchanges dominance between {donor} and "
            f"{acceptor} over this window, so it is part of the transfer pathway"
        )
    elif near:
        out["verdict"] = "energetically_involved_only"
        out["why"] = (
            f"band {band} comes closer to the pair than the pair does to itself, "
            f"so a two-state treatment is not justified. But its own character "
            f"stays on {sorted(set(dominant))} and never moves between {donor} "
            f"and {acceptor}: it redistributes population *within* its own "
            "fragment and cannot by itself carry fragment transfer"
        )
    else:
        out["verdict"] = "not_involved"
        out["why"] = (
            f"band {band} neither approaches the pair more closely than the pair "
            "does itself, nor changes its dominant fragment"
        )
    if out["occupation"] is None:
        out["occupation_note"] = (
            "no SHPROP occupation was supplied, so whether this state actually "
            "carries population over the window is unknown. Energetic proximity "
            "and character are not occupation"
        )
    return out


def alignment_scan(
    character,
    couplings,
    band_i: int,
    band_j: int,
    donor: str = "BCF",
    acceptor: str = "PCBM",
    offsets: Sequence[int] = tuple(range(-8, 9)),
) -> Dict[str, Any]:
    """Confirm the PROCAR and EIGTXT frame axes are the same axis.

    At an avoided crossing, character mixing is maximal where the gap is
    minimal.  If the two files share a frame axis, the association between
    mixing and gap closure is strongest at offset zero and weaker either side;
    if they are shifted, the maximum sits elsewhere.

    The test is meaningful only on a pair carrying *different* fragments whose
    mixing actually varies, so it reports whether it could run at all rather
    than returning a number from a degenerate comparison.
    """
    band_at = character.band_index()
    state_of = {int(b): k for k, b in enumerate(character.bands)}
    si, sj = state_of.get(int(band_i)), state_of.get(int(band_j))
    if si is None or sj is None:
        raise EpisodeError("both bands must be in the character table")
    gi, gj = character.groups.index(donor), character.groups.index(acceptor)

    w = character.weights[:, band_at[int(band_i)]]
    mixing = np.minimum(w[:, gi], w[:, gj])
    if float(mixing.std()) <= 1e-12:
        return {
            "ran": False,
            "reason": (
                f"band {band_i} shows no variation in {donor}/{acceptor} mixing, "
                "so gap closure has nothing to correlate against"
            ),
        }
    gap = np.abs(couplings.energies[:, si] - couplings.energies[:, sj])
    n = int(min(mixing.size, gap.size))

    scores: Dict[int, float] = {}
    for shift in offsets:
        lo, hi = max(0, -shift), n - max(0, shift)
        if hi - lo < 16:
            continue
        a = mixing[lo:hi]
        b = gap[lo + shift: hi + shift]
        if a.std() <= 1e-12 or b.std() <= 1e-12:
            continue
        scores[int(shift)] = float(np.corrcoef(a, -np.log(b + 1e-12))[0, 1])
    if not scores:
        return {"ran": False, "reason": "no offset had enough overlapping samples"}

    order = sorted(scores, key=lambda k: -scores[k])
    best = order[0]
    second = order[1] if len(order) > 1 else None
    margin = scores[best] - scores[second] if second is not None else float("inf")
    return {
        "ran": True,
        "band_i": int(band_i),
        "band_j": int(band_j),
        "n_samples": n,
        "scores": scores,
        "best_offset": int(best),
        "score_at_best": scores[best],
        "score_at_zero": scores.get(0),
        "second_best_offset": second,
        "margin_over_second": float(margin),
        "aligned": bool(best == 0),
        "unique": bool(best == 0 and margin > 0),
        "note": (
            "mixing against -log(gap), scanned over integer frame offsets. A "
            "maximum at offset 0 means the PROCAR-derived character and the "
            "EIGTXT energies index the same frames. The margin over the "
            "next-best offset says how uniquely; a small margin is not a "
            "failure, but it is not a confirmation either"
        ),
    }
