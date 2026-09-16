"""Frame-dependent subsystem character from PROCAR projections.

An adiabatic band index can exchange BCF/PCBM/perovskite character during an
MD trajectory.  Projection-weighted populations therefore combine every
*original* SHPROP history with the PROCAR frame that actually generated that
NAMD step, and only then average across SHPROP files.

What this module produces is a *diagonal*, projection-weighted subsystem
population, not the exact subsystem population of the propagated state.  See
:data:`POPULATION_DEFINITION`.  Nothing here is called simply "the physical
population", because the two approximations behind it -- dropping the
electronic coherences and renormalizing PAW-sphere weight over the declared
groups -- are properties of the input files, not of the physics, and a reader
has to be able to see them.

The implementation is deliberately strict: no nearest-energy band tracking,
no inferred frame alignment, and no implicit k-point/spin averaging.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .io.hefei import (
    DEFAULT_CHUNK_ROWS,
    ShpropStructure,
    iter_shprop_chunks,
    shprop_structure,
)
from .io.procar import (
    ProcarFormatError,
    procar_band_numbers,
    procar_structure,
    read_procar_ion_totals,
)
from .memory_budget import BudgetError, estimate_memory
from .populations import (
    CONSERVATION_ATOL,
    ConfigError,
    StateMap,
    validate_band_numbers,
)
from .presets import bands_for_campaign_name
from .streaming import (
    ConservationTally,
    OnlineEnsemble,
    StreamingError,
    TimeGridCheck,
    chunk_plan,
)


class CharacterError(ValueError):
    """Raised when state-character analysis would require guessing."""


#: The quantity this module computes, stated in full.  Carried into every
#: report so the qualification travels with the numbers rather than living
#: only in documentation the reader may not open.
POPULATION_DEFINITION = (
    "Projection-weighted DIAGONAL subsystem population: "
    "P_g^(r)(t) = sum_i P_i^(r)(t) * w_ig[f_r(t)], formed for each original "
    "SHPROP history r and only then averaged over r. "
    "It is not the exact subsystem population of the propagated electronic "
    "state. Tr[rho(t) P_g] = sum_i rho_ii <i|P_g|i> + sum_{i!=j} rho_ij "
    "<j|P_g|i>; SHPROP records only the diagonal rho_ii, and a PROCAR records "
    "only diagonal, band-by-band projections, so the coherence term is absent "
    "from the inputs entirely. It is omitted, not estimated, and its size is "
    "not bounded by anything reported here. "
    "The weights are a second qualification: w_ig = W_ig / sum_g W_ig is the "
    "share of the PAW-sphere weight that fell inside the declared groups, not "
    "a fraction of the whole band. Interstitial and undeclared-atom weight is "
    "divided away by that normalization; captured_projection reports how much "
    "was there to begin with."
)

#: Short label for axes, column headers and one-line summaries.
POPULATION_LABEL = "projection-weighted diagonal subsystem population"

#: Above this, the per-history projected populations are not retained in the
#: result.  The ensemble mean and standard error never depend on them -- they
#: come from a running accumulator -- so dropping them costs a diagnostic
#: array, not a number anyone reports.
PER_FILE_MEMORY_BUDGET_BYTES = 256 * 1024 * 1024


@dataclass
class AtomGroupMap:
    """One-based PROCAR ion indices assigned to physical subsystems."""

    groups: Dict[str, List[int]]
    complete_atoms: bool = True
    min_projection_weight: float = 0.5

    @classmethod
    def from_json(cls, path) -> "AtomGroupMap":
        payload = _load_json(Path(path), "atom-group map")
        raw = payload.get("groups")
        if not isinstance(raw, dict) or not raw:
            raise CharacterError(
                f"{path}: atom-group JSON needs a nonempty 'groups' object"
            )
        complete = payload.get("complete_atoms", True)
        if type(complete) is not bool:
            raise CharacterError("complete_atoms must be a JSON boolean")
        groups = {str(name): _expand_atom_spec(spec) for name, spec in raw.items()}
        obj = cls(
            groups=groups,
            complete_atoms=complete,
            min_projection_weight=float(payload.get("min_projection_weight", 0.5)),
        )
        obj.validate()
        return obj

    def validate(self) -> None:
        if not (0.0 <= self.min_projection_weight <= 2.0):
            raise CharacterError("min_projection_weight must lie between 0 and 2")
        seen: Dict[int, str] = {}
        for name, atoms in self.groups.items():
            if not atoms:
                raise CharacterError(f"atom group {name!r} is empty")
            for atom in atoms:
                if atom <= 0:
                    raise CharacterError("PROCAR ion indices are one-based positive integers")
                if atom in seen:
                    raise CharacterError(
                        f"PROCAR ion {atom} appears in both {seen[atom]!r} and {name!r}"
                    )
                seen[atom] = name

    @property
    def names(self) -> List[str]:
        return list(self.groups)


@dataclass
class ProjectionManifest:
    """Resolved mapping between engine frame numbers and PROCAR files."""

    frames: List[Tuple[int, Path]]
    cycle_length: Optional[int] = None


@dataclass
class ProjectionSeries:
    """Normalized subsystem character for selected bands over MD frames."""

    frames: np.ndarray
    bands: np.ndarray
    group_names: List[str]
    weights: np.ndarray  # (nframe, nband, ngroup), normalized over declared groups
    captured_projection: np.ndarray  # raw sum over declared groups
    total_projection: np.ndarray  # raw sum over every PROCAR ion
    source_paths: List[Path]
    quality_threshold: float
    #: What was actually read: declared, required, parsed, skipped, bytes.
    parse_stats: Dict[str, Any] = field(default_factory=dict)
    cycle_length: Optional[int] = None
    #: Frame period actually used to align time points to frames, whether it
    #: came from the manifest or from ``NSW - 1``.  ``None`` for a linear
    #: analysis, or when the series was loaded without an alignment plan and
    #: so has no idea whether the frames form a ring.
    cycle_period: Optional[int] = None
    #: How many SHPROP histories actually take the ``cycle_period -> 1`` step.
    #: Counted from the resolved frame series, never inferred from coverage.
    #: ``None`` means nobody told this series, which is not the same as zero.
    cycle_wrap_histories: Optional[int] = None

    def frame_index(self) -> Dict[int, int]:
        return {int(frame): i for i, frame in enumerate(self.frames)}

    def band_index(self) -> Dict[int, int]:
        return {int(band): i for i, band in enumerate(self.bands)}

    def cycle_wrap_status(self) -> Dict[str, Any]:
        """Whether the ``period -> 1`` step is examined, and if not, why not.

        The loaded frames are held in ascending order, so a plain scan over
        consecutive entries never compares the last frame of a cycle with the
        first.  In a cyclic campaign that step is taken by the dynamics like
        any other, and a character change across it is a real change in the
        state the trajectory occupies.  It is still reported separately from
        an ordinary adjacent step, because the nuclear geometry does *not*
        evolve continuously there: the trajectory restarts at frame 1, so the
        change reflects the discontinuity of re-using one MD run cyclically.

        Every branch is labelled.  Silence about this edge is what the caller
        must never get.
        """
        frames = {int(frame) for frame in self.frames}
        status: Dict[str, Any] = {
            "examined": False,
            "period": self.cycle_period,
            "frame_before": None,
            "frame_after": None,
            "histories_traversing": self.cycle_wrap_histories,
        }
        if self.cycle_period is None:
            status["state"] = "not_cyclic_or_not_declared"
            status["note"] = (
                "no cyclic period is attached to this projection series, so the "
                "last-frame to first-frame step was not examined. A linear "
                "analysis has no such step; a cyclic one loaded without an "
                "alignment plan has one that could not be identified here"
            )
            return status
        period = int(self.cycle_period)
        status["frame_before"] = period
        status["frame_after"] = 1
        if period == 1:
            # Every time step uses frame 1; the "wrap" would compare a frame
            # with itself and could never find a change. Saying it was
            # examined would imply a check that carries no information.
            status["state"] = "degenerate_period"
            status["note"] = (
                "the cyclic period is 1, so every time point uses frame 1 and "
                "the wrap step would compare that frame with itself; there is "
                "no character change it could detect"
            )
            return status
        if self.cycle_wrap_histories is None:
            status["state"] = "traversal_unknown"
            status["note"] = (
                f"the cyclic period is {period}, but this series was not told "
                "how many histories step across it, so the wrap step is "
                "excluded rather than guessed from frame coverage"
            )
            return status
        if self.cycle_wrap_histories <= 0:
            status["state"] = "not_traversed"
            status["note"] = (
                f"no supplied history reaches frame {period} and continues, so "
                "the wrap step is not part of this campaign and is excluded"
            )
            return status
        if period not in frames or 1 not in frames:
            missing = sorted({period, 1} - frames)
            status["state"] = "frames_not_loaded"
            status["note"] = (
                f"{self.cycle_wrap_histories} history/histories cross the wrap, "
                f"but frame(s) {missing} are absent from the loaded projection, "
                "so the step cannot be evaluated"
            )
            return status
        status["examined"] = True
        status["state"] = "examined"
        status["note"] = (
            f"frame {period} -> frame 1 is one step of the dynamics, taken by "
            f"{self.cycle_wrap_histories} history/histories, and is compared. "
            "It is labelled 'cycle_wrap' rather than 'adjacent' because the "
            "nuclear geometry jumps there: the cyclic mapping restarts the MD "
            "trajectory instead of continuing it"
        )
        return status

    def examined_transitions(self) -> List[Dict[str, Any]]:
        """Every frame-to-frame step this analysis is able to compare.

        One definition, used by both the per-band dominance statistics and the
        swap table, so the two can never disagree about what a swap is.  Each
        entry carries the index pair into ``frames``/``weights``, the MD frame
        gap, and how well resolved the step is:

        ``adjacent``
            consecutive MD frames; a change here is located exactly.
        ``across_gap``
            the intervening MD frames were never loaded, so the change
            happened somewhere inside the gap, not in one step.
        ``cycle_wrap``
            the ``period -> 1`` step of a cyclic campaign; one step of the
            dynamics, but a discontinuity of the nuclear trajectory.
        """
        frames = [int(frame) for frame in self.frames]
        transitions: List[Dict[str, Any]] = []
        for i in range(1, len(frames)):
            gap = frames[i] - frames[i - 1]
            transitions.append(
                {
                    "before_index": i - 1,
                    "after_index": i,
                    "frame_before": frames[i - 1],
                    "frame_after": frames[i],
                    "frame_gap": gap,
                    "resolution": "adjacent" if gap == 1 else "across_gap",
                }
            )
        wrap = self.cycle_wrap_status()
        if wrap["examined"]:
            lookup = self.frame_index()
            transitions.append(
                {
                    "before_index": lookup[int(self.cycle_period)],
                    "after_index": lookup[1],
                    "frame_before": int(self.cycle_period),
                    "frame_after": 1,
                    # One step of the dynamics, whatever the frame numbers do.
                    "frame_gap": 1,
                    "resolution": "cycle_wrap",
                }
            )
        return transitions

    def quality_summary(self, worst_n: int = 10) -> Dict[str, Any]:
        """How much of each band actually lands inside the declared subsystems.

        The normalized weights divide by this, so a band whose captured
        projection is small has character that is mostly an artefact of the
        division.  These are diagnostics: nothing here discards, repairs or
        reweights a low-projection sample, because doing so silently would
        change the science.
        """
        captured = self.captured_projection
        threshold = self.quality_threshold
        below = captured < threshold
        nframes, nbands = captured.shape

        per_band: List[Dict[str, Any]] = []
        for bi in range(nbands):
            column = captured[:, bi]
            per_band.append(
                {
                    "band": int(self.bands[bi]),
                    "median_captured_projection": float(np.median(column)),
                    "minimum_captured_projection": float(np.min(column)),
                    "frame_of_minimum": int(self.frames[int(np.argmin(column))]),
                    "samples_below_threshold": int(np.count_nonzero(column < threshold)),
                    "fraction_below_threshold": float(np.mean(column < threshold)),
                }
            )

        per_frame_minimum = captured.min(axis=1)
        worst_frames = np.argsort(per_frame_minimum)[: max(0, worst_n)]

        flat = captured.reshape(-1)
        order = np.argsort(flat)[: max(0, worst_n)]
        worst_samples = []
        for position in order:
            fi, bi = divmod(int(position), nbands)
            worst_samples.append(
                {
                    "frame": int(self.frames[fi]),
                    "band": int(self.bands[bi]),
                    "captured_projection": float(captured[fi, bi]),
                    "total_projection": float(self.total_projection[fi, bi]),
                    "source": str(self.source_paths[fi]),
                }
            )

        return {
            "quality_threshold": threshold,
            "n_frames": int(nframes),
            "n_bands": int(nbands),
            "minimum_captured_projection": float(np.min(captured)),
            "p1_captured_projection": float(np.percentile(captured, 1)),
            "p5_captured_projection": float(np.percentile(captured, 5)),
            "median_captured_projection": float(np.median(captured)),
            "p95_captured_projection": float(np.percentile(captured, 95)),
            "maximum_captured_projection": float(np.max(captured)),
            "samples_below_threshold": int(np.count_nonzero(below)),
            "fraction_below_threshold": float(np.mean(below)),
            "per_band": per_band,
            "worst_frame_band_samples": worst_samples,
            "worst_frames_by_minimum_capture": [
                {
                    "frame": int(self.frames[int(fi)]),
                    "minimum_captured_projection": float(per_frame_minimum[int(fi)]),
                }
                for fi in worst_frames
            ],
            "note": (
                "captured_projection is the raw PROCAR weight falling inside the "
                "declared atom groups, before normalization. Values well below one "
                "mean the band lies largely outside every declared PAW sphere, so its "
                "normalized character is poorly determined. Nothing is discarded or "
                "repaired on this basis"
            ),
        }

    def swap_events(
        self, dominance_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """Every dominant-character change over every examined transition.

        The single source of truth for what counts as a swap.  Both the
        per-band dominance statistics and the swap table are built from this
        list, so a band's swap count and the swap rows for that band cannot
        drift apart, and neither can ignore a frame gap or a cycle wrap.
        """
        if not (0.0 < dominance_threshold <= 1.0):
            raise CharacterError("dominance_threshold must lie in (0, 1]")
        transitions = self.examined_transitions()
        events: List[Dict[str, Any]] = []
        for bi in range(self.weights.shape[1]):
            weights = self.weights[:, bi, :]
            dominant = np.argmax(weights, axis=1)
            confidence = np.max(weights, axis=1)
            for step in transitions:
                before = int(dominant[step["before_index"]])
                after = int(dominant[step["after_index"]])
                if before == after:
                    continue
                events.append(
                    {
                        "band_index": bi,
                        "band": int(self.bands[bi]),
                        "frame_before": step["frame_before"],
                        "frame_after": step["frame_after"],
                        "dominant_before": self.group_names[before],
                        "dominant_after": self.group_names[after],
                        "confidence_before": float(confidence[step["before_index"]]),
                        "confidence_after": float(confidence[step["after_index"]]),
                        "frame_gap": step["frame_gap"],
                        "resolution": step["resolution"],
                    }
                )
        return events

    def dominance_summary(self, dominance_threshold: float = 0.6) -> Dict[str, Any]:
        """Per-band dominant-character occupancy and swap counts.

        The swap counts here use the same transition set as
        :func:`character_swap_rows`: gaps left by selective loading are
        counted apart from resolved adjacent steps, and a cyclic campaign's
        ``period -> 1`` step is included when it is actually traversed.  Summing
        ``dominant_character_swaps`` over bands reproduces the campaign total
        in the swap summary exactly.
        """
        events = self.swap_events(dominance_threshold)
        by_band: Dict[int, List[Dict[str, Any]]] = {}
        for event in events:
            by_band.setdefault(event["band_index"], []).append(event)
        transitions = self.examined_transitions()
        bands_out: List[Dict[str, Any]] = []
        for bi in range(self.weights.shape[1]):
            weights = self.weights[:, bi, :]
            dominant = np.argmax(weights, axis=1)
            confidence = np.max(weights, axis=1)
            occupancy = {
                name: float(np.mean(dominant == gi))
                for gi, name in enumerate(self.group_names)
            }
            band_events = by_band.get(bi, [])
            counts = {"adjacent": 0, "across_gap": 0, "cycle_wrap": 0}
            for event in band_events:
                counts[event["resolution"]] += 1
            bands_out.append(
                {
                    "band": int(self.bands[bi]),
                    "dominant_occupancy_fraction": occupancy,
                    "most_common_character": max(occupancy, key=occupancy.get),
                    "dominant_character_swaps": len(band_events),
                    "adjacent_swaps": counts["adjacent"],
                    "changes_across_a_frame_gap": counts["across_gap"],
                    "cycle_wrap_swaps": counts["cycle_wrap"],
                    "median_dominant_weight": float(np.median(confidence)),
                    "mixed_samples": int(np.count_nonzero(confidence < dominance_threshold)),
                    "mixed_fraction": float(np.mean(confidence < dominance_threshold)),
                }
            )
        return {
            "dominance_threshold": dominance_threshold,
            "transitions_examined_per_band": len(transitions),
            "cycle_wrap": self.cycle_wrap_status(),
            "per_band": bands_out,
            "note": (
                "occupancy fractions are over the frames actually loaded, and "
                "swap counts use the same transition set as character_swaps.csv: "
                "adjacent, across_gap and cycle_wrap are counted separately and "
                "summing dominant_character_swaps over bands gives the campaign "
                "total reported there"
            ),
        }


@dataclass
class CharacterPopulationResult:
    """Projection-weighted diagonal subsystem populations, and how they were made.

    ``mean`` and ``per_file`` hold the quantity defined by
    :data:`POPULATION_DEFINITION`.  They are not the exact subsystem
    populations of the propagated state; the qualification is carried on the
    object as ``population_definition`` so anything writing a report can
    reproduce it without restating it from memory.
    """

    time_ns: np.ndarray
    group_names: List[str]
    #: Per-history projected populations, ``(nfiles, ntime, ngroups)``.  It is
    #: ``None`` when the campaign is large enough that retaining it would cost
    #: more than :data:`PER_FILE_MEMORY_BUDGET_BYTES`; the mean and SEM are
    #: accumulated online and do not depend on it.
    per_file: Optional[np.ndarray]  # (nfiles, ntime, ngroups)
    mean: np.ndarray
    sem: Optional[np.ndarray]
    fixed_mean: Dict[str, np.ndarray]
    projection: ProjectionSeries
    file_alignment: List[Dict[str, Any]]
    conservation: Dict[str, float]
    population_definition: str = POPULATION_DEFINITION
    #: How the SHPROP tables were read: chunk size, chunk count per history,
    #: and whether per-history results were retained.
    io: Dict[str, Any] = field(default_factory=dict)
    #: Accumulators still owning memory-mapped spill files, if any.  ``mean``
    #: and ``sem`` are views onto them, so a caller reads what it needs and
    #: then calls :meth:`release`.
    accumulators: List[Any] = field(default_factory=list, repr=False)

    def release(self) -> None:
        """Drop memory-mapped accumulators and delete their spill files.

        Call it once every output has been written. ``mean`` and ``sem`` must
        not be read afterwards when spill files were in use; nothing is
        deleted when the accumulators were ordinary arrays.
        """
        for accumulator in self.accumulators:
            accumulator.release()
        self.accumulators = []


def _load_json(path: Path, what: str) -> Any:
    """Read JSON, naming the file when it is malformed or unreadable."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise CharacterError(f"{path}: {what} could not be read ({exc})") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise CharacterError(
            f"{path}: {what} is not valid JSON - {exc.msg} at line {exc.lineno}, "
            f"column {exc.colno}"
        ) from exc


def _expand_atom_spec(spec: Any) -> List[int]:
    if not isinstance(spec, list):
        raise CharacterError("each atom group must be a JSON list of integers/ranges")
    atoms: List[int] = []
    for item in spec:
        if type(item) is int:
            atoms.append(item)
            continue
        if isinstance(item, str):
            text = item.strip()
            if "-" in text:
                left, right = text.split("-", 1)
                try:
                    start, stop = int(left), int(right)
                except ValueError as exc:
                    raise CharacterError(f"invalid atom range {item!r}") from exc
                if stop < start:
                    raise CharacterError(f"atom range decreases: {item!r}")
                atoms.extend(range(start, stop + 1))
                continue
            try:
                atoms.append(int(text))
                continue
            except ValueError as exc:
                raise CharacterError(f"invalid atom selector {item!r}") from exc
        raise CharacterError(f"invalid atom selector {item!r}")
    if len(set(atoms)) != len(atoms):
        raise CharacterError("an atom group contains duplicate ion indices")
    return atoms


def _load_projection_manifest(path) -> ProjectionManifest:
    manifest_path = Path(path)
    payload = _load_json(manifest_path, "projection manifest")
    base = manifest_path.parent
    entries: List[Tuple[int, Path]] = []

    raw_cycle = payload.get("cycle_length")
    cycle_length = None
    if raw_cycle is not None:
        if type(raw_cycle) is not int or raw_cycle <= 0:
            raise CharacterError("projection manifest cycle_length must be a positive integer")
        cycle_length = raw_cycle

    if "frames" in payload:
        if not isinstance(payload["frames"], list) or not payload["frames"]:
            raise CharacterError("projection manifest 'frames' must be a nonempty list")
        for record in payload["frames"]:
            try:
                frame = int(record["frame"])
                raw_path = str(record["procar"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CharacterError("each projection frame needs integer frame and procar") from exc
            procar = Path(raw_path)
            if not procar.is_absolute():
                procar = base / procar
            entries.append((frame, procar))
    elif "procar_pattern" in payload:
        try:
            first = int(payload["first_frame"])
            last = int(payload["last_frame"])
            step = int(payload.get("frame_step", 1))
            pattern = str(payload["procar_pattern"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CharacterError(
                f"{manifest_path}: a pattern manifest needs procar_pattern, "
                f"first_frame and last_frame ({exc})"
            ) from exc
        if step <= 0 or last < first:
            raise CharacterError(
                f"{manifest_path}: frame range first_frame={first}, "
                f"last_frame={last}, frame_step={step} is empty or invalid; "
                "frame_step must be >= 1 and last_frame >= first_frame"
            )
        # Without a {frame} field every frame would resolve to the same file,
        # and the analysis would report constant character as though it were
        # frame-dependent.  That is silent and unrecoverable, so refuse it.
        if "{frame" not in pattern:
            raise CharacterError(
                f"{manifest_path}: procar_pattern {pattern!r} contains no "
                "'{frame}' field, so every frame would resolve to the same "
                "PROCAR and the character would be constant while appearing "
                "frame-dependent. Add {frame} (for example 'frames/{frame:04d}/PROCAR')."
            )
        for frame in range(first, last + 1, step):
            try:
                rendered = pattern.format(frame=frame)
            except (KeyError, IndexError, ValueError) as exc:
                raise CharacterError(
                    f"{manifest_path}: procar_pattern {pattern!r} could not be "
                    f"expanded for frame {frame}: {type(exc).__name__} {exc}. "
                    "The only field this package substitutes is {frame}."
                ) from exc
            procar = Path(rendered)
            if not procar.is_absolute():
                procar = base / procar
            entries.append((frame, procar))
    else:
        raise CharacterError(
            f"{manifest_path}: a projection manifest needs either a 'frames' list "
            "or a 'procar_pattern' with first_frame/last_frame"
        )

    frames = [frame for frame, _ in entries]
    if any(frame <= 0 for frame in frames):
        raise CharacterError("projection frame numbers must be positive")
    if len(set(frames)) != len(frames):
        raise CharacterError("projection manifest contains duplicate frame numbers")
    entries.sort(key=lambda item: item[0])
    missing = [str(p) for _, p in entries if not p.is_file()]
    if missing:
        preview = ", ".join(missing[:3])
        more = "" if len(missing) <= 3 else f" (+{len(missing) - 3} more)"
        raise CharacterError(f"projection manifest references missing PROCAR files: {preview}{more}")

    if cycle_length is not None:
        available = set(frames)
        required_cycle = set(range(1, cycle_length + 1))
        missing_cycle = sorted(required_cycle - available)
        if missing_cycle:
            preview = missing_cycle[:10]
            raise CharacterError(
                f"cycle_length={cycle_length} but the manifest lacks cycle frames "
                f"{preview}{'...' if len(missing_cycle) > 10 else ''}"
            )
    return ProjectionManifest(frames=entries, cycle_length=cycle_length)


def load_projection_series(
    manifest_path,
    atom_groups: AtomGroupMap,
    required_bands: Sequence[int],
    required_frames: Optional[Sequence[int]] = None,
    manifest: Optional[ProjectionManifest] = None,
    cycle_period: Optional[int] = None,
    cycle_wrap_histories: Optional[int] = None,
) -> ProjectionSeries:
    """Load subsystem weights for exactly the bands needed by the NAMD basis.

    ``required_frames`` restricts parsing to the electronic frames the supplied
    SHPROP histories actually visit.  A cyclic campaign can declare two thousand
    PROCAR files and touch only a fraction of them, and each one is large, so
    parsing the whole manifest to use part of it is the dominant cost.  The
    manifest is still validated in full -- every declared file must exist, frame
    numbers must be unique and positive, and a declared ``cycle_length`` must be
    completely covered -- because those checks are cheap and dropping them to
    save time would let a gap go unnoticed until it silently changed a number.

    ``manifest`` lets a caller that already parsed the manifest pass it in
    rather than re-reading it.

    ``cycle_period`` and ``cycle_wrap_histories`` carry the alignment facts a
    plain projection load cannot know: the frame period actually used, and how
    many histories genuinely step from that frame back to frame 1.  They are
    what lets the swap diagnostics treat the frame sequence as a ring.  Left
    unset, the wrap step is excluded and labelled as unevaluated rather than
    inferred from which frames happen to be present.
    """
    required = [int(band) for band in required_bands]
    if not required or len(set(required)) != len(required):
        raise CharacterError("required band list must be nonempty and unique")
    manifest = manifest if manifest is not None else _load_projection_manifest(manifest_path)
    entries = manifest.frames
    if required_frames is not None:
        wanted = {int(frame) for frame in required_frames}
        if not wanted:
            raise CharacterError("required frame list must be nonempty")
        available = {frame for frame, _ in entries}
        missing = sorted(wanted - available)
        if missing:
            preview = missing[:10]
            raise CharacterError(
                f"projection manifest lacks {len(missing)} required electronic frames "
                f"{preview}{'...' if len(missing) > 10 else ''}; the manifest covers "
                f"{min(available)}..{max(available)} ({len(available)} frames). "
                "Add the missing PROCAR files, or correct NAMDTINI/frame-mode/cycle_length."
            )
        entries = [(frame, path) for frame, path in entries if frame in wanted]
    group_names = atom_groups.names
    frame_weights: List[np.ndarray] = []
    captured_rows: List[np.ndarray] = []
    total_rows: List[np.ndarray] = []
    nions_expected: Optional[int] = None

    first_path: Optional[Path] = None
    # Counters, so the report can state what was read rather than what was
    # declared. A 1999-frame manifest whose histories visit twenty frames must
    # be able to prove it parsed twenty files.
    parsed_paths: List[Path] = []
    bytes_read = 0
    for _frame, path in entries:
        parsed_paths.append(path)
        try:
            bytes_read += path.stat().st_size
        except OSError:
            pass
        try:
            projection = read_procar_ion_totals(path, bands=required)
        except ProcarFormatError as exc:
            raise CharacterError(str(exc)) from exc
        if nions_expected is None:
            nions_expected = projection.nions
            first_path = path
            flat_atoms = sorted(atom for atoms in atom_groups.groups.values() for atom in atoms)
            if flat_atoms and flat_atoms[-1] > nions_expected:
                raise CharacterError(
                    f"atom-group map references PROCAR ion {flat_atoms[-1]} but {path} "
                    f"has only {nions_expected} ions. Fix the atom-group JSON, or check "
                    "that it was written for this structure."
                )
            if atom_groups.complete_atoms:
                expected = list(range(1, nions_expected + 1))
                if flat_atoms != expected:
                    assigned = set(flat_atoms)
                    unassigned = sorted(set(expected) - assigned)
                    raise CharacterError(
                        "complete_atoms: true requires every PROCAR ion 1.."
                        f"{nions_expected} to appear exactly once across the atom "
                        f"groups. {path} has {nions_expected} ions; "
                        f"{len(assigned)} are assigned and {len(unassigned)} are not"
                        + (
                            f" (first missing: {unassigned[:10]}"
                            f"{'...' if len(unassigned) > 10 else ''})"
                            if unassigned
                            else ""
                        )
                        + ". Either complete the groups or set complete_atoms: false."
                    )
        elif projection.nions != nions_expected:
            raise CharacterError(
                f"{path}: {projection.nions} ions, but {first_path} has "
                f"{nions_expected}. Every PROCAR in one projection manifest must "
                "describe the same structure."
            )

        band_lookup = projection.band_index()
        missing_bands = [band for band in required if band not in band_lookup]
        if missing_bands:
            raise CharacterError(
                f"{path}: missing required VASP bands {missing_bands}. The NAMD basis "
                f"needs bands {required[0]}..{required[-1]}; widen the PROCAR band "
                "range or correct BMIN/BMAX."
            )

        raw = np.empty((len(required), len(group_names)), dtype=float)
        total = np.empty(len(required), dtype=float)
        for bi, band in enumerate(required):
            ion_values = projection.ion_totals[band_lookup[band]]
            total[bi] = float(np.sum(ion_values))
            for gi, group in enumerate(group_names):
                indices = np.asarray(atom_groups.groups[group], dtype=int) - 1
                raw[bi, gi] = float(np.sum(ion_values[indices]))
        captured = raw.sum(axis=1)
        if np.any(captured <= 1.0e-14):
            bad_index = np.flatnonzero(captured <= 1.0e-14)
            bad = [required[i] for i in bad_index]
            worst = float(np.min(captured))
            raise CharacterError(
                f"{path}: the declared subsystems capture essentially zero projection "
                f"(smallest {worst:.3g}, expected order 1) for VASP band(s) {bad[:10]}"
                f"{'...' if len(bad) > 10 else ''}; normalizing their character would "
                "divide by zero and produce meaningless weights. Either those bands lie "
                "outside the PAW spheres of every declared group, or the atom-group map "
                "does not match this structure."
            )
        frame_weights.append(raw / captured[:, None])
        captured_rows.append(captured)
        total_rows.append(total)

    return ProjectionSeries(
        frames=np.asarray([frame for frame, _ in entries], dtype=int),
        bands=np.asarray(required, dtype=int),
        group_names=group_names,
        weights=np.stack(frame_weights, axis=0),
        captured_projection=np.stack(captured_rows, axis=0),
        total_projection=np.stack(total_rows, axis=0),
        source_paths=[path for _, path in entries],
        quality_threshold=atom_groups.min_projection_weight,
        parse_stats={
            "manifest_declared_frames": len(manifest.frames),
            "frames_required": len(entries),
            "frames_parsed": len(parsed_paths),
            "frames_skipped": max(0, len(manifest.frames) - len(entries)),
            "unique_files_parsed": len({p.resolve() for p in parsed_paths}),
            "bytes_read": int(bytes_read),
            "bands_requested": len(required),
            "note": (
                "each required frame is parsed exactly once, however many histories "
                "visit it: the manifest is de-duplicated to the set of frames the "
                "plan needs before any file is opened. Only the requested bands are "
                "retained; other bands are walked so the file structure is still "
                "checked, but their projections are never stored"
            ),
        },
        cycle_length=manifest.cycle_length,
        cycle_period=cycle_period,
        cycle_wrap_histories=cycle_wrap_histories,
    )


def _int_metadata(metadata: Mapping[str, Any], key: str, path: Path) -> int:
    value = metadata.get(key)
    if type(value) is int:
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise CharacterError(
        f"{path}: SHPROP header lacks integer {key}; frame-dependent character "
        "cannot be aligned by inference"
    )


def aligned_frames(
    metadata: Mapping[str, Any],
    ntime: int,
    mode: str,
    path: Path,
    cycle_length: Optional[int] = None,
) -> np.ndarray:
    """Return the electronic-structure frame used at each NAMD time point.

    For ``dish-cyclic`` an explicit projection-manifest ``cycle_length`` wins
    over ``NSW - 1`` from the SHPROP header.  This is necessary for archived
    campaigns whose engine/input bookkeeping does not exactly match the number
    of saved electronic frames.  The chosen period is recorded in provenance.
    """
    start = _int_metadata(metadata, "NAMDTINI", path)
    if start < 1:
        raise CharacterError(
            f"{path}: NAMDTINI is {start}, but it is a one-based MD frame index "
            "and must be at least 1. A non-positive value shifts every frame "
            "assignment without changing anything that looks wrong downstream."
        )
    tion = np.arange(1, ntime + 1, dtype=int)
    if mode == "linear":
        return start + tion - 1
    if mode == "dish-cyclic":
        if cycle_length is None:
            nsw = _int_metadata(metadata, "NSW", path)
            period = nsw - 1
        else:
            period = cycle_length
        if period <= 0:
            raise CharacterError(f"{path}: cyclic frame period must be positive")
        frames = np.mod(tion + start - 1, period)
        frames[frames == 0] = period
        return frames
    raise CharacterError(f"unknown frame alignment mode {mode!r}")


@dataclass
class AnalysisPlan:
    """Everything decided before a single PROCAR is parsed.

    Building this is cheap: it reads the SHPROP headers and the projection
    manifest, and does all the arithmetic that decides which electronic frames
    are needed.  Both the preflight and the real run go through it, so what
    preflight checks is exactly what the run will do.
    """

    shprop_paths: List[Path]
    manifest: ProjectionManifest
    bmin: int
    bmax: int
    bands: List[int]
    frame_mode: str
    frames_by_file: List[np.ndarray]
    required_frames: List[int]
    alignments: List[Dict[str, Any]]
    n_time: List[int]
    manifest_frames: List[int]
    #: Header/shape records for each history, from the planning scan.  Kept so
    #: that the analysis pass does not re-scan a multi-hundred-megabyte file
    #: just to learn its column count.
    shprop_structures: List[ShpropStructure] = field(default_factory=list)
    #: Which provenance source supplied the exact VASP band numbers.
    band_numbers_source: str = "unresolved"
    #: Disagreements between the chosen basis and what the files themselves
    #: carry.  Reported, never silently resolved.
    band_provenance_conflicts: List[str] = field(default_factory=list)

    @property
    def manifest_frame_count(self) -> int:
        return len(self.manifest_frames)

    @property
    def required_frame_count(self) -> int:
        return len(self.required_frames)

    def missing_frames(self) -> List[int]:
        available = {frame for frame, _ in self.manifest.frames}
        return sorted(set(self.required_frames) - available)

    def cycle_period(self) -> Optional[int]:
        """The one frame period in force, or ``None`` if there is not exactly one.

        ``plan_analysis`` already refuses a cyclic campaign whose histories
        imply different periods, so more than one value here means the mode is
        linear or the period was never resolved.
        """
        if self.frame_mode != "dish-cyclic":
            return None
        periods = {record.get("cycle_length_used") for record in self.alignments}
        if len(periods) != 1:
            return None
        period = periods.pop()
        return int(period) if period is not None else None

    def cycle_wrap_histories(self) -> Optional[int]:
        """How many histories actually take the ``period -> 1`` step.

        Counted from the resolved per-history frame series, not deduced from
        which frames the union happens to contain: a campaign can load both
        frame 1 and frame ``period`` without any single history stepping
        between them.
        """
        period = self.cycle_period()
        if period is None:
            return None
        crossings = 0
        for frames in self.frames_by_file:
            values = np.asarray(frames, dtype=int)
            if values.size < 2:
                continue
            if np.any((values[:-1] == period) & (values[1:] == 1)):
                crossings += 1
        return crossings

    def consumption(self) -> Dict[str, Any]:
        """Declared manifest range against the subset actually consumed."""
        available = sorted(frame for frame, _ in self.manifest.frames)
        return {
            "manifest_declared_frames": len(available),
            "manifest_frame_min": available[0] if available else None,
            "manifest_frame_max": available[-1] if available else None,
            "frames_required_by_shprop": len(self.required_frames),
            "frames_required_min": self.required_frames[0] if self.required_frames else None,
            "frames_required_max": self.required_frames[-1] if self.required_frames else None,
            "procars_parsed": len(self.required_frames),
            "procars_skipped": max(0, len(available) - len(self.required_frames)),
            "note": (
                "only the electronic frames the supplied SHPROP histories actually "
                "visit are parsed; the whole manifest is still validated for "
                "existence, uniqueness and cycle coverage"
            ),
        }


# --------------------------------------------------------------------------
# Provenance: where the basis, the sampling origin and the period come from
# --------------------------------------------------------------------------
#
# Production SHPROP files are plain numeric tables. BMIN/BMAX are properties of
# the NAMD input, not guaranteed SHPROP fields, and NAMDTINI often survives only
# in the historical ``SHPROP.<start-frame>`` filename. Each quantity therefore
# has an ordered list of independent sources. Higher sources win; sources that
# are both present and disagree are refused rather than reconciled; and no
# source at all is refused rather than guessed.

#: ``SHPROP.<positive integer>`` -- the legacy way a history records its start.
_SHPROP_SUFFIX = re.compile(r"^SHPROP\.(\d+)$", re.IGNORECASE)


def _optional_int(value: Any) -> Optional[int]:
    """An integer, or ``None`` when the metadata key is absent or unusable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _required_int(value: Any, key: str, path: Path) -> Optional[int]:
    """An integer, ``None`` if the key is absent -- but an error if it is junk.

    Absent and unreadable are different things.  Treating ``NAMDTINI = 37.5``
    as absent silently demotes to the next provenance source, so a file that
    states something wrong is handled as though it stated nothing.  A key that
    is present must be usable.
    """
    if value is None:
        return None
    number = _optional_int(value)
    if number is None:
        raise CharacterError(
            f"{path}: SHPROP metadata has {key}={value!r}, which is not an integer. "
            "A key that is present must be readable: treating it as absent would "
            "silently fall through to a lower-precedence source and use a value "
            "this file contradicts."
        )
    return number


def _header_band_windows(
    records: Sequence[ShpropStructure],
) -> Tuple[Dict[Tuple[int, int], List[str]], List[str]]:
    """Optional BMIN/BMAX windows the files carry, and which files carry none."""
    windows: Dict[Tuple[int, int], List[str]] = {}
    without: List[str] = []
    for record in records:
        low = _required_int(record.metadata.get("BMIN"), "BMIN", record.path)
        high = _required_int(record.metadata.get("BMAX"), "BMAX", record.path)
        if low is None or high is None:
            without.append(record.path.name)
            continue
        if high < low:
            raise CharacterError(
                f"{record.path}: optional SHPROP metadata has BMAX={high} < "
                f"BMIN={low}; the basis window must increase"
            )
        windows.setdefault((low, high), []).append(record.path.name)
    return windows, without


def band_provenance_conflicts(
    bands: Sequence[int], source: str, records: Sequence[ShpropStructure]
) -> List[str]:
    """Where a chosen basis contradicts what the SHPROP files themselves say.

    The basis is the one quantity nothing downstream can detect as wrong: a
    plausible population comes out either way.  So when a higher-precedence
    source wins, the lower one is still read and compared, and a disagreement
    is reported rather than left invisible.  It is a warning, not an error --
    precedence is deliberate and an archived header can legitimately describe a
    different run -- but it is never silent.
    """
    if source == "SHPROP_BMIN_BMAX_metadata":
        return []
    windows, _without = _header_band_windows(records)
    chosen = (int(bands[0]), int(bands[-1]))
    conflicts: List[str] = []
    for window, names in sorted(windows.items()):
        if window == chosen and len(bands) == window[1] - window[0] + 1:
            continue
        sample = ", ".join(sorted(names)[:5]) + ("..." if len(names) > 5 else "")
        conflicts.append(
            f"basis {chosen[0]}..{chosen[1]} came from {source}, but "
            f"{len(names)} SHPROP file(s) carry BMIN/BMAX = {window[0]}:{window[1]} "
            f"({sample}). The higher-precedence source was used, as documented, but "
            "the two disagree about which bands this basis is. Confirm which "
            "describes the run that produced these histories."
        )
    return conflicts


def resolve_band_numbers(
    state_map: StateMap, records: Sequence[ShpropStructure]
) -> Tuple[List[int], str]:
    """Exact VASP band numbers for the basis, and where they came from.

    Precedence: ``state_map.band_numbers`` > registered campaign provenance >
    optional SHPROP ``BMIN``/``BMAX`` metadata that every history agrees on.
    """
    nstates = len(state_map.population_columns)

    def _checked(values: Sequence[Any], source: str, described: str) -> List[int]:
        try:
            return validate_band_numbers(values, nstates, source)
        except ConfigError as exc:
            raise CharacterError(
                f"{described} but the state map declares {nstates} population "
                f"columns ({list(state_map.population_columns)}). Character analysis "
                "needs exactly one declared population column per basis state, in "
                f"the same order. This is fixable in the state-map JSON. ({exc})"
            ) from exc

    if state_map.band_numbers is not None:
        return (
            _checked(
                state_map.band_numbers,
                "state_map band_numbers",
                f"state_map band_numbers declares {len(list(state_map.band_numbers))} bands",
            ),
            "state_map.band_numbers",
        )

    registered = bands_for_campaign_name(state_map.name)
    if registered is not None:
        bands, source = registered
        return (
            _checked(
                bands,
                source,
                f"campaign provenance {source} declares {len(bands)} bands "
                f"({bands[0]}:{bands[-1]})",
            ),
            source,
        )

    windows, without = _header_band_windows(records)

    if len(windows) > 1:
        parts = []
        for (low, high), names in sorted(windows.items(), key=lambda item: -len(item[1])):
            sample = ", ".join(names[:5]) + ("..." if len(names) > 5 else "")
            parts.append(f"{low}:{high} in {len(names)} file(s) ({sample})")
        raise CharacterError(
            f"SHPROP files carry {len(windows)} different optional BMIN/BMAX windows: "
            + "; ".join(parts)
            + ". They cannot all describe one basis. Declare band_numbers explicitly "
            "in the state map, or split the campaign by window."
        )
    if windows and without:
        sample = ", ".join(without[:5]) + ("..." if len(without) > 5 else "")
        raise CharacterError(
            f"{len(without)} SHPROP file(s) carry no BMIN/BMAX metadata while others "
            f"do ({sample}). A basis taken from only some of the histories would be "
            "an assumption about the rest; declare band_numbers explicitly in the "
            "state map instead."
        )
    if windows:
        (low, high) = next(iter(windows))
        bands = list(range(low, high + 1))
        return (
            _checked(
                bands,
                "SHPROP BMIN/BMAX metadata",
                f"SHPROP basis {low}:{high} has {len(bands)} states",
            ),
            "SHPROP_BMIN_BMAX_metadata",
        )

    raise CharacterError(
        "the exact VASP band numbers are not recoverable from these SHPROP files. "
        "BMIN/BMAX are properties of the NAMD input and need not appear in a SHPROP "
        "table, so nothing here can infer them. Supply them one of three ways: "
        "'band_numbers' in state_map.json (one per population column, in order), a "
        "registered campaign preset whose provenance names them, or SHPROP files "
        "whose optional metadata carries agreeing BMIN/BMAX."
    )


def resolve_namdtini(record: ShpropStructure) -> Tuple[int, str]:
    """One history's sampling origin, and where it came from.

    Precedence: optional SHPROP ``NAMDTINI`` metadata > a validated
    ``SHPROP.<integer>`` filename suffix.  When both exist they must agree:
    a disagreement means one of them describes a different run, and choosing
    either would silently shift every frame assignment.
    """
    header = _required_int(record.metadata.get("NAMDTINI"), "NAMDTINI", record.path)
    match = _SHPROP_SUFFIX.match(record.path.name)
    suffix = int(match.group(1)) if match else None

    if header is not None:
        if header < 1:
            raise CharacterError(
                f"{record.path}: NAMDTINI is {header}, but it is a one-based MD frame "
                "index and must be at least 1. A non-positive value shifts every "
                "frame assignment without changing anything that looks wrong."
            )
        if suffix is not None and suffix != header:
            raise CharacterError(
                f"{record.path}: SHPROP metadata says NAMDTINI={header} but the "
                f"filename suffix says {suffix}. Two independent provenance sources "
                "disagree about where this history starts, so frame alignment is "
                "refused rather than resolved in favour of either."
            )
        return header, "SHPROP_NAMDTINI_metadata"

    if suffix is not None:
        if suffix < 1:
            raise CharacterError(
                f"{record.path}: the filename suffix is {suffix}, but NAMDTINI is a "
                "one-based MD frame index and must be at least 1."
            )
        return suffix, "SHPROP_filename_suffix"

    raise CharacterError(
        f"{record.path}: NAMDTINI is absent from the SHPROP metadata and the filename "
        "is not of the form SHPROP.<positive integer>. The sampling origin decides "
        "which PROCAR frame every time step uses and is never guessed; supply the "
        "original history under its start-frame filename, or add the metadata."
    )


def resolve_cycle_period(
    record: ShpropStructure, manifest: "ProjectionManifest", frame_mode: str
) -> Tuple[Optional[int], str, Optional[int]]:
    """The cyclic period, its source, and the header-derived period for comparison.

    Precedence: explicit projection-manifest ``cycle_length`` > optional SHPROP
    ``NSW - 1``.  The manifest wins because an archived header can describe a
    different run length than the frames that were actually saved; the
    disagreement is recorded, never averaged away.
    """
    raw_nsw = _required_int(record.metadata.get("NSW"), "NSW", record.path)
    header_period = raw_nsw - 1 if raw_nsw is not None else None
    if frame_mode != "dish-cyclic":
        return None, "not_applicable", header_period
    if manifest.cycle_length is not None:
        return int(manifest.cycle_length), "projection_manifest", header_period
    if header_period is not None and header_period > 0:
        return header_period, "SHPROP_NSW_minus_1_metadata", header_period
    raise CharacterError(
        f"{record.path}: cyclic alignment needs a frame period. The projection "
        "manifest declares no cycle_length and this SHPROP carries no usable NSW "
        "metadata, and a period is never inferred from how many frames happen to "
        "exist. Set cycle_length in projection_manifest.json."
    )


def plan_analysis(
    shprop_paths: Sequence,
    state_map: StateMap,
    projection_manifest,
    frame_mode: str,
) -> AnalysisPlan:
    """Resolve band window and required electronic frames without parsing PROCARs."""
    paths = [Path(path) for path in shprop_paths]
    if not paths:
        raise CharacterError("no SHPROP files were supplied")
    # The state map is a declaration, and an inconsistent one would mislabel
    # every number downstream. load_population_set used to validate it on this
    # path; the streaming rewrite no longer goes through that function, so the
    # check is made here rather than lost.
    try:
        state_map.validate()
    except ValueError as exc:
        raise CharacterError(str(exc)) from exc
    if len({path.resolve() for path in paths}) != len(paths):
        raise CharacterError("duplicate population files")
    # Headers and shape only. An archived history is routinely hundreds of
    # megabytes, and planning needs the band window, NAMDTINI and the row
    # count -- not a single population value.
    records = [shprop_structure(path) for path in paths]

    # Every check below is cheap and happens before a single PROCAR is opened.
    # A campaign can declare two thousand projection files; failing on a
    # duplicate path or a stray column after parsing them all would waste the
    # whole run to report something knowable in milliseconds.
    widths: Dict[int, List[str]] = {}
    heights: Dict[int, List[str]] = {}
    for record in records:
        widths.setdefault(record.n_columns, []).append(record.path.name)
        heights.setdefault(record.n_rows, []).append(record.path.name)
    for label, grouped, unit in (
        ("column counts", widths, "column"),
        ("row counts", heights, "row"),
    ):
        if len(grouped) == 1:
            continue
        parts = []
        for size, names in sorted(grouped.items(), key=lambda item: -len(item[1])):
            sample = ", ".join(names[:5]) + ("..." if len(names) > 5 else "")
            parts.append(f"{size} {unit}(s) in {len(names)} file(s) ({sample})")
        raise CharacterError(
            f"SHPROP files have different {label}: "
            + "; ".join(parts)
            + ". No interpolation or truncation is performed."
        )
    ncolumns = next(iter(widths))
    if next(iter(heights)) < 2:
        raise CharacterError(
            "population time must strictly increase with at least two samples"
        )

    # The basis is provenance, not something a SHPROP table reveals. One
    # resolver decides it and says which source won; see resolve_band_numbers.
    bands, band_source = resolve_band_numbers(state_map, records)
    bmin, bmax = bands[0], bands[-1]
    # The basis is the one quantity nothing downstream can detect as wrong, so
    # a higher-precedence source is still compared with what the files say.
    band_conflicts = band_provenance_conflicts(bands, band_source, records)
    # Only now: a column index past the end of the table. Checked after the
    # basis-size comparison because a state map with the wrong number of states
    # usually also overruns, and "you declared 3 states for a 2-state basis" is
    # the fixable statement, not "column 4 of 4".
    needed_column = max([state_map.time_column] + list(state_map.population_columns))
    if needed_column >= ncolumns:
        raise CharacterError(
            f"configuration references column {needed_column} but the SHPROP files "
            f"have {ncolumns} columns"
        )

    manifest = _load_projection_manifest(projection_manifest)
    manifest_frames = sorted(frame for frame, _ in manifest.frames)

    frames_by_file: List[np.ndarray] = []
    alignments: List[Dict[str, Any]] = []
    n_time: List[int] = []
    periods_used: List[Optional[int]] = []
    for record in records:
        ntime = int(record.n_rows)
        n_time.append(ntime)
        # Both the origin and the period come from resolvers that say which
        # source they used, so the alignment audit records provenance rather
        # than re-deriving it afterwards from whatever the filename looks like.
        start, start_source = resolve_namdtini(record)
        period, period_source, header_period = resolve_cycle_period(
            record, manifest, frame_mode
        )
        periods_used.append(period)
        frames = aligned_frames(
            {"NAMDTINI": start, "NSW": record.metadata.get("NSW")},
            ntime,
            frame_mode,
            record.path,
            cycle_length=period,
        )
        frames_by_file.append(frames)
        alignments.append(
            _alignment_record(
                record,
                frames,
                frame_mode,
                manifest,
                bands,
                band_source,
                ntime,
                start=start,
                start_source=start_source,
                period=period,
                period_source=period_source,
                header_period=header_period,
            )
        )

    required = sorted({int(frame) for frames in frames_by_file for frame in frames})
    if frame_mode == "dish-cyclic" and manifest.cycle_length is None:
        periods = set(periods_used)
        if len(periods) > 1:
            raise CharacterError(
                "the SHPROP histories imply different cyclic periods from their "
                f"own NSW headers ({sorted(p for p in periods if p is not None)}), "
                "so each file would wrap on a different cycle and the ensemble "
                "average would mix inconsistent frame mappings. Split the campaign "
                "by period, or declare one explicit cycle_length in the projection "
                "manifest if you know which is correct."
            )
    return AnalysisPlan(
        shprop_paths=paths,
        manifest=manifest,
        bmin=bmin,
        bmax=bmax,
        bands=bands,
        frame_mode=frame_mode,
        frames_by_file=frames_by_file,
        required_frames=required,
        alignments=alignments,
        n_time=n_time,
        manifest_frames=manifest_frames,
        shprop_structures=records,
        band_numbers_source=band_source,
        band_provenance_conflicts=band_conflicts,
    )


def _alignment_record(
    record,
    frames: np.ndarray,
    frame_mode: str,
    manifest: ProjectionManifest,
    bands: Sequence[int],
    band_source: str,
    ntime: int,
    start: int,
    start_source: str,
    period: Optional[int],
    period_source: str,
    header_period: Optional[int],
) -> Dict[str, Any]:
    """Human-readable audit of how one SHPROP maps onto electronic frames.

    Every summary here is computed on the integer array rather than on a
    Python list built from it.  A six-million-row history costs about 46 bytes
    per row as a list of boxed ints -- more than the array it summarizes -- and
    preflight claims its cost does not grow with file size.
    """
    cyclic = frame_mode == "dish-cyclic"
    values = np.asarray(frames)
    # A wrap is a step that does not simply increase by one.
    wraps = int(np.count_nonzero(np.diff(values) != 1)) if values.size > 1 else 0
    return {
        "path": str(record.path.resolve()),
        "file": record.path.name,
        "NAMDTINI": int(start),
        "NAMDTINI_source": start_source,
        "NSW": record.metadata.get("NSW"),
        "n_time_points": ntime,
        "header_cycle_length": header_period,
        "cycle_length_used": int(period) if cyclic and period is not None else None,
        "cycle_length_source": period_source,
        "header_cycle_mismatch": bool(
            cyclic
            and period is not None
            and header_period is not None
            and period != header_period
        ),
        "BMIN": int(bands[0]),
        "BMAX": int(bands[-1]),
        "band_numbers": [int(b) for b in bands],
        "band_numbers_source": band_source,
        "frame_mode": frame_mode,
        "first_projection_frame": int(values[0]),
        "last_projection_frame": int(values[-1]),
        "first_five_frames": [int(v) for v in values[:5]],
        "last_five_frames": [int(v) for v in values[-5:]],
        "unique_projection_frames_used": int(np.unique(values).size),
        "wrap_count": wraps,
    }


def preflight_report(
    shprop_paths: Sequence,
    state_map: StateMap,
    projection_manifest,
    atom_groups: AtomGroupMap,
    frame_mode: str,
    memory_budget: Optional[int] = None,
    retain_per_file: str = "auto",
    memmap_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Answer the cheap questions before the expensive analysis runs.

    Reads SHPROP headers, the projection manifest, and the *header* of one
    representative PROCAR.  It parses no projection data and computes no
    populations, so it stays fast enough to run every time.

    Problems are collected rather than raised, so a single run tells you
    everything that is wrong instead of only the first thing.
    """
    problems: List[str] = []
    warnings: List[str] = []
    payload: Dict[str, Any] = {
        "frame_mode": frame_mode,
        "n_shprop_files": len(list(shprop_paths)),
    }

    try:
        plan = plan_analysis(shprop_paths, state_map, projection_manifest, frame_mode)
    except CharacterError as exc:
        payload["problems"] = [str(exc)]
        payload["warnings"] = warnings
        payload["ok"] = False
        payload["stage"] = "planning"
        payload["note"] = (
            "preflight stopped while resolving the band window and frame alignment; "
            "nothing downstream could be checked"
        )
        return payload

    payload["stage"] = "planned"
    payload["shprop"] = [
        {
            "file": record["file"],
            "path": record["path"],
            "NAMDTINI": record["NAMDTINI"],
            "NSW": record["NSW"],
            "n_time_points": record["n_time_points"],
            "header_cycle_length": record["header_cycle_length"],
            "cycle_length_used": record["cycle_length_used"],
            "cycle_length_source": record["cycle_length_source"],
            "header_cycle_mismatch": record["header_cycle_mismatch"],
            "NAMDTINI_source": record["NAMDTINI_source"],
            "first_five_frames": record["first_five_frames"],
            "last_five_frames": record["last_five_frames"],
            "unique_projection_frames_used": record["unique_projection_frames_used"],
            "wrap_count": record["wrap_count"],
        }
        for record in plan.alignments
    ]
    first_alignment = plan.alignments[0] if plan.alignments else {}
    payload["band_window"] = {
        "BMIN": plan.bmin,
        "BMAX": plan.bmax,
        "basis_size": len(plan.bands),
        "band_numbers": list(plan.bands),
        "band_numbers_source": first_alignment.get("band_numbers_source"),
        "state_map_population_columns": list(state_map.population_columns),
        "note": (
            "exact VASP band numbers and the provenance they came from. They are "
            "not inferred from a SHPROP table: a production SHPROP need not carry "
            "BMIN/BMAX at all"
        ),
    }
    warnings.extend(plan.band_provenance_conflicts)
    payload["provenance"] = {
        "band_numbers": {
            "value": list(plan.bands),
            "source": first_alignment.get("band_numbers_source"),
            "conflicts_with_shprop_metadata": list(plan.band_provenance_conflicts),
        },
        "NAMDTINI": [
            {"file": r["file"], "value": r["NAMDTINI"], "source": r["NAMDTINI_source"]}
            for r in plan.alignments
        ],
        "cycle_period": [
            {
                "file": r["file"],
                "value": r["cycle_length_used"],
                "source": r["cycle_length_source"],
                "header_derived": r["header_cycle_length"],
                "disagrees_with_header": r["header_cycle_mismatch"],
            }
            for r in plan.alignments
        ],
        "precedence": {
            "band_numbers": "state_map.band_numbers > registered campaign > agreeing SHPROP BMIN/BMAX",
            "NAMDTINI": "SHPROP metadata > validated SHPROP.<integer> filename suffix",
            "cycle_period": "projection_manifest cycle_length > SHPROP NSW-1 metadata",
        },
    }
    payload["distinct_namdtini"] = sorted({r["NAMDTINI"] for r in plan.alignments})
    payload["row_counts"] = sorted(set(plan.n_time))
    payload["frames"] = plan.consumption()
    chunk_rows, io_plan = resolve_chunk_rows(plan.shprop_structures)
    payload["shprop_io"] = {
        **io_plan,
        "total_bytes": sum(r.bytes_on_disk for r in plan.shprop_structures),
        "per_file": [
            {
                "file": r.path.name,
                "rows": r.n_rows,
                "columns": r.n_columns,
                "bytes": r.bytes_on_disk,
            }
            for r in plan.shprop_structures
        ],
        "preflight_note": (
            "preflight read only headers, row counts and the first and last row of "
            "each history; no SHPROP table was materialized, so its memory use does "
            "not grow with file size. It therefore checks structure -- column "
            "count, row count, ragged rows -- but NOT every value: a history whose "
            "populations leave [0,1] or stop summing to one somewhere in the middle "
            "passes preflight and is refused by the analysis, which validates every "
            "row of every chunk"
        ),
    }

    # What the full run would allocate, decided before any PROCAR is opened so
    # that a campaign too large for the node fails here in seconds.
    try:
        memory = estimate_memory(
            n_files=len(plan.shprop_paths),
            n_time=plan.n_time[0],
            n_states=len(plan.bands),
            n_groups=len(atom_groups.names),
            n_fixed_groups=len(state_map.groups),
            n_frames=len(plan.required_frames),
            n_bands=len(plan.bands),
            n_columns=plan.shprop_structures[0].n_columns,
            chunk_rows=chunk_rows,
            budget=memory_budget,
            retain_per_file=retain_per_file,
            memmap_dir=memmap_dir,
        )
        payload["memory"] = memory.as_dict()
        problems.extend(memory.problems)
    except BudgetError as exc:
        problems.append(str(exc))
        payload["memory"] = {"error": str(exc)}

    explicit = plan.manifest.cycle_length
    header_periods = sorted(
        {r["header_cycle_length"] for r in plan.alignments if r["header_cycle_length"] is not None}
    )
    payload["cycle"] = {
        "explicit_manifest_cycle_length": explicit,
        "header_derived_periods_nsw_minus_1": header_periods,
        "disagree": bool(
            explicit is not None and header_periods and [explicit] != header_periods
        ),
        "applies_to_this_frame_mode": frame_mode == "dish-cyclic",
        # Says in advance whether the swap diagnostics will examine the
        # period -> 1 step, so a surprise there is caught before the run.
        "period_used": plan.cycle_period(),
        "histories_crossing_the_wrap": plan.cycle_wrap_histories(),
    }
    if payload["cycle"]["disagree"] and frame_mode == "dish-cyclic":
        warnings.append(
            f"projection manifest declares cycle_length={explicit} but the SHPROP "
            f"headers imply NSW-1={header_periods}. The explicit manifest value wins "
            "and the mismatch is recorded in the alignment report; confirm it is "
            "deliberate before trusting the frame mapping."
        )

    missing = plan.missing_frames()
    if missing:
        available = sorted(frame for frame, _ in plan.manifest.frames)
        problems.append(
            f"the projection manifest lacks {len(missing)} electronic frames that the "
            f"SHPROP histories require, first {missing[:10]}"
            f"{'...' if len(missing) > 10 else ''}. The manifest covers "
            f"{available[0]}..{available[-1]}. Fixable by adding those PROCAR files, "
            "or by correcting NAMDTINI, --frame-mode or cycle_length."
        )
    payload["missing_required_frames"] = missing[:50]
    payload["n_missing_required_frames"] = len(missing)

    # One representative PROCAR: header only.
    representative = None
    lookup = dict(plan.manifest.frames)
    for frame in plan.required_frames:
        if frame in lookup:
            representative = (frame, lookup[frame])
            break
    if representative is None:
        problems.append("no required frame is present in the manifest at all")
    else:
        frame, path = representative
        try:
            structure = procar_structure(path)
        except ProcarFormatError as exc:
            problems.append(f"representative PROCAR could not be read: {exc}")
        else:
            structure["frame"] = frame
            payload["representative_procar"] = structure
            if not structure["single_kpoint"]:
                problems.append(
                    f"{path}: {structure['n_kpoints']} k-points. Character analysis "
                    "requires one and will not select or average implicitly. This is "
                    "a property of the source data, not of the configuration."
                )
            if not structure["single_spin_block"]:
                problems.append(
                    f"{path}: multiple spin components "
                    f"{structure['spin_components_seen']}. Splitting or combining them "
                    "must be explicit upstream; the source data is incompatible as is."
                )
            nions = structure["n_ions"]
            declared = sorted(
                atom for atoms in atom_groups.groups.values() for atom in atoms
            )
            payload["atom_coverage"] = {
                "procar_ions": nions,
                "assigned_ions": len(declared),
                "complete_atoms": atom_groups.complete_atoms,
                "max_declared_ion": declared[-1] if declared else None,
            }
            if declared and nions is not None and declared[-1] > nions:
                problems.append(
                    f"the atom-group map references PROCAR ion {declared[-1]} but "
                    f"{path} has only {nions}. Fixable in the atom-group JSON."
                )
            elif atom_groups.complete_atoms and nions is not None:
                unassigned = sorted(set(range(1, nions + 1)) - set(declared))
                payload["atom_coverage"]["unassigned_ions"] = unassigned[:20]
                payload["atom_coverage"]["n_unassigned_ions"] = len(unassigned)
                if unassigned:
                    problems.append(
                        f"complete_atoms is true but {len(unassigned)} of the {nions} "
                        f"PROCAR ions are in no group, first {unassigned[:10]}"
                        f"{'...' if len(unassigned) > 10 else ''}. Either assign them "
                        "or set complete_atoms: false. Fixable in the atom-group JSON."
                    )
            bands_declared = structure["n_bands"]
            # Read the band numbers rather than assuming they run 1..NBANDS:
            # comparing a required band *number* against a band *count* would
            # be wrong for any file that labels its blocks differently.
            present = set(procar_band_numbers(path, limit=bands_declared))
            absent = [band for band in plan.bands if band not in present]
            payload["band_coverage"] = {
                "procar_declared_band_count": bands_declared,
                "procar_band_numbers_seen": sorted(present)[:20],
                "required_bands": [plan.bands[0], plan.bands[-1]],
                "required_bands_absent": absent,
            }
            if absent:
                problems.append(
                    f"the NAMD basis needs VASP band(s) {absent[:10]}"
                    f"{'...' if len(absent) > 10 else ''} but {path} labels bands "
                    f"{sorted(present)[:10]}{'...' if len(present) > 10 else ''}. "
                    "Either BMIN/BMAX is wrong, or the PROCAR was written over too "
                    "narrow a band range."
                )

    payload["problems"] = problems
    payload["warnings"] = warnings
    payload["ok"] = not problems
    payload["note"] = (
        "preflight reads SHPROP headers, the projection manifest and one PROCAR "
        "header. It parses no projection data and produces no subsystem populations."
    )
    return payload


#: A history smaller than this is read in one chunk under ``auto``: splitting
#: it buys nothing and the whole table is a few megabytes.
AUTO_SINGLE_CHUNK_BYTES = 64 * 1024 * 1024

SHPROP_IO_MODES = ("auto", "stream", "memory")


def resolve_chunk_rows(
    structures: Sequence[ShpropStructure],
    io_mode: str = "auto",
    chunk_rows: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Decide how many rows to read at a time, and say why.

    There is only one code path.  ``memory`` is not a second implementation of
    the analysis; it is a chunk large enough to hold the whole table at once,
    which is what the old whole-file reader did.  Keeping it that way means the
    streaming and in-memory answers are identical by construction rather than
    by two implementations agreeing.
    """
    if io_mode not in SHPROP_IO_MODES:
        raise CharacterError(
            f"unknown --shprop-io-mode {io_mode!r}; choose one of {list(SHPROP_IO_MODES)}"
        )
    if chunk_rows is not None and chunk_rows < 1:
        raise CharacterError("--shprop-chunk-rows must be at least 1")
    rows = max((record.n_rows for record in structures), default=1)
    largest = max((record.bytes_on_disk for record in structures), default=0)
    if io_mode == "memory":
        chosen = max(rows, 1)
        reason = "whole table in one chunk, as requested"
    elif chunk_rows is not None:
        chosen = chunk_rows
        reason = "explicit --shprop-chunk-rows"
    elif io_mode == "stream":
        chosen = DEFAULT_CHUNK_ROWS
        reason = "default streaming chunk"
    elif largest <= AUTO_SINGLE_CHUNK_BYTES:
        chosen = max(rows, 1)
        reason = (
            f"largest history is {largest} bytes, at or under the "
            f"{AUTO_SINGLE_CHUNK_BYTES}-byte single-chunk threshold"
        )
    else:
        chosen = DEFAULT_CHUNK_ROWS
        reason = (
            f"largest history is {largest} bytes, above the "
            f"{AUTO_SINGLE_CHUNK_BYTES}-byte single-chunk threshold"
        )
    return chosen, {
        "requested_mode": io_mode,
        "requested_chunk_rows": chunk_rows,
        "chunk_rows": int(chosen),
        "rows_per_history": int(rows),
        "largest_history_bytes": int(largest),
        "reason": reason,
        "note": (
            "chunking changes only how much of a history is resident at once; "
            "the ensemble mean, standard error and every validation are "
            "identical at any chunk size"
        ),
    }


def character_populations(
    shprop_paths: Sequence,
    state_map: StateMap,
    projection_manifest,
    atom_groups: AtomGroupMap,
    frame_mode: str,
    plan: Optional[AnalysisPlan] = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    memmap_dir: Optional[Path] = None,
    keep_per_file: Optional[bool] = None,
) -> CharacterPopulationResult:
    """Projection-weight each original SHPROP history, then average.

    Produces the diagonal subsystem population of
    :data:`POPULATION_DEFINITION`, not the exact subsystem population: the
    coherences SHPROP does not record are absent from the result, and the
    weights are normalized over the declared groups.

    The ordering is essential.  Different SHPROP histories may have different
    NAMDTINI values, so averaging their adiabatic populations first destroys
    the phase needed to select the correct PROCAR frame.
    """
    paths = [Path(path) for path in shprop_paths]
    if plan is None:
        plan = plan_analysis(paths, state_map, projection_manifest, frame_mode)
    else:
        # The plan carries one frame series per file, zipped against the files
        # below. A plan built from different paths, or a different mode, would
        # pair the wrong frames with the wrong history silently.
        planned = [Path(item).resolve() for item in plan.shprop_paths]
        given = [item.resolve() for item in paths]
        if planned != given:
            raise CharacterError(
                "the supplied AnalysisPlan was built from different SHPROP files "
                f"than were passed: plan has {[p.name for p in plan.shprop_paths]}, "
                f"call has {[p.name for p in paths]}. Frames would be paired with "
                "the wrong history."
            )
        if plan.frame_mode != frame_mode:
            raise CharacterError(
                f"the supplied AnalysisPlan was built for frame mode "
                f"{plan.frame_mode!r} but {frame_mode!r} was requested"
            )
    bands = plan.bands
    if chunk_rows < 1:
        raise CharacterError("shprop_chunk_rows must be at least 1")

    projection = load_projection_series(
        projection_manifest,
        atom_groups,
        bands,
        required_frames=plan.required_frames,
        manifest=plan.manifest,
        cycle_period=plan.cycle_period(),
        cycle_wrap_histories=plan.cycle_wrap_histories(),
    )
    frame_lookup = projection.frame_index()
    band_lookup = projection.band_index()
    projection_band_indices = np.asarray([band_lookup[band] for band in bands], dtype=int)
    selected = projection.weights[:, projection_band_indices, :]

    group_names = projection.group_names
    fixed_names = list(state_map.groups)
    ntime = plan.n_time[0]

    projected_acc = OnlineEnsemble(
        ntime, len(group_names), memmap_dir=memmap_dir, name="projected"
    )
    fixed_acc = OnlineEnsemble(ntime, len(fixed_names), memmap_dir=memmap_dir, name="fixed")
    grid = TimeGridCheck(reference=np.empty(ntime, dtype=float))
    tally = ConservationTally(atol=CONSERVATION_ATOL)
    columns = np.asarray(state_map.population_columns, dtype=int)
    fixed_columns = [np.asarray(state_map.groups[name], dtype=int) for name in fixed_names]

    keep = keep_per_file
    if keep is None:
        keep = len(paths) * ntime * len(group_names) * 8 <= PER_FILE_MEMORY_BUDGET_BYTES
    per_file = (
        np.empty((len(paths), ntime, len(group_names)), dtype=float) if keep else None
    )

    alignments: List[Dict[str, Any]] = list(plan.alignments)
    for file_index, (path, frames) in enumerate(zip(paths, plan.frames_by_file)):
        projected_acc.begin_file()
        fixed_acc.begin_file()
        grid.begin_file()
        rows_seen = 0
        for offset, chunk in iter_shprop_chunks(path, chunk_rows):
            rows = chunk.shape[0]
            rows_seen += rows
            if rows_seen > ntime:
                raise CharacterError(
                    f"{path}: more rows than the {ntime} the planning pass counted; "
                    "the file changed under the analysis"
                )
            times = chunk[:, state_map.time_column]
            if file_index == 0:
                grid.reference[offset : offset + rows] = times
            try:
                grid.observe(path, offset, times)
            except StreamingError as exc:
                raise CharacterError(str(exc)) from exc

            pops = chunk[:, columns]
            tally.observe(pops)
            if not tally.in_unit_range():
                raise CharacterError(
                    "population columns outside [0,1]; verify state map "
                    f"({path}, rows {offset}..{offset + rows - 1})"
                )
            if state_map.complete_population and not tally.conserved():
                raise CharacterError(
                    "complete populations must sum to one in every file "
                    f"({path}, rows {offset}..{offset + rows - 1}: totals reach "
                    f"[{tally.total_min:.8g}, {tally.total_max:.8g}])"
                )

            # Frames for exactly these rows. The alignment is resolved once for
            # the whole history, so a chunk is a slice of it and the cyclic
            # mapping cannot drift at a chunk boundary.
            frame_slice = frames[offset : offset + rows]
            frame_indices = np.asarray(
                [frame_lookup[int(frame)] for frame in frame_slice], dtype=int
            )
            weights = selected[frame_indices]
            # Diagonal contraction: SHPROP supplies only rho_ii and PROCAR only
            # <i|P_g|i>, so no coherence term exists in the inputs to contract.
            projected = np.einsum("ts,tsg->tg", pops, weights)
            projected_acc.update(offset, projected)
            if per_file is not None:
                per_file[file_index, offset : offset + rows, :] = projected

            fixed_chunk = np.stack(
                [chunk[:, group].sum(axis=1) for group in fixed_columns], axis=1
            )
            fixed_acc.update(offset, fixed_chunk)
        if rows_seen != ntime:
            raise CharacterError(
                f"{path}: {rows_seen} rows streamed but planning counted {ntime}; "
                "the file changed under the analysis"
            )
        try:
            grid.finish_file(path)
            projected_acc.finish_file()
            fixed_acc.finish_file()
        except StreamingError as exc:
            raise CharacterError(str(exc)) from exc

    mean = projected_acc.mean_array()
    sem = projected_acc.sem()
    fixed_mean_array = fixed_acc.mean_array()
    fixed_mean: Dict[str, np.ndarray] = {
        name: fixed_mean_array[:, index] for index, name in enumerate(fixed_names)
    }
    time_ns = grid.reference / state_map.to_ns

    total = mean.sum(axis=1)
    conservation = {
        "min": float(np.min(total)),
        "max": float(np.max(total)),
        "mean": float(np.mean(total)),
        "max_abs_deviation_from_one": float(np.max(np.abs(total - 1.0))),
    }
    if state_map.complete_population and conservation["max_abs_deviation_from_one"] > 1.0e-5:
        raise CharacterError(
            "projection-weighted diagonal subsystem populations do not conserve the "
            f"complete SHPROP population: total ranges [{conservation['min']:.8g}, "
            f"{conservation['max']:.8g}] and departs from 1 by up to "
            f"{conservation['max_abs_deviation_from_one']:.3g} (tolerance 1e-5). "
            "Check that the state map covers the whole basis and that the atom "
            "groups capture the projection; this is not repaired automatically"
        )

    io_summary = {
        "mode": "streaming",
        "shprop_chunk_rows": int(chunk_rows),
        "chunks_per_history": len(chunk_plan(ntime, chunk_rows)),
        "per_file_retained": bool(per_file is not None),
        "accumulator_memmap_dir": str(memmap_dir) if memmap_dir else None,
        "accumulator_spill_files": [
            str(path)
            for accumulator in (projected_acc, fixed_acc)
            for path in accumulator.spill_files
        ],
        "note": (
            "each history is read once, in row chunks, and folded into a running "
            "ensemble mean and variance; no (nfiles, nrows, ncolumns) stack is "
            "built. The projected and fixed-column populations come from the same "
            "pass, so a history is never read twice"
        ),
    }

    return CharacterPopulationResult(
        time_ns=time_ns,
        group_names=group_names,
        per_file=per_file,
        mean=mean,
        sem=sem,
        fixed_mean=fixed_mean,
        projection=projection,
        file_alignment=alignments,
        conservation=conservation,
        io=io_summary,
        accumulators=[projected_acc, fixed_acc],
    )


def character_swap_rows(
    projection: ProjectionSeries, dominance_threshold: float = 0.6
) -> Tuple[List[List[Any]], Dict[str, Any]]:
    """Report dominant-subsystem changes along the underlying MD trajectory.

    Only the electronic frames the SHPROP histories actually visit are loaded,
    so consecutive entries in ``projection.frames`` need not be adjacent MD
    frames.  A change observed across a gap did happen, but it happened
    *somewhere inside* the gap and this analysis did not look there.  Such a
    change is reported with its frame gap and counted separately from an
    adjacent, fully resolved swap, rather than being presented as a single
    step between two distant frames.

    In a cyclic campaign the frames form a ring, and the ``period -> 1`` step
    is taken by the dynamics like any other.  It is examined when some history
    actually takes it, labelled ``cycle_wrap`` rather than ``adjacent``, and
    the summary states explicitly when it was excluded and why.

    The transition set and the swap definition are shared with
    :meth:`ProjectionSeries.dominance_summary`, so per-band counts sum to the
    totals here.
    """
    events = projection.swap_events(dominance_threshold)
    rows = [
        [
            event["band"],
            event["frame_before"],
            event["frame_after"],
            event["dominant_before"],
            event["dominant_after"],
            event["confidence_before"],
            event["confidence_after"],
            event["frame_gap"],
            event["resolution"],
        ]
        for event in events
    ]
    confidence = np.max(projection.weights, axis=2)
    mixed = int(np.count_nonzero(confidence < dominance_threshold))
    total = projection.weights.shape[0] * projection.weights.shape[1]
    counts = {"adjacent": 0, "across_gap": 0, "cycle_wrap": 0}
    for event in events:
        counts[event["resolution"]] += 1

    frames_list = [int(frame) for frame in projection.frames]
    unobserved = sum(
        max(0, b - a - 1) for a, b in zip(frames_list, frames_list[1:])
    )
    wrap = projection.cycle_wrap_status()
    if unobserved:
        resolution_note = (
            "only the electronic frames the SHPROP histories visit are loaded, so "
            f"{unobserved} MD frame(s) lie between examined frames and were not "
            "inspected. A change marked 'across_gap' happened somewhere inside "
            "that gap, not in one step between the two frames named. Swap counts "
            "are a lower bound on the number of character changes along the full "
            "MD trajectory"
        )
    else:
        resolution_note = (
            "every examined frame is adjacent to the next, so the swap count "
            "resolves every change over the range examined"
        )
    summary = {
        "dominance_threshold": dominance_threshold,
        "dominant_character_swaps": len(events),
        "adjacent_swaps": counts["adjacent"],
        "changes_across_a_frame_gap": counts["across_gap"],
        "cycle_wrap_swaps": counts["cycle_wrap"],
        "mixed_frame_band_samples": mixed,
        "total_frame_band_samples": total,
        "mixed_fraction": float(mixed / total) if total else 0.0,
        "frames_examined": len(frames_list),
        "transitions_examined_per_band": len(projection.examined_transitions()),
        "md_frames_skipped_between_examined_frames": unobserved,
        "cycle_wrap": wrap,
        "cycle_wrap_note": wrap["note"],
        "resolution_note": resolution_note,
    }
    return rows, summary


DISCREPANCY_HEADER = [
    "group",
    "max_abs_difference",
    "time_of_max_ns",
    "rms_difference",
    "integrated_abs_difference_ns",
    "mean_signed_difference",
    "fixed_at_max",
    "projected_at_max",
]


def fixed_vs_projected_summary(result: "CharacterPopulationResult") -> Dict[str, Any]:
    """How far the dynamic character moves the answer, per shared group.

    Compares the projection-weighted population against the fixed-column
    population on the same time grid, for groups named in both.  A large
    discrepancy says the fixed map was mislabelling occupation; a small one
    says the fixed map was adequate for this campaign.  Neither is a defect.
    """
    time_ns = result.time_ns
    lookup = {name: i for i, name in enumerate(result.group_names)}
    shared = [name for name in result.fixed_mean if name in lookup]
    rows: List[List[Any]] = []
    records: List[Dict[str, Any]] = []
    integrate = getattr(np, "trapezoid", None) or np.trapz
    for name in shared:
        fixed = np.asarray(result.fixed_mean[name], dtype=float)
        projected = result.mean[:, lookup[name]]
        difference = projected - fixed
        peak = int(np.argmax(np.abs(difference)))
        record = {
            "group": name,
            "max_abs_difference": float(np.abs(difference)[peak]),
            "time_of_max_ns": float(time_ns[peak]),
            "rms_difference": float(np.sqrt(np.mean(difference**2))),
            "integrated_abs_difference_ns": float(integrate(np.abs(difference), x=time_ns)),
            "mean_signed_difference": float(np.mean(difference)),
            "fixed_at_max": float(fixed[peak]),
            "projected_at_max": float(projected[peak]),
        }
        records.append(record)
        rows.append([record[key] for key in DISCREPANCY_HEADER])

    largest = max(records, key=lambda r: r["max_abs_difference"]) if records else None
    return {
        "shared_groups": shared,
        "groups_only_in_fixed_map": sorted(set(result.fixed_mean) - set(lookup)),
        "groups_only_in_projection": sorted(set(lookup) - set(result.fixed_mean)),
        "per_group": records,
        "rows": rows,
        "largest_disagreement": largest,
        "note": (
            "the fixed comparison assigns each SHPROP column to one group for the "
            "whole trajectory; the projected one reassigns character frame by frame. "
            "A large difference means the fixed labelling was wrong somewhere, not "
            "that either number is a flux or a transfer. Both series are diagonal "
            "in the adiabatic basis, so this difference is about labelling alone "
            "and says nothing about the coherences neither one contains"
        ),
    }


def projection_table_rows(projection: ProjectionSeries) -> Iterable[List[Any]]:
    """Long-form normalized and raw character table for auditing."""
    for fi, frame in enumerate(projection.frames):
        for bi, band in enumerate(projection.bands):
            dominant = int(np.argmax(projection.weights[fi, bi]))
            for gi, group in enumerate(projection.group_names):
                yield [
                    int(frame),
                    int(band),
                    group,
                    float(projection.weights[fi, bi, gi]),
                    float(projection.captured_projection[fi, bi]),
                    float(projection.total_projection[fi, bi]),
                    projection.group_names[dominant],
                    float(np.max(projection.weights[fi, bi])),
                ]
