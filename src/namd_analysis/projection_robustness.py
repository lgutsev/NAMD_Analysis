"""Is apparent fragment mixing robust to the projection's missing weight?

``character-populations`` normalizes each band's PAW-projector weight over the
declared fragments, ``w_ig = W_ig / sum_g W_ig``, and reports the denominator
as ``captured_projection``.  That is the authoritative result of the method
and nothing here changes it.  This module asks a narrower question of the same
numbers: when a band looks mixed between two fragments (BCF and PCBM in the
FAPI study), does the mixing survive once the *absolute* weights are looked at,
or does it exist only in the normalized ratio of a small captured fraction?

Three facts organize everything below.

1. Normalization rescales every declared fragment of a sample by the same
   factor ``1 / captured``.  It therefore cannot change which declared
   fragment is largest, nor the ratio between two fragments.  "Dominant before
   normalization" can only differ from "dominant after" when the weight the
   normalization divides away -- ``1 - captured_projection`` -- is counted as
   a competitor, so that is how it is defined here.  The declared-only argmax
   is reported as well, as a checked invariant rather than a claim.

2. What normalization *does* decide is where the uncaptured weight goes: it
   allocates it to the fragments in proportion to what they already hold.
   Any real-space partition of the full band is some other allocation.  If
   the captured weight ``W_g`` of each fragment lies inside that fragment's
   region, the full-space fraction satisfies ``W_g <= f_g <= W_g + u`` with
   ``u = 1 - captured`` shared among all fragments.  From that, each sample's
   mixing metric ``min(f_a, f_b)`` has a worst case (``min(W_a, W_b)``, none
   of the missing weight on the pair), the normalized value, and a best case
   (as much of it on the pair as the budget allows).  The three are ordered,
   worst <= normalized <= best, and are all reported.

3. Low capture and mixing are different things.  A band can have low capture
   and one dominant fragment, or high capture and genuine two-fragment weight.
   The audit keeps them on separate axes and never infers one from the other.

The idealisation behind the bracket is stated in :data:`BRACKET_ASSUMPTIONS`
and carried into every report.  Nothing here discards, repairs or reweights a
sample.  Thresholds appear only as sensitivity grids; none is preferred, and
the capture grid always includes zero, i.e. every sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class RobustnessError(ValueError):
    """Raised when the audit would have to guess."""


#: Normalized-fraction thresholds tau: a sample is "mixed" at tau when both
#: fragments of the pair hold at least tau.  0.10 is the criterion Issue #4
#: counted with; it is one grid point among several, not a preferred value.
DEFAULT_NORMALIZED_THRESHOLDS: Tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)

#: Minimum captured projection used to *condition* a count, never to filter
#: the data.  Zero is always included, so the unconditioned count is always
#: the first row a reader sees.
DEFAULT_CAPTURE_MINIMA: Tuple[float, ...] = (0.0,) + tuple(
    round(0.30 + 0.02 * k, 2) for k in range(16)
)

#: Minimum raw (unnormalized) weight on *each* fragment of the pair.
DEFAULT_RAW_MINIMA: Tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10)

#: Name used for the uncaptured weight (``1 - captured``) when it competes
#: for dominance with the declared fragments.
UNCAPTURED = "uncaptured"

BRACKET_ASSUMPTIONS = (
    "The allocation bracket W_g <= f_g <= W_g + (1 - captured) assumes (i) each "
    "band is normalized to one, which holds for a single-k-point, non-spin-"
    "polarized projection; (ii) the PAW-projector weight a fragment's ions "
    "carry would also be assigned to that fragment by a full-space partition; "
    "and (iii) nothing is known about where the uncaptured weight lies. "
    "PAW projector weights are not strict integrals of the density over "
    "disjoint spheres, and overlapping spheres can make them sum above one, "
    "so the bracket is an idealisation that bounds what the projection alone "
    "can exclude. It is not a measurement. A full-space partition of the band "
    "density (see character-fullspace-prepare) measures the allocation; this "
    "bracket only says how much room there is for it to differ from the "
    "normalized one"
)

CORRELATION_NOTE = (
    "Pearson and Spearman coefficients over the frames of one band. No p-value "
    "is reported: consecutive MD frames are autocorrelated, so the effective "
    "number of independent samples is smaller than n by an unknown factor and "
    "a p-value computed from n would overstate the evidence. A coefficient is "
    "null when either series is constant, which leaves it undefined"
)

INTERPRETATION_NOTE = (
    "A correlation between low capture and apparent mixing is predicted by "
    "BOTH readings of Issue #4: genuine delocalization puts weight into the "
    "interstitial and onto several fragments at once, and a normalization "
    "artefact inflates small weights most where the denominator is smallest. "
    "The correlation therefore cannot separate them. What bears on the "
    "question is whether the mixing survives in the raw weights (the "
    "worst-case column and the raw-minimum grid), how far the allocation "
    "bracket leaves the answer open, and ultimately a full-space partition "
    "of the band density, which measures the allocation the bracket only bounds"
)


# --------------------------------------------------------------------------
# The table the audit runs on
# --------------------------------------------------------------------------


@dataclass
class ProjectionTable:
    """Normalized weights plus the denominators they were divided by.

    ``weights[f, b, g]`` is ``w_ig`` and ``captured[f, b]`` is ``sum_g W_ig``,
    exactly as ``character-populations`` computed them.  ``total[f, b]`` is
    the sum over every PROCAR ion; it equals ``captured`` when the atom
    partition is complete, and is ``None`` when the source did not carry it.
    """

    frames: np.ndarray
    bands: np.ndarray
    groups: List[str]
    weights: np.ndarray
    captured: np.ndarray
    total: Optional[np.ndarray] = None
    source: str = ""

    def __post_init__(self) -> None:
        self.frames = np.asarray(self.frames, dtype=int)
        self.bands = np.asarray(self.bands, dtype=int)
        self.weights = np.asarray(self.weights, dtype=float)
        self.captured = np.asarray(self.captured, dtype=float)
        nf, nb = len(self.frames), len(self.bands)
        if self.weights.shape != (nf, nb, len(self.groups)):
            raise RobustnessError(
                f"weights have shape {self.weights.shape}; expected "
                f"({nf}, {nb}, {len(self.groups)}) for frames x bands x groups"
            )
        if self.captured.shape != (nf, nb):
            raise RobustnessError(
                f"captured_projection has shape {self.captured.shape}; expected ({nf}, {nb})"
            )
        if self.total is not None:
            self.total = np.asarray(self.total, dtype=float)
            if self.total.shape != (nf, nb):
                raise RobustnessError(
                    f"total_projection has shape {self.total.shape}; expected ({nf}, {nb})"
                )
        if not np.all(np.isfinite(self.captured)) or np.any(self.captured <= 0.0):
            raise RobustnessError(
                "captured_projection must be finite and positive for every sample; "
                "the character run refuses a zero denominator, so a non-positive "
                "value here means the table was not written by it"
            )
        if len(set(self.frames.tolist())) != nf:
            raise RobustnessError("frame numbers in the projection table are not unique")

    @classmethod
    def from_character_series(cls, series, source: str = "") -> "ProjectionTable":
        """From :class:`namd_analysis.crossings.CharacterSeries` (the CSV reader)."""
        if getattr(series, "captured", None) is None:
            raise RobustnessError(
                "the projection character table carries no captured_projection "
                "column, so the raw weights W_ig cannot be recovered. Point this at "
                "the projection_character.csv a character-populations run wrote"
            )
        return cls(
            frames=series.frames,
            bands=series.bands,
            groups=list(series.groups),
            weights=series.weights,
            captured=series.captured,
            total=getattr(series, "total", None),
            source=source,
        )

    @classmethod
    def from_projection_series(cls, series, source: str = "") -> "ProjectionTable":
        """From :class:`namd_analysis.character.ProjectionSeries` (a live run)."""
        return cls(
            frames=series.frames,
            bands=series.bands,
            groups=list(series.group_names),
            weights=series.weights,
            captured=series.captured_projection,
            total=series.total_projection,
            source=source,
        )

    # -- derived quantities; every one is recomputed, never stored back ------

    def raw_weights(self) -> np.ndarray:
        """``W_ig = w_ig * captured``: each fragment's share of the whole band."""
        return self.weights * self.captured[..., None]

    def total_or_captured(self) -> np.ndarray:
        return self.total if self.total is not None else self.captured

    def undeclared_ion_weight(self) -> Optional[np.ndarray]:
        """Weight on PROCAR ions no group declares (zero for a complete map)."""
        if self.total is None:
            return None
        return self.total - self.captured

    def unprojected_remainder(self) -> np.ndarray:
        """``1 - total``: band weight outside every ion's projection."""
        return 1.0 - self.total_or_captured()

    def uncaptured(self) -> np.ndarray:
        """``1 - captured``: everything normalization divides away."""
        return 1.0 - self.captured

    def band_position(self, band: int) -> int:
        lookup = {int(b): i for i, b in enumerate(self.bands)}
        if int(band) not in lookup:
            raise RobustnessError(
                f"band {band} is not in the projection table (it holds bands "
                f"{[int(b) for b in self.bands]})"
            )
        return lookup[int(band)]

    def group_position(self, group: str) -> int:
        if group not in self.groups:
            raise RobustnessError(
                f"group {group!r} is not in the projection table (it holds {self.groups})"
            )
        return self.groups.index(group)


def resolve_pair(table: ProjectionTable, pair: Sequence[str]) -> Tuple[str, str]:
    names = [str(name).strip() for name in pair]
    if len(names) != 2 or names[0] == names[1]:
        raise RobustnessError(f"the mixing pair must name two distinct groups, got {list(pair)}")
    for name in names:
        table.group_position(name)
    return names[0], names[1]


# --------------------------------------------------------------------------
# Per-sample quantities
# --------------------------------------------------------------------------


def mixing_levels(
    raw_a: np.ndarray, raw_b: np.ndarray, captured: np.ndarray
) -> Dict[str, np.ndarray]:
    """Worst-case, normalized and best-case ``min(f_a, f_b)`` for each sample.

    ``u = 1 - captured`` is the uncaptured budget.  The worst case gives the
    pair none of it; normalization gives each fragment ``W_g (1 - c) / c``;
    the best case maximizes ``min(f_a, f_b)`` subject to ``f_g >= W_g`` and
    ``(f_a - W_a) + (f_b - W_b) <= u``, which is ``min(W_a, W_b) + u`` when the
    budget cannot close the gap between the two, and ``(W_a + W_b + u) / 2``
    when it can.  A negative budget (captured above one) leaves the bracket
    undefined, so best case is NaN there rather than clipped.
    """
    raw_a = np.asarray(raw_a, dtype=float)
    raw_b = np.asarray(raw_b, dtype=float)
    captured = np.asarray(captured, dtype=float)
    budget = 1.0 - captured
    worst = np.minimum(raw_a, raw_b)
    normalized = worst / captured
    best = np.minimum((raw_a + raw_b + budget) / 2.0, worst + budget)
    best = np.where(budget >= 0.0, best, np.nan)
    larger = np.maximum(raw_a, raw_b)
    with np.errstate(invalid="ignore", divide="ignore"):
        balance = np.where(larger > 0.0, worst / larger, np.nan)
    return {
        "worst_case": worst,
        "normalized": normalized,
        "best_case": best,
        "balance": balance,
    }


def allocation_class(worst: float, best: float, tau: float) -> str:
    """Where a sample stands at threshold ``tau`` across every allocation."""
    if not np.isfinite(best):
        return "bracket_undefined"
    if worst >= tau:
        return "mixed_under_every_allocation"
    if best >= tau:
        return "mixed_under_some_allocation"
    return "not_mixed_under_any_allocation"


SAMPLE_FIXED_HEADER = [
    "frame",
    "band",
    "captured_projection",
    "total_projection",
    "undeclared_ion_weight",
    "unprojected_remainder",
    "uncaptured_weight",
    "normalization_amplification",
]

SAMPLE_TAIL_HEADER = [
    "dominant_before_normalization",
    "dominant_after_normalization",
    "dominant_raw_declared_only",
    "dominance_changed_by_normalization",
    "pair",
    "pair_min_worst_case",
    "pair_min_normalized",
    "pair_min_best_case",
    "pair_balance",
    "pair_share_normalized",
]


def sample_header(groups: Sequence[str], thresholds: Sequence[float] = ()) -> List[str]:
    """Columns of the per-sample table.

    With ``thresholds``, one ``allocation_class_at_<tau>`` column per
    threshold says, for that sample, whether the pair is mixed under every
    allocation of the uncaptured weight, under some, or under none -- the
    per-sample answer to "does the mixing survive the absolute weights?".
    """
    return (
        list(SAMPLE_FIXED_HEADER)
        + [f"raw_{g}" for g in groups]
        + [f"normalized_{g}" for g in groups]
        + list(SAMPLE_TAIL_HEADER)
        + [f"allocation_class_at_{_tau_key(t)}" for t in thresholds]
    )


@dataclass
class SampleArrays:
    """Every per-(frame, band) quantity, as arrays; rows are built from these."""

    raw: np.ndarray  # (nf, nb, ng)
    remainder: np.ndarray  # (nf, nb)  1 - total
    uncaptured: np.ndarray  # (nf, nb)  1 - captured
    undeclared: Optional[np.ndarray]  # (nf, nb)
    dominant_after: np.ndarray  # (nf, nb) index into groups
    dominant_raw_declared: np.ndarray  # (nf, nb) index into groups
    dominant_before: np.ndarray  # (nf, nb) index into groups, or -1 for UNCAPTURED
    levels: Dict[str, np.ndarray]  # each (nf, nb)
    pair_share: np.ndarray  # (nf, nb)


def sample_arrays(table: ProjectionTable, pair: Tuple[str, str]) -> SampleArrays:
    raw = table.raw_weights()
    ia, ib = table.group_position(pair[0]), table.group_position(pair[1])
    remainder = table.unprojected_remainder()
    dominant_after = np.argmax(table.weights, axis=2)
    dominant_raw = np.argmax(raw, axis=2)
    # The uncaptured weight competes only when it is positive; a projection
    # summing above one has no unseen weight to compete with.
    best_raw = np.max(raw, axis=2)
    dominant_before = np.where(table.uncaptured() > best_raw, -1, dominant_raw)
    levels = mixing_levels(raw[..., ia], raw[..., ib], table.captured)
    return SampleArrays(
        raw=raw,
        remainder=remainder,
        uncaptured=table.uncaptured(),
        undeclared=table.undeclared_ion_weight(),
        dominant_after=dominant_after,
        dominant_raw_declared=dominant_raw,
        dominant_before=dominant_before,
        levels=levels,
        pair_share=table.weights[..., ia] + table.weights[..., ib],
    )


def _dominant_name(table: ProjectionTable, index: int) -> str:
    return UNCAPTURED if index < 0 else table.groups[index]


def sample_rows(
    table: ProjectionTable,
    pair: Tuple[str, str],
    arrays: Optional[SampleArrays] = None,
    thresholds: Sequence[float] = (),
) -> Iterable[List[Any]]:
    """One row per (frame, band): every sample, nothing skipped."""
    arrays = arrays if arrays is not None else sample_arrays(table, pair)
    pair_label = f"{pair[0]}/{pair[1]}"
    for fi, frame in enumerate(table.frames):
        for bi, band in enumerate(table.bands):
            captured = float(table.captured[fi, bi])
            total = None if table.total is None else float(table.total[fi, bi])
            undeclared = None if arrays.undeclared is None else float(arrays.undeclared[fi, bi])
            worst = float(arrays.levels["worst_case"][fi, bi])
            best = float(arrays.levels["best_case"][fi, bi])
            before = _dominant_name(table, int(arrays.dominant_before[fi, bi]))
            after = table.groups[int(arrays.dominant_after[fi, bi])]
            declared = table.groups[int(arrays.dominant_raw_declared[fi, bi])]
            yield (
                [
                    int(frame),
                    int(band),
                    captured,
                    total,
                    undeclared,
                    float(arrays.remainder[fi, bi]),
                    float(arrays.uncaptured[fi, bi]),
                    1.0 / captured,
                ]
                + [float(v) for v in arrays.raw[fi, bi]]
                + [float(v) for v in table.weights[fi, bi]]
                + [
                    before,
                    after,
                    declared,
                    before != after,
                    pair_label,
                    worst,
                    float(arrays.levels["normalized"][fi, bi]),
                    best,
                    float(arrays.levels["balance"][fi, bi]),
                    float(arrays.pair_share[fi, bi]),
                ]
                + [allocation_class(worst, best, tau) for tau in thresholds]
            )


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def _distribution(values: np.ndarray, frames: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(values, dtype=float)
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "p1": float(np.percentile(values, 1)),
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "frame_of_minimum": int(frames[int(np.argmin(values))]),
        "frame_of_maximum": int(frames[int(np.argmax(values))]),
    }


def _ranks(values: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(values)


def correlation(x: Sequence[float], y: Sequence[float]) -> Dict[str, Any]:
    """Pearson and Spearman coefficients, or null with the reason."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    record: Dict[str, Any] = {"n": int(x.size), "pearson": None, "spearman": None}
    if x.size < 3:
        record["undefined_because"] = "fewer than three finite pairs"
        return record
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        record["undefined_because"] = "one of the two series is constant"
        return record
    record["pearson"] = float(np.corrcoef(x, y)[0, 1])
    rx, ry = _ranks(x), _ranks(y)
    if np.ptp(rx) == 0.0 or np.ptp(ry) == 0.0:
        record["undefined_because"] = "ranks are constant"
        return record
    record["spearman"] = float(np.corrcoef(rx, ry)[0, 1])
    return record


def capture_strata(captured: np.ndarray, n_strata: int) -> Tuple[np.ndarray, np.ndarray]:
    """Assign each sample to a capture quantile stratum (0 = lowest capture).

    The edges are the band's own quantiles, so the strata describe the
    distribution; they are not cutoffs and no sample is left out of them.
    """
    if n_strata < 1:
        raise RobustnessError("the number of capture strata must be at least one")
    edges = np.quantile(captured, np.linspace(0.0, 1.0, n_strata + 1))
    # Interior edges only; searchsorted with side='right' puts a value equal
    # to an edge into the upper stratum, and the maximum into the last one.
    labels = np.searchsorted(edges[1:-1], captured, side="right")
    return labels.astype(int), edges


def _mixed(levels: Dict[str, np.ndarray], tau: float, which: str) -> np.ndarray:
    values = levels[which]
    return np.isfinite(values) & (values >= tau)


def band_summary(
    table: ProjectionTable,
    pair: Tuple[str, str],
    arrays: SampleArrays,
    bi: int,
    quality_threshold: Optional[float],
    normalized_thresholds: Sequence[float],
    n_strata: int = 4,
) -> Dict[str, Any]:
    """Everything the audit says about one band, over all of its frames."""
    frames = table.frames
    captured = table.captured[:, bi]
    levels = {key: value[:, bi] for key, value in arrays.levels.items()}
    raw = arrays.raw[:, bi, :]
    weights = table.weights[:, bi, :]

    capture = _distribution(captured, frames)
    if quality_threshold is not None:
        below = captured < quality_threshold
        capture["existing_reporting_threshold"] = float(quality_threshold)
        capture["samples_below_existing_threshold"] = int(np.count_nonzero(below))
        capture["fraction_below_existing_threshold"] = float(np.mean(below))

    names = list(table.groups)
    after_counts = {name: int(np.count_nonzero(arrays.dominant_after[:, bi] == gi))
                    for gi, name in enumerate(names)}
    before_counts = {name: int(np.count_nonzero(arrays.dominant_before[:, bi] == gi))
                     for gi, name in enumerate(names)}
    before_counts[UNCAPTURED] = int(np.count_nonzero(arrays.dominant_before[:, bi] < 0))
    invariant_violations = int(
        np.count_nonzero(arrays.dominant_raw_declared[:, bi] != arrays.dominant_after[:, bi])
    )

    correlations = {
        "capture_vs_normalized": {
            name: correlation(captured, weights[:, gi]) for gi, name in enumerate(names)
        },
        "capture_vs_raw": {
            name: correlation(captured, raw[:, gi]) for gi, name in enumerate(names)
        },
        "capture_vs_pair_min_normalized": correlation(captured, levels["normalized"]),
        "capture_vs_pair_min_worst_case": correlation(captured, levels["worst_case"]),
        "note": CORRELATION_NOTE,
    }

    strata, edges = capture_strata(captured, n_strata)
    stratum_records = []
    for s in range(n_strata):
        member = strata == s
        record: Dict[str, Any] = {
            "stratum": s + 1,
            "capture_from": float(edges[s]),
            "capture_to": float(edges[s + 1]),
            "n_samples": int(np.count_nonzero(member)),
            "mixed_normalized": {},
            "mixed_worst_case": {},
        }
        for tau in normalized_thresholds:
            key = _tau_key(tau)
            record["mixed_normalized"][key] = int(
                np.count_nonzero(member & _mixed(levels, tau, "normalized"))
            )
            record["mixed_worst_case"][key] = int(
                np.count_nonzero(member & _mixed(levels, tau, "worst_case"))
            )
        stratum_records.append(record)

    by_threshold = []
    for tau in normalized_thresholds:
        mixed = _mixed(levels, tau, "normalized")
        classes = _class_counts(levels, tau)
        entry: Dict[str, Any] = {
            "threshold": float(tau),
            "n_mixed_normalized": int(np.count_nonzero(mixed)),
            "n_mixed_under_every_allocation": classes["mixed_under_every_allocation"],
            "n_mixed_under_some_allocation": classes["mixed_under_some_allocation"],
            "n_not_mixed_under_any_allocation": classes["not_mixed_under_any_allocation"],
            "n_bracket_undefined": classes["bracket_undefined"],
            "n_normalized_mixed_that_survive_worst_case": int(
                np.count_nonzero(mixed & _mixed(levels, tau, "worst_case"))
            ),
            "median_capture_mixed": (
                float(np.median(captured[mixed])) if np.any(mixed) else None
            ),
            "median_capture_not_mixed": (
                float(np.median(captured[~mixed])) if np.any(~mixed) else None
            ),
            "n_mixed_in_lowest_capture_stratum": int(np.count_nonzero(mixed & (strata == 0))),
            "frames_mixed_normalized": [int(f) for f in frames[mixed]],
        }
        by_threshold.append(entry)

    normalized = levels["normalized"]
    top = int(np.argmax(normalized))
    return {
        "band": int(table.bands[bi]),
        "n_samples": int(len(frames)),
        "capture": capture,
        "total_projection": (
            _distribution(table.total[:, bi], frames) if table.total is not None else None
        ),
        "unprojected_remainder": _distribution(arrays.remainder[:, bi], frames),
        "undeclared_ion_weight_maximum": (
            None if arrays.undeclared is None else float(np.max(arrays.undeclared[:, bi]))
        ),
        "mean_normalized": {name: float(np.mean(weights[:, gi])) for gi, name in enumerate(names)},
        "mean_raw": {name: float(np.mean(raw[:, gi])) for gi, name in enumerate(names)},
        "dominant_after_normalization": after_counts,
        "dominant_before_normalization": before_counts,
        "samples_where_uncaptured_weight_exceeds_every_fragment": before_counts[UNCAPTURED],
        "declared_argmax_invariance_violations": invariant_violations,
        "pair": list(pair),
        "pair_min": {
            key: {
                "median": _nan_stat(np.nanmedian, levels[key]),
                "maximum": _nan_stat(np.nanmax, levels[key]),
            }
            for key in ("worst_case", "normalized", "best_case")
        },
        "frame_of_largest_normalized_pair_min": int(frames[top]),
        "correlations": correlations,
        "capture_strata": stratum_records,
        "mixing_by_threshold": by_threshold,
    }


def _nan_stat(function, values: np.ndarray) -> Optional[float]:
    values = np.asarray(values, dtype=float)
    if not np.any(np.isfinite(values)):
        return None
    return float(function(values))


def _tau_key(tau: float) -> str:
    return f"{float(tau):g}"


def _class_counts(levels: Dict[str, np.ndarray], tau: float) -> Dict[str, int]:
    worst, best = levels["worst_case"], levels["best_case"]
    undefined = ~np.isfinite(best)
    every = ~undefined & (worst >= tau)
    some = ~undefined & ~every & (best >= tau)
    none = ~undefined & ~every & ~some
    return {
        "mixed_under_every_allocation": int(np.count_nonzero(every)),
        "mixed_under_some_allocation": int(np.count_nonzero(some)),
        "not_mixed_under_any_allocation": int(np.count_nonzero(none)),
        "bracket_undefined": int(np.count_nonzero(undefined)),
    }


# --------------------------------------------------------------------------
# The sensitivity grid
# --------------------------------------------------------------------------

SENSITIVITY_HEADER = [
    "band",
    "normalized_threshold",
    "min_captured_projection",
    "min_raw_each_pair_fragment",
    "n_samples",
    "n_meeting_capture_condition",
    "n_excluded_by_capture_condition",
    "n_mixed",
    "fraction_mixed_of_all_samples",
    "fraction_mixed_of_those_meeting_capture",
]


def sensitivity_rows(
    table: ProjectionTable,
    pair: Tuple[str, str],
    arrays: SampleArrays,
    normalized_thresholds: Sequence[float],
    capture_minima: Sequence[float],
    raw_minima: Sequence[float],
) -> List[List[Any]]:
    """Mixed-sample counts over the full three-way grid, per band and overall.

    A sample counts as mixed at (tau, c_min, r_min) when both fragments of the
    pair hold at least ``tau`` of the normalized weight, its captured
    projection is at least ``c_min``, and each fragment's *raw* weight is at
    least ``r_min``.  Every row states how many samples the capture condition
    set aside, so conditioning can never pass for filtering.
    """
    ia, ib = table.group_position(pair[0]), table.group_position(pair[1])
    raw_a, raw_b = arrays.raw[..., ia], arrays.raw[..., ib]
    w_a, w_b = table.weights[..., ia], table.weights[..., ib]
    captured = table.captured
    rows: List[List[Any]] = []
    scopes: List[Tuple[Any, Optional[int]]] = [(int(b), i) for i, b in enumerate(table.bands)]
    scopes.append(("all", None))
    for label, bi in scopes:
        sel = (slice(None), bi) if bi is not None else (slice(None), slice(None))
        n_samples = int(captured[sel].size)
        for tau in normalized_thresholds:
            normalized_ok = (w_a[sel] >= tau) & (w_b[sel] >= tau)
            for c_min in capture_minima:
                capture_ok = captured[sel] >= c_min
                n_capture = int(np.count_nonzero(capture_ok))
                for r_min in raw_minima:
                    raw_ok = (raw_a[sel] >= r_min) & (raw_b[sel] >= r_min)
                    n_mixed = int(np.count_nonzero(normalized_ok & capture_ok & raw_ok))
                    rows.append(
                        [
                            label,
                            float(tau),
                            float(c_min),
                            float(r_min),
                            n_samples,
                            n_capture,
                            n_samples - n_capture,
                            n_mixed,
                            n_mixed / n_samples if n_samples else None,
                            n_mixed / n_capture if n_capture else None,
                        ]
                    )
    return rows


def parse_grid(spec: Optional[str], default: Sequence[float], name: str) -> List[float]:
    """``a,b,c`` or ``START:STOP:STEP`` (inclusive of STOP within rounding)."""
    if spec is None or not str(spec).strip():
        values = [float(v) for v in default]
    else:
        text = str(spec).strip()
        try:
            if ":" in text:
                parts = [float(p) for p in text.split(":")]
                if len(parts) != 3 or parts[2] <= 0.0 or parts[1] < parts[0]:
                    raise ValueError("expected START:STOP:STEP with STEP > 0")
                count = int(np.floor((parts[1] - parts[0]) / parts[2] + 1e-9)) + 1
                values = [round(parts[0] + k * parts[2], 10) for k in range(count)]
            else:
                values = [float(p) for p in text.split(",") if p.strip()]
        except ValueError as exc:
            raise RobustnessError(f"{name}: cannot read grid {spec!r} ({exc})") from None
    if not values:
        raise RobustnessError(f"{name}: the grid is empty")
    if any(not np.isfinite(v) or v < 0.0 for v in values):
        raise RobustnessError(f"{name}: grid values must be finite and non-negative")
    return sorted(set(values))


# --------------------------------------------------------------------------
# The whole audit
# --------------------------------------------------------------------------


@dataclass
class RobustnessAudit:
    table: ProjectionTable
    pair: Tuple[str, str]
    arrays: SampleArrays
    per_band: List[Dict[str, Any]]
    campaign: Dict[str, Any]
    grids: Dict[str, List[float]]
    quality_threshold: Optional[float]

    def band(self, band: int) -> Dict[str, Any]:
        for record in self.per_band:
            if record["band"] == int(band):
                return record
        raise RobustnessError(f"band {band} is not in the audit")


def audit(
    table: ProjectionTable,
    pair: Sequence[str] = ("BCF", "PCBM"),
    quality_threshold: Optional[float] = None,
    normalized_thresholds: Sequence[float] = DEFAULT_NORMALIZED_THRESHOLDS,
    capture_minima: Sequence[float] = DEFAULT_CAPTURE_MINIMA,
    raw_minima: Sequence[float] = DEFAULT_RAW_MINIMA,
    n_strata: int = 4,
) -> RobustnessAudit:
    """Run the audit over every band and frame of ``table``."""
    resolved = resolve_pair(table, pair)
    capture_grid = sorted(set(float(c) for c in capture_minima) | {0.0})
    if quality_threshold is not None:
        # The existing reporting threshold is always a grid point, so the
        # conditioned count at the number the character run already reports
        # is on the table without anyone having to ask for it.
        capture_grid = sorted(set(capture_grid) | {float(quality_threshold)})
    raw_grid = sorted(set(float(r) for r in raw_minima) | {0.0})
    tau_grid = sorted(set(float(t) for t in normalized_thresholds))
    if any(t <= 0.0 for t in tau_grid):
        raise RobustnessError(
            "normalized thresholds must be positive: at zero every sample is 'mixed'"
        )
    arrays = sample_arrays(table, resolved)
    per_band = [
        band_summary(table, resolved, arrays, bi, quality_threshold, tau_grid, n_strata)
        for bi in range(len(table.bands))
    ]

    campaign: Dict[str, Any] = {
        "n_frames": int(len(table.frames)),
        "n_bands": int(len(table.bands)),
        "n_samples": int(table.captured.size),
        "capture": _distribution(table.captured.reshape(-1), np.repeat(table.frames, len(table.bands))),
        "mixed_normalized_by_threshold": [],
    }
    for tau in tau_grid:
        mixed = _mixed(arrays.levels, tau, "normalized")
        by_band = {
            int(b): int(np.count_nonzero(mixed[:, i])) for i, b in enumerate(table.bands)
        }
        campaign["mixed_normalized_by_threshold"].append(
            {
                "threshold": tau,
                "n_mixed": int(np.count_nonzero(mixed)),
                "fraction_of_samples": float(np.mean(mixed)),
                "by_band": by_band,
            }
        )
    if quality_threshold is not None:
        below = table.captured < quality_threshold
        campaign["samples_below_existing_threshold"] = int(np.count_nonzero(below))
        campaign["below_existing_threshold_by_band"] = {
            int(b): int(np.count_nonzero(below[:, i])) for i, b in enumerate(table.bands)
        }
    invariant = sum(record["declared_argmax_invariance_violations"] for record in per_band)
    campaign["declared_argmax_invariance_violations"] = invariant
    return RobustnessAudit(
        table=table,
        pair=resolved,
        arrays=arrays,
        per_band=per_band,
        campaign=campaign,
        grids={
            "normalized_thresholds": tau_grid,
            "capture_minima": capture_grid,
            "raw_minima": raw_grid,
        },
        quality_threshold=quality_threshold,
    )


BY_BAND_HEADER = [
    "band",
    "n_samples",
    "capture_median",
    "capture_minimum",
    "capture_p1",
    "capture_p5",
    "capture_p25",
    "capture_p75",
    "capture_p95",
    "capture_maximum",
    "frame_of_minimum_capture",
    "samples_below_existing_threshold",
    "unprojected_remainder_median",
    "undeclared_ion_weight_maximum",
    "samples_where_uncaptured_weight_exceeds_every_fragment",
    "declared_argmax_invariance_violations",
    "pair_min_worst_case_median",
    "pair_min_normalized_median",
    "pair_min_best_case_median",
    "pair_min_normalized_maximum",
    "frame_of_largest_normalized_pair_min",
    "spearman_capture_vs_pair_min_normalized",
    "spearman_capture_vs_pair_min_worst_case",
]


def by_band_rows(result: RobustnessAudit) -> List[List[Any]]:
    """One row per band; normalized-fraction correlations are appended per group."""
    rows = []
    for record in result.per_band:
        capture = record["capture"]
        corr = record["correlations"]
        rows.append(
            [
                record["band"],
                record["n_samples"],
                capture["median"],
                capture["minimum"],
                capture["p1"],
                capture["p5"],
                capture["p25"],
                capture["p75"],
                capture["p95"],
                capture["maximum"],
                capture["frame_of_minimum"],
                capture.get("samples_below_existing_threshold"),
                record["unprojected_remainder"]["median"],
                record["undeclared_ion_weight_maximum"],
                record["samples_where_uncaptured_weight_exceeds_every_fragment"],
                record["declared_argmax_invariance_violations"],
                record["pair_min"]["worst_case"]["median"],
                record["pair_min"]["normalized"]["median"],
                record["pair_min"]["best_case"]["median"],
                record["pair_min"]["normalized"]["maximum"],
                record["frame_of_largest_normalized_pair_min"],
                corr["capture_vs_pair_min_normalized"]["spearman"],
                corr["capture_vs_pair_min_worst_case"]["spearman"],
            ]
            + [corr["capture_vs_normalized"][g]["spearman"] for g in result.table.groups]
            + [corr["capture_vs_normalized"][g]["pearson"] for g in result.table.groups]
        )
    return rows


def by_band_header(groups: Sequence[str]) -> List[str]:
    return (
        list(BY_BAND_HEADER)
        + [f"spearman_capture_vs_normalized_{g}" for g in groups]
        + [f"pearson_capture_vs_normalized_{g}" for g in groups]
    )


# --------------------------------------------------------------------------
# A focus band: marked frames and named windows
# --------------------------------------------------------------------------


@dataclass
class FrameMark:
    """A labelled set of frames, and how they were chosen."""

    label: str
    frames: List[int]
    rule: str
    role: str  # low_capture | high_capture_control | typical | most_mixed | user


def automatic_marks(
    table: ProjectionTable,
    arrays: SampleArrays,
    band: int,
    count: int = 3,
) -> List[FrameMark]:
    """Representative frames chosen by a rule a reader can re-apply.

    Ties are broken by frame number, so the selection is deterministic.
    """
    if count < 0:
        raise RobustnessError("the number of automatic marks cannot be negative")
    if count == 0:
        return []
    bi = table.band_position(band)
    frames = table.frames
    captured = table.captured[:, bi]
    order_low = np.lexsort((frames, captured))
    order_high = np.lexsort((frames, -captured))
    median = float(np.median(captured))
    order_median = np.lexsort((frames, np.abs(captured - median)))
    mixing = arrays.levels["normalized"][:, bi]
    order_mixed = np.lexsort((frames, -mixing))
    most_mixed = [int(frames[i]) for i in order_mixed[:count] if mixing[i] > 0.0]
    marks = [
        FrameMark(
            "lowest_capture",
            [int(frames[i]) for i in order_low[:count]],
            f"the {count} frames with the lowest captured_projection on band {band}",
            "low_capture",
        ),
        FrameMark(
            "highest_capture",
            [int(frames[i]) for i in order_high[:count]],
            f"the {count} frames with the highest captured_projection on band {band}; "
            "high-capture controls",
            "high_capture_control",
        ),
        FrameMark(
            "median_capture",
            [int(frames[order_median[0]])],
            f"the frame whose captured_projection is closest to band {band}'s median "
            f"({median:.4g})",
            "typical",
        ),
    ]
    if most_mixed:
        marks.append(
            FrameMark(
                "most_mixed_normalized",
                most_mixed,
                f"up to {count} frames with the largest normalized "
                f"min(w_a, w_b) on band {band}",
                "most_mixed",
            )
        )
    return marks


def parse_mark(spec: str) -> FrameMark:
    """``LABEL=F1,F2,...`` from the command line."""
    label, sep, rest = str(spec).partition("=")
    label = label.strip()
    if not sep or not label:
        raise RobustnessError(
            f"mark {spec!r} must look like LABEL=FRAME[,FRAME...], e.g. "
            "issue4_low_capture=1273,1286,1302"
        )
    if not all(c.isalnum() or c in "_-." for c in label):
        raise RobustnessError(f"mark label {label!r} must be alphanumeric with _ - .")
    try:
        frames = [int(item) for item in rest.split(",") if item.strip()]
    except ValueError:
        raise RobustnessError(f"mark {spec!r} has a non-integer frame") from None
    if not frames:
        raise RobustnessError(f"mark {spec!r} names no frames")
    return FrameMark(label, frames, "given on the command line", "user")


def check_marks(table: ProjectionTable, marks: Sequence[FrameMark]) -> None:
    present = set(int(f) for f in table.frames)
    labels = [mark.label for mark in marks]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise RobustnessError(f"mark labels must be distinct; repeated: {duplicates}")
    for mark in marks:
        missing = [f for f in mark.frames if f not in present]
        if missing:
            raise RobustnessError(
                f"mark {mark.label!r} names frames {missing} that are not in the "
                "projection table; nothing is marked by approximation"
            )


def check_windows(table: ProjectionTable, windows: Sequence[Any]) -> None:
    present = set(int(f) for f in table.frames)
    for window in windows:
        inside = [f for f in range(window.first, window.last + 1) if f in present]
        if not inside:
            raise RobustnessError(
                f"window {window.name} ({window.first}:{window.last}) holds no frame of "
                "the projection table. A window belongs to one configuration's "
                "trajectory; check it was not taken from another configuration"
            )


def focus_header(groups: Sequence[str], thresholds: Sequence[float] = ()) -> List[str]:
    return sample_header(groups, thresholds) + [
        "capture_rank_in_band",
        "capture_percentile_in_band",
        "windows",
        "window_roles",
        "marks",
    ]


def focus_rows(
    result: RobustnessAudit,
    band: int,
    marks: Sequence[FrameMark],
    windows: Sequence[Any],
) -> List[List[Any]]:
    """Every frame of one band, with its window and mark labels attached."""
    table = result.table
    bi = table.band_position(band)
    single = ProjectionTable(
        frames=table.frames,
        bands=table.bands[bi : bi + 1],
        groups=table.groups,
        weights=table.weights[:, bi : bi + 1, :],
        captured=table.captured[:, bi : bi + 1],
        total=None if table.total is None else table.total[:, bi : bi + 1],
    )
    base = list(sample_rows(single, result.pair,
                            thresholds=result.grids["normalized_thresholds"]))
    captured = table.captured[:, bi]
    ranks = _ranks(captured)
    rows = []
    for fi, row in enumerate(base):
        frame = int(table.frames[fi])
        in_windows = [w for w in windows if w.first <= frame <= w.last]
        labels = [m.label for m in marks if frame in m.frames]
        rows.append(
            row
            + [
                int(round(ranks[fi])),
                float(100.0 * (ranks[fi] - 1) / max(1, len(captured) - 1)),
                ";".join(w.name for w in in_windows),
                ";".join(w.role for w in in_windows),
                ";".join(labels),
            ]
        )
    return rows


def sample_record(result: RobustnessAudit, frame: int, band: int) -> Dict[str, Any]:
    """Every audited quantity for one (frame, band), as a dictionary."""
    table = result.table
    bi = table.band_position(band)
    lookup = {int(f): i for i, f in enumerate(table.frames)}
    if int(frame) not in lookup:
        raise RobustnessError(f"frame {frame} is not in the projection table")
    fi = lookup[int(frame)]
    header = sample_header(table.groups)
    single = ProjectionTable(
        frames=table.frames[fi : fi + 1],
        bands=table.bands[bi : bi + 1],
        groups=table.groups,
        weights=table.weights[fi : fi + 1, bi : bi + 1, :],
        captured=table.captured[fi : fi + 1, bi : bi + 1],
        total=None if table.total is None else table.total[fi : fi + 1, bi : bi + 1],
    )
    row = next(iter(sample_rows(single, result.pair)))
    record = dict(zip(header, row))
    captured = table.captured[:, bi]
    record["capture_percentile_in_band"] = float(
        100.0 * np.mean(captured < captured[fi])
    )
    record["allocation_class"] = {
        _tau_key(tau): allocation_class(
            record["pair_min_worst_case"], record["pair_min_best_case"], tau
        )
        for tau in result.grids["normalized_thresholds"]
    }
    return record


def focus_summary(
    result: RobustnessAudit,
    band: int,
    marks: Sequence[FrameMark],
    windows: Sequence[Any],
) -> Dict[str, Any]:
    """The focus band's audit, with marked frames and windows spelled out."""
    table = result.table
    bi = table.band_position(band)
    frames = table.frames
    levels = {key: value[:, bi] for key, value in result.arrays.levels.items()}
    window_records = []
    for window in windows:
        inside = (frames >= window.first) & (frames <= window.last)
        captured = table.captured[inside, bi]
        entry: Dict[str, Any] = {
            "name": window.name,
            "first": window.first,
            "last": window.last,
            "role": window.role,
            "n_frames_in_table": int(np.count_nonzero(inside)),
            "capture_median_inside": float(np.median(captured)),
            "capture_minimum_inside": float(np.min(captured)),
            "capture_median_outside": (
                float(np.median(table.captured[~inside, bi])) if np.any(~inside) else None
            ),
            "pair_min_normalized_maximum_inside": float(np.max(levels["normalized"][inside])),
            "pair_min_worst_case_maximum_inside": float(np.max(levels["worst_case"][inside])),
            "mixed_normalized_inside": {},
            "mixed_normalized_outside": {},
        }
        for tau in result.grids["normalized_thresholds"]:
            mixed = _mixed(levels, tau, "normalized")
            entry["mixed_normalized_inside"][_tau_key(tau)] = int(np.count_nonzero(mixed & inside))
            entry["mixed_normalized_outside"][_tau_key(tau)] = int(
                np.count_nonzero(mixed & ~inside)
            )
        if window.role == "control":
            entry["note"] = (
                "a CONTROL window: chosen for comparison. It is not an avoided "
                "crossing and not a character-exchange or transfer event"
            )
        window_records.append(entry)

    return {
        "band": int(band),
        "pair": list(result.pair),
        "band_audit": result.band(band),
        "marks": [
            {
                "label": mark.label,
                "role": mark.role,
                "rule": mark.rule,
                "frames": [sample_record(result, frame, band) for frame in mark.frames],
            }
            for mark in marks
        ],
        "windows": window_records,
        "windows_note": (
            "windows are marked only when they belong to the configuration whose "
            "projection table this is. Crossing windows of another configuration "
            "(for example B1-B4 on configuration B) are regions of a different "
            "trajectory and are never drawn on this one"
        ),
    }


def question_block(result: RobustnessAudit, bands: Sequence[int]) -> Dict[str, Any]:
    """The numbers that bear on 'mixed only when the absolute weight is small?'.

    Nothing here answers the question with a word.  For each focus band it
    lays side by side: how many samples are mixed after normalization; how
    many of those also clear the same threshold in the raw weights; how many
    sit in the band's lowest-capture stratum; and the capture of mixed against
    unmixed samples.  The reader reads the answer off those.
    """
    records = []
    for band in bands:
        summary = result.band(band)
        rows = []
        for entry in summary["mixing_by_threshold"]:
            rows.append(
                {
                    "threshold": entry["threshold"],
                    "n_mixed_normalized": entry["n_mixed_normalized"],
                    "n_that_survive_in_raw_weights": entry[
                        "n_normalized_mixed_that_survive_worst_case"
                    ],
                    "n_in_lowest_capture_stratum": entry["n_mixed_in_lowest_capture_stratum"],
                    "median_capture_mixed": entry["median_capture_mixed"],
                    "median_capture_not_mixed": entry["median_capture_not_mixed"],
                    "n_mixed_under_every_allocation": entry["n_mixed_under_every_allocation"],
                    "n_mixed_under_some_allocation": entry["n_mixed_under_some_allocation"],
                }
            )
        records.append(
            {
                "band": int(band),
                "capture_median": summary["capture"]["median"],
                "spearman_capture_vs_pair_min_normalized": summary["correlations"][
                    "capture_vs_pair_min_normalized"
                ]["spearman"],
                "by_threshold": rows,
            }
        )
    return {
        "question": (
            "Does the band look mixed between the pair only when its absolute "
            "projected weight is small?"
        ),
        "how_to_read": (
            "If the mixing were created by normalization, normalized-mixed samples "
            "would fail the same threshold in their raw weights, and would sit in "
            "the lowest-capture stratum. If it is carried by the raw weights, they "
            "survive the raw test and occur across capture strata. A band can "
            "also fall between the two. " + INTERPRETATION_NOTE
        ),
        "bands": records,
    }
