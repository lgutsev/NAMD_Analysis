"""First-passage branching probabilities from a fitted rate matrix.

The question this answers is "of the carriers that leave a given state, what
fraction reaches one set of states before another" -- for example, whether a
carrier sitting on an interfacial state reaches the acceptor manifold before
it reaches the valence band.  Accumulated population answers that badly,
because back-transfer erases accumulation: a carrier can reach the acceptor,
return, and leave again, and the final occupancy records none of it.

Everything here is ``model_inferred``.  It is computed from a rate matrix that
was fitted to averaged populations, so it inherits every assumption of that
fit -- the declared groups, the declared graph, constant Markovian rates, and
the fit window -- and every identifiability limit as well.  A branching
probability computed from rates the data did not determine is not a result,
and this module refuses to present one as though it were.

It is emphatically not an event count.  Averaged SHPROP populations do not
record individual hops, so nothing here counts how many carriers actually made
a given transition; see :mod:`namd_analysis.kinetics` and the scope document.

Convention.  The package writes the master equation as ``dP/dt = K P`` with
``K[i, j] = k_{j->i}``, so columns of ``K`` sum to zero.  The generator in the
usual row-stochastic sense is therefore ``Q = K.T``, with ``Q[i, j] = k_{i->j}``
and rows summing to zero.  Hitting probabilities satisfy ``Q h = 0`` on the
transient states, which is the ``K^T h = 0`` of the backward equation, with
``h = 1`` on the success set and ``h = 0`` on the failure set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .observables import COUNTERFACTUAL, MODEL_INFERRED

#: A branching probability is only given a confidence interval when it was
#: computable and identifiable in at least this fraction of resamples.
BRANCH_MIN_IDENTIFIED_FRACTION = 0.8

#: Identification verdicts for a branching observable.
IDENTIFIED = "identified"
WEAKLY_IDENTIFIED = "weakly_identified"
NOT_IDENTIFIABLE = "not_identifiable"


class BranchingError(ValueError):
    """Raised when a branching question is not well posed for this network."""


def _resolve(names: Sequence[str], groups: Sequence[str], what: str) -> List[int]:
    index = {name: i for i, name in enumerate(groups)}
    resolved = []
    for name in names:
        if name not in index:
            raise BranchingError(
                f"{what} names {name!r}, which is not one of the declared groups "
                f"{list(groups)}"
            )
        if index[name] in resolved:
            raise BranchingError(f"{what} lists {name!r} twice")
        resolved.append(index[name])
    return resolved


def _validate_sets(
    K: np.ndarray,
    groups: Sequence[str],
    success: Sequence[str],
    failure: Sequence[str],
) -> Tuple[np.ndarray, List[str], List[int], List[int]]:
    K = np.asarray(K, dtype=float)
    groups = list(groups)
    if K.ndim != 2 or K.shape[0] != K.shape[1] or K.shape[0] != len(groups):
        raise BranchingError(
            f"rate matrix has shape {K.shape}, expected ({len(groups)}, {len(groups)})"
        )
    if not np.all(np.isfinite(K)):
        raise BranchingError("the rate matrix contains non-finite entries")
    if not success:
        raise BranchingError("the success set is empty")
    if not failure:
        raise BranchingError("the failure set is empty")
    success_idx = _resolve(success, groups, "the success set")
    failure_idx = _resolve(failure, groups, "the failure set")
    overlap = set(success_idx) & set(failure_idx)
    if overlap:
        clash = sorted(groups[i] for i in overlap)
        raise BranchingError(
            f"{clash} appear in both the success and the failure set; a state "
            "cannot be both outcomes"
        )
    return K, groups, success_idx, failure_idx


def _can_reach(K: np.ndarray, targets: Sequence[int]) -> set:
    """States from which some path leads into ``targets`` (targets included)."""
    n = K.shape[0]
    reached = set(targets)
    stack = list(targets)
    while stack:
        current = stack.pop()
        # K[current, source] > 0 means source -> current, so walk edges backwards.
        for source in range(n):
            if source in reached or source == current:
                continue
            if K[current, source] > 0.0:
                reached.add(source)
                stack.append(source)
    return reached


def hitting_probabilities(
    K: np.ndarray,
    groups: Sequence[str],
    success: Sequence[str],
    failure: Sequence[str],
) -> Dict[str, float]:
    """P(reach ``success`` before ``failure``) from every group.

    Solves the backward equation on the transient states that can reach either
    absorbing set at all.  A transient state that can reach neither is not an
    error and does not make the question ill posed for the others: it simply
    never hits the success set, so its probability is zero.  Because such a
    state never hits the failure set either, the success and failure
    probabilities need not sum to one; :func:`absorption_report` accounts for
    the remainder.
    """
    K, groups, success_idx, failure_idx = _validate_sets(K, groups, success, failure)
    absorbing = set(success_idx) | set(failure_idx)
    transient = [i for i in range(len(groups)) if i not in absorbing]

    # Q[i, j] = k_{i->j}; rows sum to zero.
    Q = K.T
    values = np.zeros(len(groups), dtype=float)
    for i in success_idx:
        values[i] = 1.0

    reaching = _can_reach(K, sorted(absorbing))
    solvable = [i for i in transient if i in reaching]
    if solvable:
        A = Q[np.ix_(solvable, solvable)]
        b = -Q[np.ix_(solvable, success_idx)].sum(axis=1)
        try:
            solved = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
            raise BranchingError(
                "the transient block could not be solved even after removing "
                f"states that cannot be absorbed: {exc}"
            ) from exc
        for position, i in enumerate(solvable):
            values[i] = float(solved[position])

    return {name: float(values[i]) for i, name in enumerate(groups)}


def absorption_report(
    K: np.ndarray,
    groups: Sequence[str],
    success: Sequence[str],
    failure: Sequence[str],
) -> Dict[str, Any]:
    """Success, failure and never-absorbed probabilities from every group."""
    K, groups, success_idx, failure_idx = _validate_sets(K, groups, success, failure)
    forward = hitting_probabilities(K, groups, success, failure)
    backward = hitting_probabilities(K, groups, failure, success)
    absorbing = set(success_idx) | set(failure_idx)
    reaching = _can_reach(K, sorted(absorbing))
    stranded = [
        groups[i] for i in range(len(groups)) if i not in absorbing and i not in reaching
    ]
    unresolved = {
        name: float(max(0.0, 1.0 - forward[name] - backward[name])) for name in groups
    }
    return {
        "success": forward,
        "failure": backward,
        "unresolved": unresolved,
        "stranded_groups": stranded,
        "stranded_note": (
            f"{stranded} can reach neither outcome, so from them both "
            "probabilities are zero and the remainder is unresolved"
            if stranded
            else None
        ),
    }


def _reachable_transient(
    K: np.ndarray, source: int, absorbing: Sequence[int]
) -> List[int]:
    """Transient states reachable from ``source`` without passing an absorber."""
    absorbing = set(absorbing)
    n = K.shape[0]
    seen = set()
    stack = [source]
    while stack:
        current = stack.pop()
        if current in seen or current in absorbing:
            continue
        seen.add(current)
        for target in range(n):
            if target == current:
                continue
            if K[target, current] > 0.0 and target not in seen:
                stack.append(target)
    return sorted(seen)


def local_branching_ratio(
    K: np.ndarray,
    groups: Sequence[str],
    source: str,
    success: Sequence[str],
    failure: Sequence[str],
) -> Dict[str, Any]:
    """The competing-channel ratio k_S / (k_S + k_F), when it is exact.

    This equals the first-passage probability only when *every* transition out
    of ``source`` lands directly in the success or the failure set.  With any
    other exit the carrier can leave, come back and try again, and the simple
    ratio is not the branching probability.  In that case no number is
    returned and the reason is stated.
    """
    groups = list(groups)
    K = np.asarray(K, dtype=float)
    source_idx = _resolve([source], groups, "the source")[0]
    success_idx = set(_resolve(success, groups, "the success set"))
    failure_idx = set(_resolve(failure, groups, "the failure set"))

    exits = {
        target: float(K[target, source_idx])
        for target in range(len(groups))
        if target != source_idx and K[target, source_idx] > 0.0
    }
    if not exits:
        return {
            "available": False,
            "reason": f"no transition leaves {source!r}, so there is nothing to branch",
            "ratio": None,
        }
    other = sorted(
        groups[t] for t in exits if t not in success_idx and t not in failure_idx
    )
    if other:
        return {
            "available": False,
            "reason": (
                f"{source!r} also exits to {other}, so a carrier can leave and "
                "return before being absorbed. The competing-channel ratio is "
                "not the branching probability for this topology; use the "
                "first-passage probability instead"
            ),
            "ratio": None,
            "other_exits": other,
        }
    to_success = sum(rate for target, rate in exits.items() if target in success_idx)
    to_failure = sum(rate for target, rate in exits.items() if target in failure_idx)
    total = to_success + to_failure
    if total <= 0:
        return {
            "available": False,
            "reason": "the competing rates are zero",
            "ratio": None,
        }
    return {
        "available": True,
        "reason": None,
        "ratio": float(to_success / total),
        "complement": float(to_failure / total),
        "rate_to_success_per_ns": float(to_success),
        "rate_to_failure_per_ns": float(to_failure),
    }


@dataclass
class BranchResult:
    """P(success before failure | source), with its identification status."""

    source: str
    success: List[str]
    failure: List[str]
    probability: float
    complement: float
    unresolved: float
    stranded_groups: List[str]
    hitting_probabilities: Dict[str, float]
    contributing_rates: List[str]
    status: str
    status_reason: Optional[str]
    local_ratio: Dict[str, Any] = field(default_factory=dict)
    bootstrap: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "success_set": self.success,
            "failure_set": self.failure,
            "probability_success_before_failure": self.probability,
            "probability_failure_before_success": self.complement,
            "probability_never_absorbed": self.unresolved,
            "stranded_groups": self.stranded_groups,
            "hitting_probabilities_by_group": self.hitting_probabilities,
            "contributing_rates": self.contributing_rates,
            "branching_status": self.status,
            "branching_status_reason": self.status_reason,
            "local_branching_ratio": self.local_ratio,
            "bootstrap": self.bootstrap,
            "observable_class": {
                "probability_success_before_failure": MODEL_INFERRED,
                "local_branching_ratio": MODEL_INFERRED,
            },
            "interpretation": (
                "the probability that a carrier starting in the source group "
                "reaches the success set before the failure set, under the "
                "fitted rate matrix. It is a property of the fitted model, not "
                "a counted number of transitions, and it is not a device "
                "extraction efficiency"
            ),
        }


def _classify(
    fit, contributing: Sequence[str]
) -> Tuple[str, Optional[str]]:
    """Grade a branching observable by the identification of the rates it uses."""
    by_name = {r.name: r for r in fit.rates}
    missing = [name for name in contributing if name not in by_name]
    if missing:
        return NOT_IDENTIFIABLE, f"the fit carries no estimate for {missing}"

    fatal: List[str] = []
    soft: List[str] = []
    for name in contributing:
        rate = by_name[name]
        if rate.at_optimizer_bound:
            fatal.append(f"{name} sits on the optimizer {rate.optimizer_bound} bound")
        elif rate.null_space_identifiable is False:
            fatal.append(f"{name} lies in the null space of the Jacobian")
        elif not rate.identified:
            soft.append(f"{name} ({rate.unidentified_reason or 'not identified'})")
    if fatal:
        return NOT_IDENTIFIABLE, (
            "the branching probability depends on rates the data does not "
            "determine at all: " + "; ".join(fatal)
        )
    if soft:
        return WEAKLY_IDENTIFIED, (
            "every contributing rate is constrained by the data, but not all of "
            "them individually: " + "; ".join(soft)
            + ". The branching probability may still be determined if it depends "
            "on a combination the data does constrain; check the bootstrap"
        )
    return IDENTIFIED, None


def branch_probability(
    fit,
    source: str,
    success: Sequence[str],
    failure: Sequence[str],
) -> BranchResult:
    """First-passage branching probability from a fitted kinetic model."""
    groups = list(fit.groups)
    success = list(success)
    failure = list(failure)
    source_idx = _resolve([source], groups, "the source")[0]
    success_idx = _resolve(success, groups, "the success set")
    failure_idx = _resolve(failure, groups, "the failure set")

    report = absorption_report(fit.K, groups, success, failure)
    probabilities = report["success"]
    value = probabilities[source]
    complement = report["failure"][source]
    unresolved = report["unresolved"][source]

    absorbing = set(success_idx) | set(failure_idx)
    if source_idx in absorbing:
        contributing: List[str] = []
        status, reason = (
            IDENTIFIED,
            "the source is itself an absorbing state, so the probability is "
            "exactly 1 or 0 and depends on no fitted rate",
        )
    else:
        reachable = set(_reachable_transient(fit.K, source_idx, sorted(absorbing)))
        contributing = [
            r.name
            for r in fit.rates
            if groups.index(r.source) in reachable
        ]
        status, reason = _classify(fit, contributing)

    return BranchResult(
        source=source,
        success=success,
        failure=failure,
        probability=float(value),
        complement=float(complement),
        unresolved=float(unresolved),
        stranded_groups=list(report["stranded_groups"]),
        hitting_probabilities=probabilities,
        contributing_rates=contributing,
        status=status,
        status_reason=reason,
        local_ratio=local_branching_ratio(fit.K, groups, source, success, failure),
    )


#: Statuses whose branch value is allowed into a bootstrap distribution.
ACCEPTABLE_STATUSES = (IDENTIFIED, WEAKLY_IDENTIFIED)


def bootstrap_branch_probability(
    time_ns: np.ndarray,
    per_file_groups: np.ndarray,
    groups: Sequence[str],
    edges: Sequence[Tuple[int, int]],
    source: str,
    success: Sequence[str],
    failure: Sequence[str],
    n_resamples: int = 200,
    seed: int = 0,
    weights: Optional[np.ndarray] = None,
    min_identified_fraction: float = BRANCH_MIN_IDENTIFIED_FRACTION,
    percentiles: Tuple[float, float] = (2.5, 97.5),
    accept_statuses: Sequence[str] = ACCEPTABLE_STATUSES,
) -> Dict[str, Any]:
    """Resample whole input files and re-derive the branching probability.

    The branching observable is bootstrapped directly rather than assembled
    from separately bootstrapped rates: a probability built from the endpoints
    of independent rate intervals is not an interval for the probability.

    Every resample is refitted and put through the same identifiability tests
    as the point estimate.  A resample whose model does not support the branch
    contributes nothing to the interval and is counted instead, so an interval
    can never be made to look tight by quietly discarding the draws that
    disagreed.
    """
    from .kinetics import KineticsError, fit_master_equation

    per_file_groups = np.asarray(per_file_groups, dtype=float)
    if per_file_groups.ndim != 3:
        raise BranchingError(
            "per_file_groups must have shape (nfiles, ntime, ngroups)"
        )
    n_files = per_file_groups.shape[0]
    if n_files < 2:
        raise BranchingError("bootstrapping whole files needs at least two files")
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != per_file_groups.shape[1:]:
            raise BranchingError(
                f"weights have shape {weights.shape}, expected "
                f"{per_file_groups.shape[1:]}"
            )

    rng = np.random.default_rng(seed)
    values: List[float] = []
    statuses: Dict[str, int] = {}
    fit_successes = 0
    fit_failures = 0
    for _ in range(int(n_resamples)):
        picks = rng.integers(0, n_files, size=n_files)
        sample = per_file_groups[picks].mean(axis=0)
        try:
            fit = fit_master_equation(
                time_ns, sample, groups, edges, weights=weights, n_starts=2
            )
        except (KineticsError, ValueError, np.linalg.LinAlgError):
            fit_failures += 1
            continue
        fit_successes += 1
        try:
            result = branch_probability(fit, source, success, failure)
        except BranchingError:
            statuses["branch_not_well_posed"] = (
                statuses.get("branch_not_well_posed", 0) + 1
            )
            continue
        statuses[result.status] = statuses.get(result.status, 0) + 1
        if result.status in accept_statuses:
            values.append(result.probability)

    branch_successes = len(values)
    fraction = (
        float(branch_successes / fit_successes) if fit_successes else float("nan")
    )

    low = high = None
    if branch_successes < 20:
        status = "suppressed: fewer than 20 resamples produced a usable branch"
    elif not np.isfinite(fraction) or fraction < min_identified_fraction:
        status = (
            f"suppressed: the transition was identifiable in only "
            f"{100 * fraction:.0f}% of the {fit_successes} successful resamples, "
            f"below the {100 * min_identified_fraction:.0f}% required"
        )
    else:
        low, high = (float(v) for v in np.percentile(values, percentiles))
        status = "reported"

    return {
        "bootstrap_requested": int(n_resamples),
        "bootstrap_fit_successes": fit_successes,
        "bootstrap_fit_failures": fit_failures,
        "bootstrap_branch_successes": branch_successes,
        "branch_identified_fraction": fraction,
        "branch_status_counts": statuses,
        "branch_ci_low": low,
        "branch_ci_high": high,
        "branch_ci_status": status,
        "percentiles": list(percentiles),
        "minimum_identified_fraction": float(min_identified_fraction),
        "n_files": int(n_files),
        "weighted": weights is not None,
        "caveat": (
            "this is the spread between the input files supplied, which share a "
            "trajectory and often correlated initial conditions. It is not an "
            "ensemble error bar, and it says nothing about whether the Markovian "
            "model is right"
        ),
    }


# ---------------------------------------------------------------------------
# Extraction versus recombination


#: Name given to the absorbing state that represents onward transport.
EXTRACTED = "Extracted"


def augment_with_sink(
    K: np.ndarray, groups: Sequence[str], sink_group: str, k_escape: float
) -> Tuple[np.ndarray, List[str]]:
    """Add an absorbing onward-escape channel draining ``sink_group``."""
    groups = list(groups)
    if EXTRACTED in groups:
        raise BranchingError(f"a group is already called {EXTRACTED!r}")
    if k_escape < 0 or not np.isfinite(k_escape):
        raise BranchingError(f"escape rate {k_escape} is negative or non-finite")
    index = _resolve([sink_group], groups, "the sink group")[0]
    n = len(groups)
    augmented = np.zeros((n + 1, n + 1), dtype=float)
    augmented[:n, :n] = np.asarray(K, dtype=float)
    augmented[n, index] += k_escape
    augmented[index, index] -= k_escape
    return augmented, groups + [EXTRACTED]


@dataclass
class CompetitionPoint:
    k_escape_per_ns: float
    escape_time_ns: float
    extracted_yield: float
    recombined_yield: float
    unresolved: float

    def as_row(self) -> List[Any]:
        return [
            self.k_escape_per_ns,
            self.escape_time_ns,
            self.extracted_yield,
            self.recombined_yield,
            self.unresolved,
        ]


COMPETITION_HEADER = [
    "k_escape_per_ns",
    "escape_time_ns",
    "extracted_yield",
    "recombined_yield",
    "unresolved",
]


def _yields(
    K: np.ndarray,
    groups: Sequence[str],
    sink_group: str,
    recombined: Sequence[str],
    k_escape: float,
    p0: np.ndarray,
) -> Tuple[float, float, float]:
    augmented, names = augment_with_sink(K, groups, sink_group, k_escape)
    weights = np.zeros(len(names))
    weights[: len(groups)] = p0
    extracted = hitting_probabilities(augmented, names, [EXTRACTED], list(recombined))
    recombining = hitting_probabilities(augmented, names, list(recombined), [EXTRACTED])
    y_ext = float(sum(weights[i] * extracted[name] for i, name in enumerate(names)))
    y_rec = float(sum(weights[i] * recombining[name] for i, name in enumerate(names)))
    return y_ext, y_rec, float(max(0.0, 1.0 - y_ext - y_rec))


def extraction_competition(
    fit,
    sink_group: str,
    recombined_groups: Sequence[str],
    k_escape_per_ns: Sequence[float],
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Extraction against recombination as a function of an assumed escape rate.

    Onward transport out of ``sink_group`` is added as an absorbing channel at
    rate ``k_escape``, and recombination is treated as the competing absorbing
    outcome.  For each escape rate the first-passage yields of both outcomes
    are computed from the same generator, so the answer is a competition
    between the two, not an integral over a sink-free solution.

    Every number produced here is ``counterfactual``.  ``k_escape`` is supplied
    by the caller; nothing in the interface calculation determines it, and the
    transfer rates were fitted to data containing no extraction at all.  The
    crossover is the escape rate that onward transport *would have to* achieve,
    not a measured collection time.
    """
    groups = list(fit.groups)
    recombined = list(recombined_groups)
    _resolve([sink_group], groups, "the sink group")
    _resolve(recombined, groups, "the recombination set")
    if sink_group in recombined:
        raise BranchingError(
            f"{sink_group!r} is named as both the extraction source and a "
            "recombination outcome"
        )

    if source is None:
        p0 = np.asarray(fit.p0, dtype=float)
        source_label = "the observed initial populations"
    else:
        index = _resolve([source], groups, "the source")[0]
        p0 = np.zeros(len(groups))
        p0[index] = 1.0
        source_label = f"all population starting in {source}"

    rates = [float(k) for k in k_escape_per_ns]
    if not rates:
        raise BranchingError("no escape rates were given")
    points: List[CompetitionPoint] = []
    for k in rates:
        y_ext, y_rec, rest = _yields(fit.K, groups, sink_group, recombined, k, p0)
        points.append(
            CompetitionPoint(
                k_escape_per_ns=k,
                escape_time_ns=float(1.0 / k) if k > 0 else float("inf"),
                extracted_yield=y_ext,
                recombined_yield=y_rec,
                unresolved=rest,
            )
        )

    crossover = _crossover(fit, groups, sink_group, recombined, p0, points)
    monotonic = all(
        points[i + 1].extracted_yield >= points[i].extracted_yield - 1e-9
        for i in range(len(points) - 1)
    )

    return {
        "sink_group": sink_group,
        "recombination_set": recombined,
        "source": source_label,
        "points": [
            {
                "k_escape_per_ns": p.k_escape_per_ns,
                "escape_time_ns": p.escape_time_ns,
                "extracted_yield": p.extracted_yield,
                "recombined_yield": p.recombined_yield,
                "unresolved": p.unresolved,
            }
            for p in points
        ],
        "rows": [p.as_row() for p in points],
        "extracted_yield_is_monotonic_in_k_escape": monotonic,
        **crossover,
        "observable_class": {
            "extracted_yield": COUNTERFACTUAL,
            "recombined_yield": COUNTERFACTUAL,
            "required_escape_rate_per_ns": COUNTERFACTUAL,
            "required_escape_time_ns": COUNTERFACTUAL,
        },
        "interpretation": (
            "the escape rate is assumed, not measured. These yields say how "
            "fast onward transport out of the interfacial acceptor would have "
            "to be for extraction to outcompete recombination under the fitted "
            "rates. The simulated cell contains no electrode and no long-range "
            "transport, so this is a requirement on the transport layer and not "
            "a device collection efficiency"
        ),
    }


def _crossover(
    fit,
    groups: Sequence[str],
    sink_group: str,
    recombined: Sequence[str],
    p0: np.ndarray,
    points: Sequence[CompetitionPoint],
) -> Dict[str, Any]:
    """Find where the extracted and recombined yields cross, if they do."""

    def gap(k: float) -> float:
        y_ext, y_rec, _ = _yields(fit.K, groups, sink_group, recombined, k, p0)
        return y_ext - y_rec

    bracket: Optional[Tuple[float, float]] = None
    for a, b in zip(points, points[1:]):
        left = a.extracted_yield - a.recombined_yield
        right = b.extracted_yield - b.recombined_yield
        if left == 0.0:
            bracket = (a.k_escape_per_ns, a.k_escape_per_ns)
            break
        if left < 0.0 <= right:
            bracket = (a.k_escape_per_ns, b.k_escape_per_ns)
            break
    if bracket is None:
        first = points[0].extracted_yield - points[0].recombined_yield
        return {
            "required_escape_rate_per_ns": None,
            "required_escape_time_ns": None,
            "crossover_note": (
                "extraction already outcompetes recombination at the smallest "
                "escape rate on the grid"
                if first > 0
                else "extraction does not overtake recombination anywhere on the "
                "supplied grid; widen it if a crossover is expected"
            ),
        }
    low, high = bracket
    if low == high:
        root = low
    else:
        from scipy.optimize import brentq

        root = float(brentq(gap, low, high, xtol=1e-12, rtol=1e-10))
    return {
        "required_escape_rate_per_ns": float(root),
        "required_escape_time_ns": float(1.0 / root) if root > 0 else float("inf"),
        "crossover_note": (
            "the escape rate at which the extraction and recombination yields "
            "are equal. This is the onward transport speed that would be "
            "required, not one that was measured"
        ),
    }
