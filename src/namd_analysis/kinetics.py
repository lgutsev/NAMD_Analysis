"""Multistate kinetic fitting over declared population groups.

The model is a Markovian master equation on the declared groups,

    dP/dt = K P,    K_ij = rate j -> i  (i != j),    K_jj = -sum_{i != j} K_ij

so every column of K sums to zero and total population is conserved.  With a
constant K the solution is P(t) = exp(K t) P(0); the initial condition is taken
from the data rather than fitted.

Two things about this model deserve to be stated before any number from it is
quoted.

*It is an assumption, not a measurement.* Averaged populations are consistent
with a Markovian rate picture; they do not establish one. Memory effects,
inhomogeneity across initial conditions and states whose character changes
during the trajectory all break it, and none of them show up as a bad fit.

*Individual rates are often not identifiable.* Population curves constrain the
eigenvalues of K far better than they constrain its entries: many different
forward/backward pairs give almost the same P(t). This module therefore
reports the eigenvalue timescales separately from the rates, and flags rates
whose standard error or mutual correlation says the data did not determine
them. A rate flagged ``unidentified`` must not be quoted as a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

#: Named connectivity presets, resolved against the declared group order.
SCHEMES = ("dense", "sequential", "reversible")

#: A rate whose linearized relative standard error exceeds this is not treated
#: as determined.  The threshold is deliberately tight.  The asymptotic error is
#: a local linearization, and because P(0) is read from a noisy sample rather
#: than fitted, this package's own Monte Carlo (docs/validation.md) shows it
#: understating the true sampling spread by up to a factor of six.  At 0.1 even
#: a sixfold understatement leaves a rate known to better than a factor of two;
#: at the old value of 1.0 it did not.  Calibration showed no cost: across 200
#: well-posed fits, every correctly recovered rate has a relative error far
#: below this, so tightening the threshold rejected none of them.
UNIDENTIFIED_REL_SE = 0.1

#: Largest factor by which the linearized error was seen to understate the
#: empirical spread when P(0) comes from data.  Reported, not applied.
ASYMPTOTIC_UNDERSTATEMENT_FACTOR = 6.0

#: Two rates correlated above this are degenerate: the data sees their
#: combination, not each one.
DEGENERATE_CORRELATION = 0.95

#: Fraction of a parameter axis that may lie in the numerical null space of the
#: Jacobian before the rate is refused.  ``||V_null[j, :]||`` is 0 when the axis
#: is entirely inside the range space and 1 when the residuals are completely
#: blind to it; a rate sharing a two-way degeneracy sits near 1/sqrt(2).  At
#: 0.1, a hundredth of the axis lying in a blind direction is already enough to
#: refuse the rate, which is the intended asymmetry: the cost of wrongly
#: refusing a rate is a missing number, the cost of wrongly admitting one is a
#: number that is not a measurement.
NULL_PARTICIPATION_LIMIT = 0.1

#: Fraction of *successful* bootstrap resamples in which a rate must be
#: identified before a percentile interval for it is reported.  A rate the data
#: determines in a fifth of the draws has no interval worth quoting: the
#: percentiles would be taken over draws in which the optimizer's stopping
#: point, not the data, set the value.
BOOTSTRAP_MIN_IDENTIFIED_FRACTION = 0.8

#: Decades either side of ``1/window`` within which a rate may be fitted.  Wide
#: on purpose: the bound exists to keep the optimizer in a representable range,
#: not to express a physical prior, and a rate that reaches it is reported as
#: unidentified rather than quietly clamped.
RATE_BOUND_DECADES = 9.0


class KineticsError(ValueError):
    """Raised when a kinetic model cannot be built or fitted as requested."""


def null_space_analysis(jacobian: np.ndarray) -> Dict[str, Any]:
    """Rank, nullity and per-parameter null-space participation of a Jacobian.

    A rate can be unidentifiable even when its own Jacobian column is far from
    zero and it correlates with no single other rate: it is enough that some
    *combination* containing it leaves the residuals unchanged.  The right null
    space of J names exactly those combinations.

    The numerical rank uses ``sigma_max * max(shape) * sqrt(eps)``, which scales
    with the matrix rather than assuming an absolute scale for the residuals.
    The square root is not a fudge.  The covariance this feeds is obtained by
    inverting ``J^T J``, whose condition number is the *square* of J's, so the
    effective precision of that solve is ``sqrt(eps)``, not ``eps``.  A
    direction below this tolerance is one the pseudo-inverse will discard --
    and a pseudo-inverse reports **zero** variance for a discarded direction,
    not infinite, which would stamp exactly the least determined rates
    ``identified: true``.  Taking the rank at the precision of the computation
    that consumes it is what closes that gap.

    Returns ``rank``, ``nullity``, ``tolerance``, ``singular_values`` and
    ``participation``, where ``participation[j] = ||V_null[j, :]||`` lies in
    [0, 1]: zero when the axis of rate ``j`` is entirely inside the range
    space, one when the data is blind to that rate alone.
    """
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.ndim != 2:
        raise KineticsError("the Jacobian must be two-dimensional")
    n_parameters = jacobian.shape[1]
    if n_parameters == 0:
        return {
            "rank": 0,
            "nullity": 0,
            "tolerance": 0.0,
            "singular_values": [],
            "participation": np.zeros(0),
        }

    _, singular, vt = np.linalg.svd(jacobian, full_matrices=True)
    largest = float(singular.max()) if singular.size else 0.0
    tolerance = largest * max(jacobian.shape) * np.sqrt(np.finfo(float).eps)
    rank = int(np.count_nonzero(singular > tolerance))
    # ``vt`` is (n_parameters, n_parameters) with full_matrices=True, so the
    # rows past the rank span the right null space even when the Jacobian has
    # fewer rows than columns.
    null_basis = vt[rank:].T
    participation = (
        np.linalg.norm(null_basis, axis=1)
        if null_basis.size
        else np.zeros(n_parameters)
    )
    return {
        "rank": rank,
        "nullity": int(n_parameters - rank),
        "tolerance": float(tolerance),
        "singular_values": [float(value) for value in singular],
        "participation": np.clip(participation, 0.0, 1.0),
    }


def independent_residual_count(
    observed: np.ndarray, atol: float = 1.0e-5
) -> Tuple[int, int, bool]:
    """How many residual coordinates actually carry independent information.

    Two redundancies have to be taken out before a residual variance is scaled
    by a degrees-of-freedom count, or the covariance is divided by observations
    that were never free to disagree with the model.

    *Conservation.*  The rate matrix conserves total population by
    construction, so when the observed populations also sum to a constant the
    residual vector at every time is orthogonal to the all-ones direction.
    Only ``G - 1`` of its ``G`` coordinates are free.  This is the same
    ``G - 1`` subspace that ``compare_schemes`` scores through Helmert
    contrasts; because those contrasts are orthonormal, the sum of squares is
    identical in either representation and only the *count* changes.

    *The conditioned initial row.*  ``P(0)`` is read from the first sample and
    the model is started there, so the first row's residual is identically
    zero.  It is not an observation about the rates.

    Returns ``(n_independent, per_time, conserved)``.
    """
    observed = np.asarray(observed, dtype=float)
    n_points, n_groups = observed.shape
    totals = observed.sum(axis=1)
    conserved = bool(
        n_groups > 1 and np.allclose(totals, totals[0], atol=atol, rtol=0.0)
    )
    per_time = n_groups - 1 if conserved else n_groups
    return max((n_points - 1) * per_time, 1), per_time, conserved


# ---------------------------------------------------------------------------
# Model construction


def parse_edges(spec: str, groups: Sequence[str]) -> List[Tuple[int, int]]:
    """Resolve a scheme specification into ``(source, target)`` index pairs.

    ``spec`` is either a preset name or a comma-separated list of transitions
    written ``SOURCE->TARGET`` using the declared group names.
    """
    names = list(groups)
    index = {name: i for i, name in enumerate(names)}
    if len(index) != len(names):
        raise KineticsError("group names must be unique")

    text = spec.strip()
    if text == "dense":
        return [(j, i) for i in range(len(names)) for j in range(len(names)) if i != j]
    if text == "sequential":
        return [(i, i + 1) for i in range(len(names) - 1)]
    if text == "reversible":
        edges = []
        for i in range(len(names) - 1):
            edges.append((i, i + 1))
            edges.append((i + 1, i))
        return edges

    edges: List[Tuple[int, int]] = []
    seen = set()
    for token in text.split(","):
        piece = token.strip()
        if not piece:
            continue
        if "->" not in piece:
            raise KineticsError(
                f"transition {piece!r} is not of the form SOURCE->TARGET, and is "
                f"not one of the presets {SCHEMES}"
            )
        source, _, target = piece.partition("->")
        source, target = source.strip(), target.strip()
        for name in (source, target):
            if name not in index:
                raise KineticsError(
                    f"transition {piece!r} names {name!r}, which is not a declared "
                    f"group; groups are {names}"
                )
        if source == target:
            raise KineticsError(f"transition {piece!r} is a self-loop")
        pair = (index[source], index[target])
        if pair in seen:
            raise KineticsError(f"transition {piece!r} is listed twice")
        seen.add(pair)
        edges.append(pair)
    if not edges:
        raise KineticsError("no transitions were declared")
    return edges


def edge_names(edges: Sequence[Tuple[int, int]], groups: Sequence[str]) -> List[str]:
    return [f"{groups[s]}->{groups[t]}" for s, t in edges]


def build_rate_matrix(
    rates: Sequence[float], edges: Sequence[Tuple[int, int]], n_groups: int
) -> np.ndarray:
    """Assemble K from the rates on the allowed edges."""
    if len(rates) != len(edges):
        raise KineticsError(f"{len(rates)} rates for {len(edges)} edges")
    K = np.zeros((n_groups, n_groups), dtype=float)
    for rate, (source, target) in zip(rates, edges):
        K[target, source] += rate
        K[source, source] -= rate
    return K


def propagate(K: np.ndarray, p0: np.ndarray, time: np.ndarray) -> np.ndarray:
    """P(t) = exp(K t) P(0) on a uniform time grid, shape ``(ntime, ngroups)``."""
    time = np.asarray(time, dtype=float)
    if time.size < 2:
        raise KineticsError("at least two time points are needed")
    steps = np.diff(time)
    if not np.allclose(steps, steps[0], rtol=1e-8, atol=0.0):
        raise KineticsError(
            "the time grid is not uniform; the propagator is built once per step "
            "and cannot be reused on an irregular grid"
        )
    propagator = expm(K * steps[0])
    out = np.empty((time.size, K.shape[0]), dtype=float)
    out[0] = p0
    for n in range(1, time.size):
        out[n] = propagator @ out[n - 1]
    return out


# ---------------------------------------------------------------------------
# Fitting


@dataclass
class RateEstimate:
    name: str
    source: str
    target: str
    rate_per_ns: float
    stderr_per_ns: Optional[float]
    relative_stderr: Optional[float]
    lifetime_ns: float
    identified: bool
    degenerate_with: List[str] = field(default_factory=list)
    unidentified_reason: Optional[str] = None
    bootstrap_ci_per_ns: Optional[List[float]] = None
    #: ``||V_null[j, :]||``: how much of this rate's parameter axis lies in the
    #: numerical null space of the Jacobian.
    null_space_participation: Optional[float] = None
    null_space_identifiable: Optional[bool] = None
    #: The optimizer stopped on the edge of the allowed log-rate range.
    at_optimizer_bound: bool = False
    optimizer_bound: Optional[str] = None
    bootstrap_successes: Optional[int] = None
    bootstrap_identified: Optional[int] = None
    bootstrap_identified_fraction: Optional[float] = None
    bootstrap_ci_status: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "transition": self.name,
            "source": self.source,
            "target": self.target,
            "rate_per_ns": self.rate_per_ns,
            "stderr_per_ns": self.stderr_per_ns,
            "relative_stderr": self.relative_stderr,
            "lifetime_ns": self.lifetime_ns,
            "identified": self.identified,
            "degenerate_with": self.degenerate_with,
            "unidentified_reason": self.unidentified_reason,
            "null_space_participation": self.null_space_participation,
            "null_space_identifiable": self.null_space_identifiable,
            "at_optimizer_bound": self.at_optimizer_bound,
            "optimizer_bound": self.optimizer_bound,
            "bootstrap_ci_per_ns": self.bootstrap_ci_per_ns,
            "bootstrap_successes": self.bootstrap_successes,
            "bootstrap_identified": self.bootstrap_identified,
            "bootstrap_identified_fraction": self.bootstrap_identified_fraction,
            "bootstrap_ci_status": self.bootstrap_ci_status,
        }


@dataclass
class KineticFit:
    groups: List[str]
    edges: List[Tuple[int, int]]
    rates: List[RateEstimate]
    K: np.ndarray
    time_ns: np.ndarray
    observed: np.ndarray
    model: np.ndarray
    p0: np.ndarray
    window: Tuple[float, float]
    cost: float
    r_squared_total: float
    r_squared_per_group: Dict[str, float]
    rms_residual_per_group: Dict[str, float]
    eigen_timescales_ns: List[float]
    condition_number: float
    rank_deficient: bool
    correlation: Optional[np.ndarray]
    n_points: int
    n_parameters: int
    jacobian_rank: int = 0
    jacobian_nullity: int = 0
    null_space_tolerance: float = 0.0
    n_independent_residuals: int = 0
    independent_coordinates_per_time: int = 0
    population_conserved: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def identified_rates(self) -> List[RateEstimate]:
        return [r for r in self.rates if r.identified]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "groups": self.groups,
            "scheme_edges": edge_names(self.edges, self.groups),
            "window_ns": list(self.window),
            "n_time_points": self.n_points,
            "n_parameters": self.n_parameters,
            "initial_population": {
                name: float(value) for name, value in zip(self.groups, self.p0)
            },
            "rates": [rate.as_dict() for rate in self.rates],
            "rate_matrix_per_ns": self.K.tolist(),
            "eigen_timescales_ns": self.eigen_timescales_ns,
            "eigen_timescales_note": (
                "relaxation times from the eigenvalues of K. These are what the "
                "population curves actually constrain; individual rates may not be "
                "determined even when these are"
            ),
            "fit_quality": {
                "cost": self.cost,
                "r_squared_total": self.r_squared_total,
                "r_squared_per_group": self.r_squared_per_group,
                "rms_residual_per_group": self.rms_residual_per_group,
            },
            "standard_errors": {
                "method": (
                    "asymptotic, from the Jacobian at the optimum, with the "
                    "uncertainty of P(0) marginalized out as a nuisance direction"
                ),
                "caveat": (
                    "a local linearization and a LOWER BOUND. Because P(0) is read "
                    "from a noisy sample rather than fitted, the empirical spread "
                    "was measured at up to "
                    f"{ASYMPTOTIC_UNDERSTATEMENT_FACTOR:g}x these values. Prefer the "
                    "bootstrap interval when more than one input file is available."
                ),
            },
            "residual_dimension": {
                "n_time_points": self.n_points,
                "n_groups": len(self.groups),
                "population_conservation_detected": self.population_conserved,
                "independent_coordinates_per_time": self.independent_coordinates_per_time,
                "n_independent_residuals": self.n_independent_residuals,
                "conditioned_initial_row_excluded": True,
                "note": (
                    "the rates are estimated in the natural population "
                    "coordinates, but the residual variance is scaled by the "
                    "coordinates that were free to disagree with the model. When "
                    "the populations are complete and conserved, the residual is "
                    "orthogonal to the all-ones direction, so only G-1 of the G "
                    "coordinates per time are independent -- the same subspace "
                    "compare-schemes scores through Helmert contrasts, which are "
                    "orthonormal and therefore leave the sum of squares "
                    "unchanged. The first row is excluded because P(0) is "
                    "conditioned on, not fitted"
                ),
            },
            "identifiability": {
                "jacobian_condition_number": self.condition_number,
                "jacobian_is_singular": not np.isfinite(self.condition_number),
                "jacobian_rank_deficient": self.rank_deficient,
                "jacobian_rank": self.jacobian_rank,
                "jacobian_nullity": self.jacobian_nullity,
                "null_space_tolerance": self.null_space_tolerance,
                "null_space_participation": {
                    r.name: r.null_space_participation for r in self.rates
                },
                "null_space_identifiable": {
                    r.name: r.null_space_identifiable for r in self.rates
                },
                "null_space_participation_limit": NULL_PARTICIPATION_LIMIT,
                "null_space_note": (
                    "||V_null[j, :]|| from an SVD of the rate Jacobian: a rate can "
                    "join a blind combination even when its own column is far from "
                    "zero and it correlates strongly with no single other rate. "
                    "This supplements the blind-column, relative-standard-error "
                    "and correlation checks; it does not replace them"
                ),
                "rates_at_an_optimizer_bound": [
                    r.name for r in self.rates if r.at_optimizer_bound
                ],
                "initial_condition_uncertainty_propagated": True,
                "unidentified_threshold_relative_stderr": UNIDENTIFIED_REL_SE,
                "degenerate_threshold_correlation": DEGENERATE_CORRELATION,
                "correlation_matrix": (
                    self.correlation.tolist() if self.correlation is not None else None
                ),
                "n_unidentified": sum(1 for r in self.rates if not r.identified),
                "unidentified_reasons": {
                    r.name: r.unidentified_reason
                    for r in self.rates
                    if r.unidentified_reason
                },
            },
            "model_assumptions": [
                "a Markovian master equation with time-independent rates",
                "the declared groups form a closed system",
                "every trajectory in the average obeys the same rate matrix",
            ],
            "warnings": self.warnings,
        }


def _initial_condition_jacobian(
    log_rates: np.ndarray,
    time: np.ndarray,
    observed: np.ndarray,
    edges: Sequence[Tuple[int, int]],
    weights: np.ndarray,
) -> np.ndarray:
    """d(residual)/d(P(0)) along directions that conserve total population.

    P(0) is read from the first sample, which is as noisy as any other point.
    These columns let that noise be marginalized out of the rate covariance.
    The directions are ``e_j - e_0``, which keep the populations summing to the
    same total, so they explore only the perturbations the model allows.
    """
    n_groups = observed.shape[1]
    if n_groups < 2:
        return np.zeros((observed.size, 0))

    K = build_rate_matrix(np.exp(log_rates), edges, n_groups)
    base = propagate(K, observed[0], time)
    scale = max(float(np.max(np.abs(observed[0]))), 1.0) * 1e-6

    columns = []
    for j in range(1, n_groups):
        direction = np.zeros(n_groups)
        direction[j] = 1.0
        direction[0] = -1.0
        forward = propagate(K, observed[0] + scale * direction, time)
        columns.append((((forward - base) / scale) * weights).ravel())
    return np.column_stack(columns)


def _residuals(
    log_rates: np.ndarray,
    time: np.ndarray,
    observed: np.ndarray,
    edges: Sequence[Tuple[int, int]],
    weights: np.ndarray,
) -> np.ndarray:
    K = build_rate_matrix(np.exp(log_rates), edges, observed.shape[1])
    model = propagate(K, observed[0], time)
    return ((model - observed) * weights).ravel()


def fit_master_equation(
    time_ns: np.ndarray,
    observed: np.ndarray,
    groups: Sequence[str],
    edges: Sequence[Tuple[int, int]],
    weights: Optional[np.ndarray] = None,
    n_starts: int = 5,
    max_nfev: int = 4000,
    bound_decades: float = RATE_BOUND_DECADES,
) -> KineticFit:
    """Fit K by least squares in log-rate space, with multistart.

    Rates are fitted as ``log k`` so they stay positive and so the standard
    error in log space is directly the relative standard error on the rate.

    ``bound_decades`` sets how far either side of ``1/window`` a rate may run.
    A rate that stops *on* that edge is reported at the bound and refused: a
    boundary solution is not an interior optimum, and the local quadratic
    picture behind the covariance does not describe it.  Narrow the range only
    to express a real physical prior, never to make a rate look determined.
    """
    time_ns = np.asarray(time_ns, dtype=float)
    observed = np.asarray(observed, dtype=float)
    groups = list(groups)
    edges = [tuple(e) for e in edges]

    if observed.ndim != 2 or observed.shape[1] != len(groups):
        raise KineticsError(
            f"observed has shape {observed.shape}, expected (ntime, {len(groups)})"
        )
    if observed.shape[0] != time_ns.size:
        raise KineticsError("time and observed disagree on the number of samples")
    if len(edges) == 0:
        raise KineticsError("no transitions to fit")
    if observed.shape[0] <= len(edges):
        raise KineticsError(
            f"{observed.shape[0]} time points cannot constrain {len(edges)} rates"
        )

    if weights is None:
        weight_array = np.ones_like(observed)
    else:
        weight_array = np.asarray(weights, dtype=float)
        if weight_array.shape != observed.shape:
            raise KineticsError("weights must have the same shape as observed")

    span = float(time_ns[-1] - time_ns[0])
    if span <= 0:
        raise KineticsError("the time window has no extent")
    base = 1.0 / span

    best = None
    for factor in np.geomspace(0.05, 20.0, n_starts):
        start = np.full(len(edges), np.log(base * factor))
        try:
            candidate = least_squares(
                _residuals,
                start,
                args=(time_ns, observed, edges, weight_array),
                method="trf",
                bounds=(
                    np.log(base) - bound_decades * np.log(10.0),
                    np.log(base) + bound_decades * np.log(10.0),
                ),
                max_nfev=max_nfev,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        if best is None or candidate.cost < best.cost:
            best = candidate
    if best is None:
        raise KineticsError("every optimizer start failed")

    rates = np.exp(best.x)
    K = build_rate_matrix(rates, edges, len(groups))
    model = propagate(K, observed[0], time_ns)

    residual = model - observed
    n_points = observed.shape[0]
    n_parameters = len(edges)
    n_independent, per_time, conserved = independent_residual_count(observed)

    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((observed - observed.mean(axis=0)) ** 2))
    r2_total = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    r2_group: Dict[str, float] = {}
    rms_group: Dict[str, float] = {}
    for index, name in enumerate(groups):
        column = observed[:, index]
        denominator = float(np.sum((column - column.mean()) ** 2))
        numerator = float(np.sum(residual[:, index] ** 2))
        r2_group[name] = (
            1.0 - numerator / denominator if denominator > 0 else float("nan")
        )
        rms_group[name] = float(np.sqrt(numerator / n_points))

    # Asymptotic covariance in log space.  Two things have to be right here or
    # the identifiability flag becomes worse than useless.
    #
    # First, the Jacobian is augmented with the directions in which P(0) can
    # move.  P(0) is read off the first sample rather than fitted, but that
    # sample carries the same noise as every other point, and its error
    # propagates into every rate.  Treating it as exact understates the
    # standard errors by a large factor.  The nuisance directions are taken
    # inside the simplex (they conserve total population) and are marginalized
    # out, so the point estimate is unchanged and only the uncertainty grows.
    #
    # Second, a pseudo-inverse reports *zero* variance for a direction the data
    # does not constrain at all, which would mark exactly the worst rates as
    # identified.  Blind directions are detected from the Jacobian and given
    # infinite variance instead.
    jacobian = best.jac
    stderr_log: Optional[np.ndarray] = None
    correlation: Optional[np.ndarray] = None
    condition = float("inf")
    rank_deficient = False
    n_rates = len(edges)
    blind = np.zeros(n_rates, dtype=bool)
    participation = np.zeros(n_rates, dtype=float)
    jacobian_rank = n_rates
    jacobian_nullity = 0
    null_tolerance = 0.0

    # least_squares reports which parameters stopped on the edge of the allowed
    # log-rate range.  A boundary solution is not an interior optimum, so the
    # local quadratic picture behind the covariance does not apply to it.
    active = np.asarray(getattr(best, "active_mask", np.zeros(n_rates)), dtype=int)
    if active.shape != (n_rates,):
        active = np.zeros(n_rates, dtype=int)

    try:
        column_norm = np.linalg.norm(jacobian, axis=0)
        scale = float(column_norm.max()) if column_norm.size else 0.0
        blind = column_norm <= max(scale, 1.0) * 1e-10

        null_space = null_space_analysis(jacobian)
        participation = np.asarray(null_space["participation"], dtype=float)
        jacobian_rank = int(null_space["rank"])
        jacobian_nullity = int(null_space["nullity"])
        null_tolerance = float(null_space["tolerance"])

        nuisance = _initial_condition_jacobian(
            best.x, time_ns, observed, edges, weight_array
        )
        augmented = np.hstack([jacobian, nuisance]) if nuisance.size else jacobian
        n_nuisance = augmented.shape[1] - n_rates
        # Scale the residual variance by the coordinates that were free to
        # disagree with the model, not by every number in the table.  With
        # complete conserved populations the redundant conservation direction
        # carries no information, and the conditioned P(0) row carries none
        # either; counting them inflates the divisor and understates the
        # standard errors, which are already a lower bound.
        dof_augmented = max(n_independent - n_rates - n_nuisance, 1)

        gram = jacobian.T @ jacobian
        condition = float(np.linalg.cond(gram))
        singular = np.linalg.svd(gram, compute_uv=False)
        if singular.size:
            tol = float(singular.max()) * max(gram.shape) * np.finfo(float).eps
            rank_deficient = bool(singular.min() <= tol)

        gram_augmented = augmented.T @ augmented
        covariance_augmented = np.linalg.pinv(gram_augmented) * (
            2.0 * best.cost / dof_augmented
        )
        covariance = np.array(covariance_augmented[:n_rates, :n_rates], dtype=float)
        raw_variance = np.diag(covariance).astype(float).copy()
        # The correlation is taken from the covariance as computed, before any
        # variance is overridden below. Overriding first would send the whole
        # row to NaN and silently disable the pairwise degeneracy check on
        # exactly the rates that need every check applied to them.
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_sigma = np.sqrt(np.where(raw_variance > 0, raw_variance, np.nan))
            outer = np.outer(raw_sigma, raw_sigma)
            correlation = np.where(
                np.isfinite(outer) & (outer > 0), covariance / outer, np.nan
            )

        # A direction the pseudo-inverse discarded comes back with ZERO
        # variance, not infinite. Blind columns and null-space participants are
        # both such directions, so both get infinite variance here rather than a
        # "0 +- 0" that reads like a precise measurement.
        variance = raw_variance.copy()
        variance[blind | (participation > NULL_PARTICIPATION_LIMIT)] = np.inf
        variance[variance < 0] = np.inf
        stderr_log = np.sqrt(variance)
    except (np.linalg.LinAlgError, ValueError):
        stderr_log = None
        correlation = None

    warnings: List[str] = []
    estimates: List[RateEstimate] = []
    names = edge_names(edges, groups)
    for index, (source, target) in enumerate(edges):
        rel_se = float(stderr_log[index]) if stderr_log is not None else None
        reason: Optional[str] = None
        share = float(participation[index])
        null_ok = share <= NULL_PARTICIPATION_LIMIT
        bound = (
            "lower" if active[index] < 0 else "upper" if active[index] > 0 else None
        )
        if rel_se is None:
            identified = False
            reason = "the covariance of the fit could not be computed"
        elif bool(blind[index]):
            identified = False
            reason = (
                "the residuals do not change when this rate changes, so the data "
                "carries no information about it; the reported value is where the "
                "optimizer happened to stop"
            )
        elif bound is not None:
            identified = False
            reason = (
                f"the optimum lies on the allowed parameter boundary ({bound} "
                "bound), so the data do not determine an interior estimate for "
                "this rate and a local uncertainty around it does not apply"
            )
        elif not null_ok:
            identified = False
            reason = (
                "this rate participates in a null-space direction of the "
                f"Jacobian (participation {share:.3g} exceeds "
                f"{NULL_PARTICIPATION_LIMIT}); changes in a combination "
                "containing this rate leave the fitted populations unchanged"
            )
        elif rel_se > UNIDENTIFIED_REL_SE:
            identified = False
            reason = (
                f"relative standard error {rel_se:.3g} exceeds {UNIDENTIFIED_REL_SE}, "
                "and that error is itself a lower bound"
            )
        else:
            identified = True

        # The checks above are independent, and each can fire on its own. Keep
        # every one of them applied even when an earlier one already answered.
        if bound is not None or not null_ok:
            identified = False

        partners: List[str] = []
        if correlation is not None and not bool(blind[index]) and null_ok:
            for other in range(len(edges)):
                if other == index:
                    continue
                # Correlations with a structurally unidentified direction are
                # artifacts of the pseudo-inverse, not evidence that the
                # identifiable rate is degenerate with that parameter. In
                # particular, a discarded blind/null-space direction can carry
                # an arbitrary covariance orientation that changes across
                # LAPACK/NumPy versions. Such a parameter is already refused
                # by the stronger blind/null-space checks above and must not
                # contaminate otherwise identifiable rates.
                if bool(blind[other]) or participation[other] > NULL_PARTICIPATION_LIMIT:
                    continue
                value = correlation[index, other]
                if np.isfinite(value) and abs(value) >= DEGENERATE_CORRELATION:
                    partners.append(names[other])
        if partners:
            identified = False
            if reason is None:
                reason = (
                    "correlated above "
                    f"{DEGENERATE_CORRELATION} with {', '.join(partners)}; the data "
                    "sees the combination, not this rate on its own"
                )
        estimates.append(
            RateEstimate(
                name=names[index],
                source=groups[source],
                target=groups[target],
                rate_per_ns=float(rates[index]),
                stderr_per_ns=(
                    float(rates[index] * rel_se)
                    if rel_se is not None and np.isfinite(rel_se)
                    else None
                ),
                relative_stderr=(
                    rel_se if rel_se is not None and np.isfinite(rel_se) else None
                ),
                lifetime_ns=float(1.0 / rates[index]) if rates[index] > 0 else float("inf"),
                identified=identified,
                degenerate_with=partners,
                unidentified_reason=reason,
                null_space_participation=share,
                null_space_identifiable=null_ok,
                at_optimizer_bound=bound is not None,
                optimizer_bound=bound,
            )
        )

    unidentified = [e.name for e in estimates if not e.identified]
    if unidentified:
        warnings.append(
            "the population curves do not determine these rates individually: "
            + ", ".join(unidentified)
            + "; quote the eigenvalue timescales instead"
        )
    if not np.isfinite(condition) or condition > 1e8:
        detail = (
            "singular" if not np.isfinite(condition) else f"condition number {condition:.3g}"
        )
        warnings.append(
            f"the Jacobian is ill-conditioned ({detail}); the rate matrix is close "
            "to a degenerate family that fits equally well"
        )
    if rank_deficient:
        warnings.append(
            "the Jacobian is rank deficient: at least one direction in rate space "
            "leaves the populations unchanged, so that combination of rates is not "
            "determined by this data at any precision"
        )
    if jacobian_nullity:
        involved = [
            names[index]
            for index in range(n_rates)
            if participation[index] > NULL_PARTICIPATION_LIMIT
        ]
        warnings.append(
            f"the rate Jacobian has rank {jacobian_rank} with nullity "
            f"{jacobian_nullity}: {jacobian_nullity} combination(s) of rates leave "
            "the fitted populations unchanged"
            + (f", involving {', '.join(involved)}" if involved else "")
        )
    bounded = [e.name for e in estimates if e.at_optimizer_bound]
    if bounded:
        warnings.append(
            "these rates stopped on the edge of the allowed range, which is not an "
            "interior optimum and carries no ordinary local uncertainty: "
            + ", ".join(bounded)
        )
    if np.isfinite(r2_total) and r2_total < 0.95:
        warnings.append(
            f"R^2 = {r2_total:.4f} over all groups; this scheme does not describe "
            "the data, so its rates are not meaningful"
        )

    eigenvalues = np.linalg.eigvals(K)
    timescales = []
    for value in sorted(eigenvalues, key=lambda v: v.real):
        real = float(value.real)
        if real < -1e-12:
            timescales.append(-1.0 / real)
    fastest = 1.0 / float(np.max(np.abs(np.diag(K)))) if np.any(np.diag(K)) else None
    if fastest is not None and fastest < 2.0 * float(time_ns[1] - time_ns[0]):
        warnings.append(
            "the fastest fitted process is comparable to the sample spacing and "
            "is not resolved by this time grid"
        )
    if timescales and timescales[-1] > 10.0 * span:
        warnings.append(
            f"the slowest fitted timescale ({timescales[-1]:.4g} ns) is more than "
            f"ten times the fitted window ({span:.4g} ns) and is an extrapolation"
        )

    return KineticFit(
        groups=groups,
        edges=edges,
        rates=estimates,
        K=K,
        time_ns=time_ns,
        observed=observed,
        model=model,
        p0=observed[0].copy(),
        window=(float(time_ns[0]), float(time_ns[-1])),
        cost=float(best.cost),
        r_squared_total=float(r2_total),
        r_squared_per_group=r2_group,
        rms_residual_per_group=rms_group,
        eigen_timescales_ns=[float(t) for t in timescales],
        condition_number=condition,
        rank_deficient=rank_deficient,
        correlation=correlation,
        n_points=int(n_points),
        n_parameters=int(n_parameters),
        jacobian_rank=int(jacobian_rank),
        jacobian_nullity=int(jacobian_nullity),
        null_space_tolerance=float(null_tolerance),
        n_independent_residuals=int(n_independent),
        independent_coordinates_per_time=int(per_time),
        population_conserved=bool(conserved),
        warnings=warnings,
    )


def bootstrap_rates(
    time_ns: np.ndarray,
    per_file_groups: np.ndarray,
    groups: Sequence[str],
    edges: Sequence[Tuple[int, int]],
    n_resamples: int = 200,
    seed: int = 0,
    percentiles: Tuple[float, float] = (2.5, 97.5),
    weights: Optional[np.ndarray] = None,
    min_successes: int = 20,
    min_identified_fraction: float = BOOTSTRAP_MIN_IDENTIFIED_FRACTION,
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    """Resample whole input files with replacement and refit.

    This propagates the spread *between the files supplied*, which share a
    trajectory and often correlated initial conditions. It is not an ensemble
    error bar, and it says nothing about whether the Markovian model is right.

    ``weights`` must be the same weights used for the point estimate, otherwise
    the interval is centred on a different estimator than the rate it is
    printed beside and can exclude it.

    An interval is reported for a rate only when the bootstrap produced enough
    successful fits *and* that rate was identified in at least
    ``min_identified_fraction`` of them; it is then taken over the identified
    resamples alone.  An optimizer that converges is not the same thing as data
    that determines a rate: keeping the value from every converged draw
    produces a tight-looking percentile band around a number the data never
    fixed.  Suppressed intervals are reported as suppressed, with the counts,
    rather than silently omitted.

    Returns ``(intervals, diagnostics)``; the diagnostics record how many
    resamples converged and, per rate, how many of those identified it, so an
    interval built from a subset is never reported as if every draw had
    succeeded.
    """
    per_file_groups = np.asarray(per_file_groups, dtype=float)
    if per_file_groups.ndim != 3:
        raise KineticsError("per_file_groups must have shape (nfiles, ntime, ngroups)")
    n_files = per_file_groups.shape[0]
    if n_files < 2:
        raise KineticsError("bootstrapping needs at least two input files")

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != per_file_groups.shape[1:]:
            raise KineticsError(
                f"weights have shape {weights.shape}, expected "
                f"{per_file_groups.shape[1:]} to match the resampled data"
            )

    rng = np.random.default_rng(seed)
    names = edge_names(edges, groups)
    # Every converged draw, and separately the draws in which each rate was
    # actually identified. A percentile taken over draws where the optimizer's
    # stopping point set the value is narrow for the wrong reason.
    draws: Dict[str, List[float]] = {name: [] for name in names}
    identified_draws: Dict[str, List[float]] = {name: [] for name in names}
    successes = 0
    failures = 0
    for _ in range(n_resamples):
        picks = rng.integers(0, n_files, size=n_files)
        sample = per_file_groups[picks].mean(axis=0)
        try:
            fit = fit_master_equation(
                time_ns, sample, groups, edges, weights=weights, n_starts=2
            )
        except (KineticsError, ValueError, np.linalg.LinAlgError):
            failures += 1
            continue
        successes += 1
        for estimate in fit.rates:
            draws[estimate.name].append(estimate.rate_per_ns)
            if estimate.identified:
                identified_draws[estimate.name].append(estimate.rate_per_ns)

    intervals: Dict[str, List[float]] = {}
    per_rate: Dict[str, Dict[str, Any]] = {}
    enough_globally = successes >= min_successes
    for name in names:
        n_identified = len(identified_draws[name])
        fraction = n_identified / successes if successes else 0.0
        record: Dict[str, Any] = {
            "bootstrap_successes": successes,
            "bootstrap_identified": n_identified,
            "bootstrap_identified_fraction": fraction,
            "raw_distribution_n": len(draws[name]),
        }
        if draws[name]:
            # Kept for diagnostics only; this is never the headline interval.
            low, high = np.percentile(draws[name], percentiles)
            record["raw_optimizer_percentiles_per_ns"] = [float(low), float(high)]
        if not enough_globally:
            record["bootstrap_ci_status"] = "insufficient_successful_resamples"
            record["note"] = (
                f"only {successes} resamples converged, fewer than the "
                f"{min_successes} required, so no interval is reported"
            )
        elif n_identified == 0:
            record["bootstrap_ci_status"] = "suppressed_never_identified"
            record["note"] = (
                f"this rate was identified in none of the {successes} successful "
                "resamples, so there is no draw an interval could be taken over"
            )
        elif fraction < min_identified_fraction:
            record["bootstrap_ci_status"] = "suppressed_low_identified_fraction"
            record["note"] = (
                f"bootstrap interval suppressed because the transition was "
                f"identifiable in only {fraction:.0%} of the {successes} "
                f"successful resamples, below the required "
                f"{min_identified_fraction:.0%}. The {successes - n_identified} "
                "unidentified draws are counted here rather than quietly dropped "
                "into a narrow-looking percentile"
            )
        else:
            low, high = np.percentile(identified_draws[name], percentiles)
            intervals[name] = [float(low), float(high)]
            record["bootstrap_ci_status"] = "reported"
            record["interval_per_ns"] = intervals[name]
            record["note"] = (
                f"percentiles over the {n_identified} resamples in which this rate "
                "was identified"
            )
        per_rate[name] = record

    diagnostics: Dict[str, Any] = {
        "requested_resamples": int(n_resamples),
        "converged_resamples": int(successes),
        "failed_resamples": int(failures),
        "minimum_for_an_interval": int(min_successes),
        "minimum_identified_fraction": float(min_identified_fraction),
        "percentiles": list(percentiles),
        "weighted": weights is not None,
        "intervals_reported": len(intervals),
        "per_rate": per_rate,
        "policy": (
            "an interval is reported only when the bootstrap as a whole produced "
            "enough successful fits AND the rate was identified in at least "
            f"{min_identified_fraction:.0%} of them; the interval is then taken "
            "over the identified resamples alone. The raw optimizer percentiles "
            "over every converged draw are kept for diagnostics and are not an "
            "inferential interval"
        ),
    }
    if failures:
        diagnostics["note"] = (
            f"{failures} of {n_resamples} resamples did not converge and were "
            "dropped; the interval is computed from the ones that did, which may "
            "not be a representative subset"
        )
    if not enough_globally:
        diagnostics["note"] = (
            f"only {successes} resamples converged, fewer than the {min_successes} "
            "required, so no interval is reported"
        )
    suppressed = [
        name
        for name, record in per_rate.items()
        if record["bootstrap_ci_status"]
        in ("suppressed_low_identified_fraction", "suppressed_never_identified")
    ]
    if suppressed:
        diagnostics["suppressed_for_low_identified_fraction"] = suppressed
    return intervals, diagnostics


# ---------------------------------------------------------------------------
# Extraction sink


@dataclass
class SinkPoint:
    k_escape_per_ns: float
    escape_time_ns: float
    collected_final: float
    recombined_final: float
    remaining_final: float
    populations_final: Dict[str, float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "k_escape_per_ns": self.k_escape_per_ns,
            "escape_time_ns": self.escape_time_ns,
            "collected_final": self.collected_final,
            "recombined_final": self.recombined_final,
            "remaining_final": self.remaining_final,
            "populations_final": self.populations_final,
        }


def sink_sweep(
    fit: KineticFit,
    sink_group: str,
    escape_rates_per_ns: Sequence[float],
    recombined_group: Optional[str] = None,
    time_ns: Optional[np.ndarray] = None,
) -> List[SinkPoint]:
    """Propagate the fitted system with an absorbing extraction channel added.

    The sink is a genuine extra channel in the propagated system: population
    that escapes from ``sink_group`` leaves the dynamics and cannot return or
    recombine. It is **not** an integral taken over the sink-free solution.

    The transfer rates come from a fit to data with no extraction in it, so
    this answers a counterfactual: given those rates, how fast would onward
    transport have to be to outcompete the return and recombination channels?
    ``k_escape`` is physics supplied by the user; nothing in the interface
    calculation determines it.
    """
    if sink_group not in fit.groups:
        raise KineticsError(
            f"sink group {sink_group!r} is not one of {fit.groups}"
        )
    if recombined_group is not None and recombined_group not in fit.groups:
        raise KineticsError(
            f"recombined group {recombined_group!r} is not one of {fit.groups}"
        )

    grid = fit.time_ns if time_ns is None else np.asarray(time_ns, dtype=float)
    n = len(fit.groups)
    sink_index = fit.groups.index(sink_group)
    recombined_index = (
        fit.groups.index(recombined_group) if recombined_group is not None else None
    )

    points: List[SinkPoint] = []
    for rate in escape_rates_per_ns:
        if rate < 0:
            raise KineticsError(f"escape rate {rate} is negative")
        augmented = np.zeros((n + 1, n + 1), dtype=float)
        augmented[:n, :n] = fit.K
        augmented[n, sink_index] = rate
        augmented[sink_index, sink_index] -= rate
        p0 = np.zeros(n + 1)
        p0[:n] = fit.p0
        trajectory = propagate(augmented, p0, grid)
        final = trajectory[-1]
        points.append(
            SinkPoint(
                k_escape_per_ns=float(rate),
                escape_time_ns=float(1.0 / rate) if rate > 0 else float("inf"),
                collected_final=float(final[n]),
                recombined_final=(
                    float(final[recombined_index])
                    if recombined_index is not None
                    else float("nan")
                ),
                remaining_final=float(final[:n].sum()),
                populations_final={
                    name: float(value) for name, value in zip(fit.groups, final[:n])
                },
            )
        )
    return points


SINK_HEADER = [
    "k_escape_per_ns",
    "escape_time_ns",
    "collected_final",
    "recombined_final",
    "remaining_final",
]

RATE_HEADER = [
    "transition",
    "rate_per_ns",
    "stderr_per_ns",
    "relative_stderr",
    "lifetime_ns",
    "identified",
    "unidentified_reason",
    "degenerate_with",
    "null_space_participation",
    "null_space_identifiable",
    "at_optimizer_bound",
    "optimizer_bound",
    "bootstrap_ci_status",
    "bootstrap_identified_fraction",
    "bootstrap_low_per_ns",
    "bootstrap_high_per_ns",
]


def rate_rows(fit: KineticFit) -> List[List[Any]]:
    rows = []
    for estimate in fit.rates:
        ci = estimate.bootstrap_ci_per_ns or [None, None]
        rows.append(
            [
                estimate.name,
                estimate.rate_per_ns,
                estimate.stderr_per_ns,
                estimate.relative_stderr,
                estimate.lifetime_ns,
                estimate.identified,
                estimate.unidentified_reason or "",
                ";".join(estimate.degenerate_with),
                estimate.null_space_participation,
                estimate.null_space_identifiable,
                estimate.at_optimizer_bound,
                estimate.optimizer_bound or "",
                estimate.bootstrap_ci_status or "",
                estimate.bootstrap_identified_fraction,
                ci[0],
                ci[1],
            ]
        )
    return rows


def sink_rows(points: Sequence[SinkPoint]) -> List[List[Any]]:
    return [
        [
            p.k_escape_per_ns,
            p.escape_time_ns,
            p.collected_final,
            p.recombined_final,
            p.remaining_final,
        ]
        for p in points
    ]
