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


class KineticsError(ValueError):
    """Raised when a kinetic model cannot be built or fitted as requested."""


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
            "bootstrap_ci_per_ns": self.bootstrap_ci_per_ns,
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
            "identifiability": {
                "jacobian_condition_number": self.condition_number,
                "jacobian_is_singular": not np.isfinite(self.condition_number),
                "jacobian_rank_deficient": self.rank_deficient,
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
) -> KineticFit:
    """Fit K by least squares in log-rate space, with multistart.

    Rates are fitted as ``log k`` so they stay positive and so the standard
    error in log space is directly the relative standard error on the rate.
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
                bounds=(np.log(1e-9 * base), np.log(1e9 * base)),
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
    dof = max(n_points * len(groups) - n_parameters, 1)

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

    try:
        column_norm = np.linalg.norm(jacobian, axis=0)
        scale = float(column_norm.max()) if column_norm.size else 0.0
        blind = column_norm <= max(scale, 1.0) * 1e-10

        nuisance = _initial_condition_jacobian(
            best.x, time_ns, observed, edges, weight_array
        )
        augmented = np.hstack([jacobian, nuisance]) if nuisance.size else jacobian
        n_nuisance = augmented.shape[1] - n_rates
        dof_augmented = max(n_points * len(groups) - n_rates - n_nuisance, 1)

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
        variance = np.diag(covariance).astype(float).copy()
        variance[blind] = np.inf
        variance[variance < 0] = np.inf
        stderr_log = np.sqrt(variance)
        with np.errstate(divide="ignore", invalid="ignore"):
            outer = np.outer(stderr_log, stderr_log)
            correlation = np.where(
                np.isfinite(outer) & (outer > 0), covariance / outer, np.nan
            )
    except (np.linalg.LinAlgError, ValueError):
        stderr_log = None
        correlation = None

    warnings: List[str] = []
    estimates: List[RateEstimate] = []
    names = edge_names(edges, groups)
    for index, (source, target) in enumerate(edges):
        rel_se = float(stderr_log[index]) if stderr_log is not None else None
        reason: Optional[str] = None
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
        elif rel_se > UNIDENTIFIED_REL_SE:
            identified = False
            reason = (
                f"relative standard error {rel_se:.3g} exceeds {UNIDENTIFIED_REL_SE}, "
                "and that error is itself a lower bound"
            )
        else:
            identified = True

        partners: List[str] = []
        if correlation is not None:
            for other in range(len(edges)):
                if other == index:
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
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    """Resample whole input files with replacement and refit.

    This propagates the spread *between the files supplied*, which share a
    trajectory and often correlated initial conditions. It is not an ensemble
    error bar, and it says nothing about whether the Markovian model is right.

    ``weights`` must be the same weights used for the point estimate, otherwise
    the interval is centred on a different estimator than the rate it is
    printed beside and can exclude it.

    Returns ``(intervals, diagnostics)``; the diagnostics record how many
    resamples actually converged, so an interval built from a subset is never
    reported as if every draw had succeeded.
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
    draws: Dict[str, List[float]] = {name: [] for name in names}
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

    intervals: Dict[str, List[float]] = {}
    for name, values in draws.items():
        if len(values) >= min_successes:
            low, high = np.percentile(values, percentiles)
            intervals[name] = [float(low), float(high)]

    diagnostics: Dict[str, Any] = {
        "requested_resamples": int(n_resamples),
        "converged_resamples": int(successes),
        "failed_resamples": int(failures),
        "minimum_for_an_interval": int(min_successes),
        "percentiles": list(percentiles),
        "weighted": weights is not None,
        "intervals_reported": len(intervals),
    }
    if failures:
        diagnostics["note"] = (
            f"{failures} of {n_resamples} resamples did not converge and were "
            "dropped; the interval is computed from the ones that did, which may "
            "not be a representative subset"
        )
    if not intervals:
        diagnostics["note"] = (
            f"only {successes} resamples converged, fewer than the {min_successes} "
            "required, so no interval is reported"
        )
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
