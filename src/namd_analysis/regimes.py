"""Early and late dynamical regimes, analysed separately rather than merged.

A fit that begins after a fast transient cannot see it.  If an acceptor
population rises and relaxes inside the first hundred picoseconds, a late-window
fit reports the relaxed value and says nothing about the transfer that produced
it -- not because the transfer did not happen, but because the window did not
look.  This module analyses declared windows side by side and asks whether one
kinetic description is adequate for both, rather than assuming it is.

Three things are kept strictly apart, in the vocabulary used everywhere else in
this package:

``observed_from_SHPROP``
    populations, peaks, integrals -- read from the histories.
``model_inferred``
    rates, timescales, information criteria -- conditional on a declared
    kinetic graph and on the Markovian constant-rate assumption.
``interpretive``
    sentences a reader might build from the above.  Labelled as such, never
    asserted as measurement.

**No window is a universal constant.**  The early window defaults to 0-100 ps
because that is the interval under examination, not because anything in the
data selects it.  The late window has no default at all: the manuscript's fit
window is not encoded anywhere in this repository -- the shipped comparison
template says so in as many words -- so it must be supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .kinetics import (
    KineticFit,
    KineticsError,
    bootstrap_rates,
    fit_master_equation,
)
from .observables import MODEL_INFERRED, OBSERVED
from .transient import Window, analyze

#: The interval this analysis was built to characterize.  A default because it
#: was asked for by name, not because the data select it; it is reported as an
#: explicit choice in every output.
DEFAULT_EARLY_WINDOW_NS = (0.0, 0.1)

#: Why the late window has no default.
LATE_WINDOW_NOTE = (
    "The late window is not defaulted. No fit window from the manuscript is "
    "encoded anywhere in this repository -- examples/bcf_pcbm/"
    "comparison_manifest.template.json states that the windows 'are not "
    "universal constants and are not defaulted anywhere in the code' -- so "
    "inventing one here would put a number into the science that nobody chose. "
    "Supply it explicitly."
)

#: What the early default means.
EARLY_WINDOW_NOTE = (
    "The early window defaults to 0-100 ps because that is the interval being "
    "characterized, not because any feature of the data selects it. Where one "
    "regime ends and the next begins is a property of the system; vary it and "
    "see whether the conclusions move."
)


class RegimeError(ValueError):
    """Raised when a regime analysis would have to assume something."""


# --------------------------------------------------------------------------
# One regime
# --------------------------------------------------------------------------


@dataclass
class RegimeFit:
    """Everything known about one window: observables, then a model."""

    name: str
    window: Window
    start_ns: float
    end_ns: float
    n_points: int
    metrics: List[Any]
    fit: Optional[KineticFit]
    fit_error: Optional[str]
    bootstrap: Optional[Dict[str, Any]]
    identifiability: Dict[str, Any] = field(default_factory=dict)

    @property
    def identified(self) -> bool:
        return bool(self.fit is not None and self.fit.identified_rates)

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "regime": self.name,
            "window_ns": [self.start_ns, self.end_ns],
            "n_time_points": self.n_points,
            "observables": {
                "observable_class": OBSERVED,
                "per_group": [
                    {
                        "group": m.group,
                        "initial_population": m.initial_population,
                        "final_population": m.final_population,
                        "net_change": m.net_change,
                        "peak_population": m.peak_population,
                        "peak_time_ns": m.peak_time_ns,
                        "minimum_population": m.minimum_population,
                        "minimum_time_ns": getattr(m, "minimum_time_ns", None),
                        "integrated_population_ns": m.integrated_population_ns,
                    }
                    for m in self.metrics
                ],
            },
        }
        if self.fit is not None:
            payload["kinetics"] = {"observable_class": MODEL_INFERRED, **self.fit.as_dict()}
        else:
            payload["kinetics"] = {
                "observable_class": MODEL_INFERRED,
                "status": "not_fitted",
                "reason": self.fit_error,
                "note": (
                    "no rates are reported for this window. A window that cannot "
                    "constrain a rate is reported as such rather than given one"
                ),
            }
        payload["identifiability"] = self.identifiability
        if self.bootstrap is not None:
            payload["bootstrap"] = self.bootstrap
        return payload


def _identifiability(fit: Optional[KineticFit], reason: Optional[str]) -> Dict[str, Any]:
    if fit is None:
        return {
            "status": "not_fitted",
            "detail": reason,
            "identified_rates": 0,
            "total_rates": 0,
        }
    identified = fit.identified_rates
    total = fit.rates
    if not identified:
        status = "none_identified"
    elif len(identified) < len(total):
        status = "partially_identified"
    else:
        status = "all_identified"
    return {
        "status": status,
        "identified_rates": len(identified),
        "total_rates": len(total),
        "unidentified": [
            r.name for r in total if not r.identified
        ],
        "rank_deficient": fit.rank_deficient,
        "jacobian_nullity": fit.jacobian_nullity,
        "condition_number": fit.condition_number,
        "n_independent_residuals": fit.n_independent_residuals,
        "warnings": list(fit.warnings),
        "note": (
            "a rate that is not identified is not a small rate: the data do not "
            "constrain it. Eigen-timescales can be determined even when the "
            "individual rates are not"
        ),
    }


def fit_regime(
    name: str,
    window: Window,
    time_ns: np.ndarray,
    observed: np.ndarray,
    groups: Sequence[str],
    edges: Sequence[Tuple[int, int]],
    per_file: Optional[np.ndarray] = None,
    n_starts: int = 5,
    n_bootstrap: int = 0,
    seed: int = 0,
) -> RegimeFit:
    """Observables and, where the window supports one, a kinetic fit.

    A window that cannot constrain the declared graph returns
    ``fit=None`` with the reason recorded.  That is a result, not a failure to
    be worked around by widening the window or dropping edges.
    """
    time_ns = np.asarray(time_ns, dtype=float)
    observed = np.asarray(observed, dtype=float)
    mask = window.mask(time_ns)
    sub_time = time_ns[mask]
    sub_obs = observed[mask]
    low, high = window.resolve(time_ns)

    series = [(group, observed[mask, index]) for index, group in enumerate(groups)]
    metrics = analyze(sub_time, series, [window])

    fit: Optional[KineticFit] = None
    reason: Optional[str] = None
    try:
        fit = fit_master_equation(sub_time, sub_obs, groups, edges, n_starts=n_starts)
    except (KineticsError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        reason = str(exc)

    bootstrap: Optional[Dict[str, Any]] = None
    if fit is not None and n_bootstrap and per_file is not None and per_file.shape[0] >= 2:
        try:
            intervals, diagnostics = bootstrap_rates(
                sub_time,
                per_file[:, mask, :],
                groups,
                edges,
                n_resamples=n_bootstrap,
                seed=seed,
            )
            bootstrap = {
                "observable_class": MODEL_INFERRED,
                "n_resamples": n_bootstrap,
                "intervals": intervals,
                "diagnostics": diagnostics,
            }
        except (KineticsError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            bootstrap = {"status": "unavailable", "reason": str(exc)}

    return RegimeFit(
        name=name,
        window=window,
        start_ns=low,
        end_ns=high,
        n_points=int(np.count_nonzero(mask)),
        metrics=metrics,
        fit=fit,
        fit_error=reason,
        bootstrap=bootstrap,
        identifiability=_identifiability(fit, reason),
    )


# --------------------------------------------------------------------------
# One model everywhere, or one per regime
# --------------------------------------------------------------------------


def _helmert(n: int) -> np.ndarray:
    """Orthonormal contrasts spanning the G-1 conserved subspace.

    The same basis :mod:`comparison` scores on, so a regime score and a scheme
    score are the same kind of number.
    """
    from scipy.linalg import helmert

    return helmert(n)


def _information_criteria(
    rss: float, n: int, k: int
) -> Dict[str, Optional[float]]:
    """The same conditional-Gaussian scores compare_schemes uses."""
    variance = max(rss / n, np.finfo(float).eps ** 2)
    likelihood_term = n * np.log(variance)
    aic = float(likelihood_term + 2 * k)
    return {
        "aic": aic,
        "bic": float(likelihood_term + k * np.log(n)),
        "aicc": float(aic + 2 * k * (k + 1) / (n - k - 1)) if n > k + 1 else None,
        "rss_contrasts": float(rss),
        "n_observations": int(n),
        "k_parameters": int(k),
    }


def _score_fit(fit: KineticFit, groups: Sequence[str]) -> Dict[str, Any]:
    contrast = _helmert(len(groups))
    residual = (fit.model - fit.observed)[1:] @ contrast.T
    rss = float(np.sum(residual**2))
    n = (fit.n_points - 1) * (len(groups) - 1)
    k = len(fit.edges) + 1
    return _information_criteria(rss, n, k)


def compare_piecewise(
    time_ns: np.ndarray,
    observed: np.ndarray,
    groups: Sequence[str],
    edges: Sequence[Tuple[int, int]],
    early: RegimeFit,
    late: RegimeFit,
    criterion: str = "aicc",
    n_starts: int = 5,
) -> Dict[str, Any]:
    """One global model against early-plus-late, scored the same way.

    This is **model comparison, not evidence of a mechanistic transition**. A
    piecewise description winning says the single constant-rate matrix does not
    describe both windows equally well; it does not say what changed, and it
    does not establish that anything physical switched at the boundary. The
    boundary was chosen, not fitted.
    """
    if criterion not in {"aic", "aicc", "bic"}:
        raise RegimeError("criterion must be aic, aicc or bic")
    time_ns = np.asarray(time_ns, dtype=float)
    observed = np.asarray(observed, dtype=float)

    span = Window("global", float(min(early.start_ns, late.start_ns)),
                 float(max(early.end_ns, late.end_ns)))
    mask = span.mask(time_ns)
    global_fit: Optional[KineticFit] = None
    global_error: Optional[str] = None
    try:
        global_fit = fit_master_equation(
            time_ns[mask], observed[mask], groups, edges, n_starts=n_starts
        )
    except (KineticsError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        global_error = str(exc)

    result: Dict[str, Any] = {
        "observable_class": MODEL_INFERRED,
        "criterion": criterion,
        "windows": {
            "global_ns": [span.resolve(time_ns)[0], span.resolve(time_ns)[1]],
            "early_ns": [early.start_ns, early.end_ns],
            "late_ns": [late.start_ns, late.end_ns],
        },
        "what_this_is": (
            "model comparison, NOT proof of a mechanistic transition. A piecewise "
            "description scoring better says one constant-rate matrix does not "
            "describe both windows equally well. It does not say what changed, "
            "and the boundary between the windows was chosen rather than fitted"
        ),
    }

    if global_fit is None:
        result["status"] = "global_fit_failed"
        result["global"] = {"status": "not_fitted", "reason": global_error}
        result["conclusion"] = (
            "the single global model could not be fitted, so no comparison is "
            "possible. That is itself informative: the declared graph does not "
            "describe the whole interval"
        )
        return result

    global_scores = _score_fit(global_fit, groups)
    result["global"] = {
        "scores": global_scores,
        "identifiability": _identifiability(global_fit, None),
        "eigen_timescales_ns": global_fit.eigen_timescales_ns,
    }

    pieces = []
    total_rss = 0.0
    total_n = 0
    total_k = 0
    unfitted = []
    for regime in (early, late):
        if regime.fit is None:
            unfitted.append(regime.name)
            continue
        scores = _score_fit(regime.fit, groups)
        pieces.append({"regime": regime.name, "scores": scores})
        total_rss += scores["rss_contrasts"]
        total_n += scores["n_observations"]
        total_k += scores["k_parameters"]
    result["pieces"] = pieces

    if unfitted:
        result["status"] = "piecewise_incomplete"
        result["unfitted_regimes"] = unfitted
        result["conclusion"] = (
            f"the {', '.join(unfitted)} window could not be fitted, so the "
            "piecewise description is not scoreable against the global one. "
            "No rates are invented to complete it"
        )
        return result

    piecewise_scores = _information_criteria(total_rss, total_n, total_k)
    result["piecewise_combined"] = piecewise_scores
    a = global_scores.get(criterion)
    b = piecewise_scores.get(criterion)
    if a is None or b is None:
        result["status"] = "insufficient_observations"
        result["conclusion"] = (
            f"{criterion} is undefined for one of the models at this sample count"
        )
        return result

    result["status"] = "scored"
    result["delta"] = float(b - a)
    result["preferred"] = "piecewise" if b < a else "global"
    result["conclusion"] = (
        f"{criterion} prefers the {result['preferred']} description by "
        f"{abs(b - a):.4g}. Lower is better. This compares descriptions on the "
        "same observations; it does not establish a mechanism, and a preference "
        "for the piecewise form does not locate a transition -- the boundary was "
        "supplied, not inferred"
    )
    result["interpretation_limits"] = [
        "conditional-Gaussian scores on G-1 orthonormal contrasts, the same "
        "convention the scheme comparison uses",
        "time samples are treated as independent for the score only; "
        "autocorrelation invalidates a formal likelihood reading",
        "the piecewise model has strictly more parameters and fits a partition "
        "of the same data; the criterion penalizes the parameter count but not "
        "the freedom to have chosen the boundary",
        "a rank-deficient or boundary-stopped fit violates the regularity "
        "assumptions behind any information criterion",
    ]
    return result


# --------------------------------------------------------------------------
# The reviewer's questions
# --------------------------------------------------------------------------


def early_transient_narrative(
    early: RegimeFit,
    late: RegimeFit,
    acceptor: str = "PCBM",
    donor: str = "BCF",
    ground: str = "VBM",
) -> Dict[str, Any]:
    """Answer, from the observables, the questions a referee will ask.

    Every answer is derived from populations, so each is labelled
    ``observed_from_SHPROP``.  Where a question cannot be answered from the
    data present, it says so instead of guessing.
    """
    early_by_group = {m.group: m for m in early.metrics}
    late_by_group = {m.group: m for m in late.metrics}
    findings: List[Dict[str, Any]] = []

    def _record(question: str, answer: str, evidence: Dict[str, Any], kind: str = OBSERVED):
        findings.append(
            {"question": question, "answer": answer, "evidence": evidence, "observable_class": kind}
        )

    # 1. Does the acceptor rise early?
    if acceptor in early_by_group:
        m = early_by_group[acceptor]
        rose = m.peak_population > m.initial_population + 1e-9
        _record(
            f"Does {acceptor} population rise within the early window?",
            (
                f"yes: it reaches {m.peak_population:.4g} at {m.peak_time_ns:.4g} ns "
                f"from {m.initial_population:.4g} at the window start"
                if rose
                else f"no: it never exceeds its initial {m.initial_population:.4g}"
            ),
            {
                "initial": m.initial_population,
                "peak": m.peak_population,
                "peak_time_ns": m.peak_time_ns,
                "final": m.final_population,
            },
        )

        # 3. Overshoot that is gone before the late window starts?
        if acceptor in late_by_group:
            late_m = late_by_group[acceptor]
            overshoot = m.peak_population - late_m.initial_population
            transient_peak = rose and m.peak_population > late_m.peak_population + 1e-9
            _record(
                f"Is there a transient {acceptor} maximum that disappears before "
                "the late window begins?",
                (
                    f"yes: the early peak {m.peak_population:.4g} exceeds anything in "
                    f"the late window (max {late_m.peak_population:.4g}), and the late "
                    f"window opens at {late_m.initial_population:.4g} -- an overshoot of "
                    f"{overshoot:.4g} has already relaxed"
                    if transient_peak
                    else "no: the early peak does not exceed the late-window maximum"
                ),
                {
                    "early_peak": m.peak_population,
                    "late_window_initial": late_m.initial_population,
                    "late_peak": late_m.peak_population,
                    "overshoot_relaxed_before_late_window": float(overshoot),
                },
            )
            _record(
                "Does the late fit miss early transfer because the acceptor has "
                "already relaxed?",
                (
                    f"the late window opens at {late_m.initial_population:.4g}, below the "
                    f"early peak of {m.peak_population:.4g}. A fit starting at "
                    f"{late.start_ns:g} ns cannot see that occupation, and must not be "
                    "described as showing the group does not change"
                    if transient_peak
                    else "no evidence of an early peak that the late window misses"
                ),
                {
                    "late_window_start_ns": late.start_ns,
                    "early_peak": m.peak_population,
                    "late_window_initial": late_m.initial_population,
                },
                kind="interpretive" if transient_peak else OBSERVED,
            )

    # 2. Where does the donor go?
    if donor in early_by_group:
        d = early_by_group[donor]
        gains = {
            name: m.net_change
            for name, m in early_by_group.items()
            if name != donor and m.net_change > 0
        }
        total_gain = sum(gains.values()) or float("nan")
        shares = {name: value / total_gain for name, value in gains.items()} if gains else {}
        if d.net_change < 0 and gains:
            ranked = sorted(shares.items(), key=lambda kv: -kv[1])
            leader = ranked[0]
            answer = (
                f"{donor} loses {abs(d.net_change):.4g} over the early window. The "
                f"groups that gain are "
                + ", ".join(f"{n} {shares[n]*100:.0f}%" for n, _ in ranked)
                + f". The largest single acceptor of that loss is {leader[0]}"
            )
        elif d.net_change < 0:
            answer = (
                f"{donor} loses {abs(d.net_change):.4g} but no other declared group "
                "gains; the loss is not accounted for within this map"
            )
        else:
            answer = f"{donor} does not deplete over the early window ({d.net_change:+.4g})"
        _record(
            f"Does {donor} depopulate into {acceptor}, {ground}, or both, early on?",
            answer
            + ". Co-movement of populations is not a measured flux: these are net "
            "changes over the window, not directed transfer",
            {"donor_net_change": d.net_change, "gain_share": shares},
        )

    # 4. Different dominant behaviour early vs late?
    def _dominant_change(by_group):
        ranked = sorted(by_group.values(), key=lambda m: -abs(m.net_change))
        return ranked[0] if ranked else None

    a, b = _dominant_change(early_by_group), _dominant_change(late_by_group)
    if a is not None and b is not None:
        _record(
            "Are the dominant population changes qualitatively different early and late?",
            (
                f"early, the largest net change is {a.group} ({a.net_change:+.4g}); "
                f"late, it is {b.group} ({b.net_change:+.4g})"
                + (
                    ". Different groups dominate the two windows"
                    if a.group != b.group
                    else ". The same group dominates both windows"
                )
            ),
            {
                "early_dominant": {"group": a.group, "net_change": a.net_change},
                "late_dominant": {"group": b.group, "net_change": b.net_change},
            },
        )

    # 5. Is one kinetic scheme adequate for both?
    _record(
        "Is the same kinetic scheme adequate in both windows?",
        (
            f"early identifiability: {early.identifiability['status']}; "
            f"late identifiability: {late.identifiability['status']}. "
            "Adequacy is answered by the piecewise model comparison, not by this "
            "table; an unidentifiable window means the data do not constrain the "
            "rates, not that the rates are small"
        ),
        {
            "early": early.identifiability["status"],
            "late": late.identifiability["status"],
        },
        kind=MODEL_INFERRED,
    )

    return {
        "acceptor": acceptor,
        "donor": donor,
        "ground": ground,
        "findings": findings,
        "note": (
            "every answer above is read from populations over declared windows. "
            "A net change is not a flux, co-movement is not transfer, and none of "
            "this is a hop count"
        ),
    }
