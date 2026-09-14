"""Single-exponential decay fitting with explicit windows and diagnostics.

The model is deliberately restrictive:

    P(t) = P(t_start) * exp[-(t - t_start) / tau]

The prefactor is *fixed* to the first sample inside the fit window.  There is
no offset and no normalization to the maximum.  This is only meaningful for a
single resolved decay; plateaus, sequential trapping and competing channels
need a different kinetic model, and a high R^2 does not by itself establish a
mechanism.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


class FitError(ValueError):
    """Raised when the requested window cannot support a decay fit."""


@dataclass
class ExponentialFit:
    group: Optional[str]
    tau: float
    tau_unit: str
    amplitude: float
    window: Tuple[float, float]
    n_points: int
    r_squared: float
    rms_residual: float
    max_abs_residual: float
    decay_fraction_in_window: float
    windows_per_tau: float
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["window"] = list(self.window)
        return payload


def _model(t, tau, amplitude, t0):
    return amplitude * np.exp(-(t - t0) / tau)


def fit_single_exponential(
    time: np.ndarray,
    values: np.ndarray,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    group: Optional[str] = None,
    time_unit: str = "ns",
    min_points: int = 5,
) -> ExponentialFit:
    """Fit a fixed-prefactor exponential decay inside an explicit time window."""
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    if time.shape != values.shape:
        raise FitError("time and value arrays have different shapes")

    if time.ndim != 1 or time.size < 2 or not np.isfinite(time).all() or not np.isfinite(values).all():
        raise FitError("finite one-dimensional arrays with >=2 samples required")
    if np.any(np.diff(time) <= 0):
        raise FitError("time must strictly increase")
    if any(v is not None and not np.isfinite(v) for v in [t_start, t_end]):
        raise FitError("window bounds must be finite")
    lo = time[0] if t_start is None else float(t_start)
    hi = time[-1] if t_end is None else float(t_end)
    if hi <= lo:
        raise FitError(f"empty fit window [{lo}, {hi}]")
    mask = (time >= lo) & (time <= hi)
    t = time[mask]
    y = values[mask]
    if t.size < min_points:
        raise FitError(
            f"fit window [{lo}, {hi}] contains {t.size} samples, fewer than {min_points}"
        )

    amplitude = float(y[0])
    if amplitude <= 0:
        raise FitError(
            "the first sample in the fit window is not positive; "
            "a fixed-prefactor decay fit is not defined"
        )

    warnings: List[str] = []
    span = float(t[-1] - t[0])
    # Reject data that does not decay: a rise or a flat trace has no lifetime.
    late = float(np.mean(y[-max(1, y.size // 5) :]))
    early = float(np.mean(y[: max(1, y.size // 5)]))
    if late >= early:
        raise FitError(
            "population does not decay over the requested window "
            f"(mean of last fifth {late:.6g} >= mean of first fifth {early:.6g}); "
            "no single-exponential lifetime is defined"
        )

    t0 = float(t[0])
    guess = max(span / 3.0, np.finfo(float).tiny)
    popt, _ = curve_fit(
        lambda tt, tau: _model(tt, tau, amplitude, t0),
        t,
        y,
        p0=[guess],
        bounds=(1e-12, np.inf),
        maxfev=20000,
    )
    tau = float(popt[0])

    predicted = _model(t, tau, amplitude, t0)
    residual = y - predicted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    decay_fraction = 1.0 - float(y[-1] / amplitude)
    windows_per_tau = span / tau if tau > 0 else float("inf")

    if windows_per_tau < 1.0:
        warnings.append(
            f"the window spans only {windows_per_tau:.2f} tau; the lifetime is an "
            "extrapolation well beyond the simulated time and is not resolved"
        )
    if decay_fraction < 0.2:
        warnings.append(
            f"the population falls by only {100 * decay_fraction:.1f}% inside the window"
        )
    if np.isfinite(r_squared) and r_squared < 0.9:
        warnings.append(
            f"R^2 = {r_squared:.4f}; a single exponential does not describe this trace"
        )
    if np.isfinite(r_squared) and r_squared < 0.0:
        warnings.append(
            "R^2 is negative: the fit is worse than the mean of the data"
        )

    return ExponentialFit(
        group=group,
        tau=tau,
        tau_unit=time_unit,
        amplitude=amplitude,
        window=(float(t[0]), float(t[-1])),
        n_points=int(t.size),
        r_squared=float(r_squared),
        rms_residual=float(np.sqrt(np.mean(residual**2))),
        max_abs_residual=float(np.max(np.abs(residual))),
        decay_fraction_in_window=float(decay_fraction),
        windows_per_tau=float(windows_per_tau),
        warnings=warnings,
    )


def bootstrap_single_exponential(time, per_file_values, *, n_resamples=200,
                                 seed=0, confidence=0.95, **fit_options):
    """Resample whole files, including their noisy first samples; never time rows.

    Interval is conditional on the supplied files and single-decay model. Shared
    trajectory bias and temporal correlations are not independent replications.
    """
    files = np.asarray(per_file_values, dtype=float)
    if files.ndim != 2 or files.shape[1] != np.asarray(time).size or not np.isfinite(files).all():
        raise FitError("bootstrap needs finite (n_files, n_times) populations")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)) or n_resamples < 20:
        raise FitError("bootstrap needs at least 20 resamples")
    if not 0 < confidence < 1:
        raise FitError("confidence must be between zero and one")
    result = {"method": "whole-file percentile bootstrap", "confidence": confidence,
              "seed": seed, "requested": n_resamples, "successful": 0,
              "failed": 0, "poor_fit_resamples": 0, "tau_interval": None,
              "tau_unit": fit_options.get("time_unit", "ns"),
              "note": "Between-file spread only; correlated starts and shared bias can cause undercoverage. Not mechanism confidence."}
    if len(files) < 2:
        result["status"] = "unavailable_single_file"
        return result
    if np.array_equal(files, np.broadcast_to(files[0], files.shape)):
        result["status"] = "unavailable_identical_files"
        return result
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        sample = files[rng.integers(0, len(files), len(files))].mean(axis=0)
        try:
            fit = fit_single_exponential(time, sample, **fit_options)
            if not np.isfinite(fit.tau):
                raise FitError("nonfinite bootstrap lifetime")
            values.append(fit.tau)
            if not np.isfinite(fit.r_squared) or fit.r_squared < .9:
                result["poor_fit_resamples"] += 1
        except (ValueError, RuntimeError):
            result["failed"] += 1
    result["successful"] = len(values)
    if len(values) < max(20, .8 * n_resamples):
        result["status"] = "insufficient_successful_resamples"
    else:
        alpha = (1 - confidence) / 2
        result["tau_interval"] = np.quantile(values, [alpha, 1-alpha]).tolist()
        result["status"] = "estimated" if not result["poor_fit_resamples"] else "model_mismatch_warning"
    return result
