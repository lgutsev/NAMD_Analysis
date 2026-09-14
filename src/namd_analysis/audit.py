"""EIGTXT / NATXT dimension checks and pair-resolved coupling statistics.

The coupling magnitudes are read against ``hbar/dt``, the energy scale set by
the discrete electronic timestep.  For dt = 1 fs, ``hbar/dt`` is 0.658 eV.  A
coupling approaching that scale says the finite-difference evaluation of the
NAC has broken down over one step; it does not say the physical matrix element
is that large.  The audit reports where a distribution sits relative to that
scale and never filters, rescales or rejects a sample because of it.

A magnitude that many samples share *exactly* is a separate observation.  It
did not come out of the dynamics, so some upstream step put it there; this
module reports it as an engineered ceiling and says so in those words.  What
that ceiling did to the statistics depends entirely on the upstream rule --
zeroing a pathological sample and truncating a valid one have opposite
consequences -- and that rule is not recoverable from the file.  It is
therefore declared, through :mod:`namd_analysis.nac_policy`, or left
undeclared and not guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .io.hefei import (
    load_run_directory,
    read_dephtime,
    read_eigtxt,
    read_inicon,
    read_natxt,
)
from .nac_policy import UNDECLARED, NacPolicy
from .units import NAC_UNITS, hbar_over_dt_ev, hbar_over_dt_mev, nac_to_mev


class AuditError(ValueError):
    """Raised when the run directory cannot be audited as given."""


#: Fractions of ``hbar/dt`` at which the coupling distribution is counted.
TIMESTEP_LIMIT_FRACTIONS = (0.8, 0.9)

#: Above this many off-diagonal samples sharing one exact magnitude, that
#: magnitude is reported as an engineered ceiling.  Two samples is the
#: antisymmetric pair of a single genuine extremum.
REPEATED_CEILING_MIN_SAMPLES = 2

#: Default fraction of samples inside the 0.8 * hbar/dt region that is treated
#: as unremarkable.  Zero: any sample there is named.
DEFAULT_WARNING_FRACTION = 0.0


def _ceiling_policy_detail(
    policy: Optional[NacPolicy], ceiling_mev: float, dt_fs: float
) -> str:
    """The sentence that follows an engineered-ceiling observation."""
    if policy is None:
        return (
            ". No upstream NAC handling policy was declared for this dataset, so "
            "the rule that produced the ceiling is not known here and is not "
            "guessed at. This is a reported observation, not a defect"
        )
    ceiling_ev = ceiling_mev / 1000.0
    pieces = []
    warning = policy.warning_threshold_ev
    if warning is not None and abs(ceiling_ev - warning) <= 1e-6 * max(warning, 1.0):
        pieces.append(
            f"this matches the declared upstream warning threshold of "
            f"{warning:g} eV"
        )
    limit_ev = policy.limit_ev(dt_fs)
    if limit_ev is not None:
        pieces.append(
            f"the declared numerical limit is hbar/dt = {limit_ev:.4g} eV at "
            f"dt = {dt_fs:g} fs"
        )
    if policy.reject_above_ev is not None and policy.action_above_limit:
        pieces.append(
            f"the declared policy is to {policy.action_above_limit} couplings "
            f"above {policy.reject_above_ev:g} eV"
        )
    if policy.truncates_valid_values:
        pieces.append(
            "that policy replaces a valid coupling with the limit, so every "
            "mean, RMS and integral built on the affected samples is a lower "
            "bound"
        )
    else:
        pieces.append(
            "that policy does not replace valid couplings with the ceiling "
            "value, so the affected statistics are not described as lower bounds"
        )
    return ". " + "; ".join(pieces) if pieces else ""


def _ceiling_interpretation(
    policy: Optional[NacPolicy], ceiling_mev: Optional[float]
) -> str:
    if ceiling_mev is None:
        return "no magnitude is shared by enough samples to indicate a ceiling"
    base = (
        f"repeated values at exactly {ceiling_mev / 1000.0:.4g} eV are consistent "
        "with an intentionally imposed upstream NAC safety ceiling"
    )
    if policy is None:
        return (
            base
            + "; no policy was declared for this dataset, so no particular "
            "upstream rule is attributed and no claim is made about how the "
            "affected statistics were changed"
        )
    if policy.truncates_valid_values:
        return (
            base
            + "; the declared policy truncates valid couplings at the limit, so "
            "means, RMS values and integrals over the affected samples are "
            "lower bounds"
        )
    return (
        base
        + f"; the declared policy is to {policy.action_above_limit or 'leave'} "
        "couplings beyond the numerical limit rather than to truncate valid "
        "ones, so the affected statistics are not lower bounds on that account"
    )


@dataclass
class PairStat:
    i: int
    j: int
    band_i: Optional[int]
    band_j: Optional[int]
    mean_gap_ev: float
    rms_gap_ev: float
    mean_abs_nac_mev: float
    rms_nac_mev: float
    p95_abs_nac_mev: float
    max_abs_nac_mev: float
    samples_above_threshold: int
    samples_at_global_max: int
    samples_at_engineered_ceiling: int
    max_abs_nac_over_hbar_dt: float
    fraction_above_0p8_hbar_dt: float
    fraction_above_0p9_hbar_dt: float
    sample_sum_mev: float
    time_integral_mev_fs: float

    def as_row(self) -> List[Any]:
        return [
            self.i,
            self.j,
            self.band_i,
            self.band_j,
            self.mean_gap_ev,
            self.rms_gap_ev,
            self.mean_abs_nac_mev,
            self.rms_nac_mev,
            self.p95_abs_nac_mev,
            self.max_abs_nac_mev,
            self.samples_above_threshold,
            self.samples_at_global_max,
            self.samples_at_engineered_ceiling,
            self.max_abs_nac_over_hbar_dt,
            self.fraction_above_0p8_hbar_dt,
            self.fraction_above_0p9_hbar_dt,
            self.sample_sum_mev,
            self.time_integral_mev_fs,
        ]


PAIR_HEADER = [
    "state_i",
    "state_j",
    "band_i",
    "band_j",
    "mean_gap_eV",
    "rms_gap_eV",
    "mean_abs_nac_meV",
    "rms_nac_meV",
    "p95_abs_nac_meV",
    "max_abs_nac_meV",
    "samples_above_threshold",
    "samples_at_global_max",
    "samples_at_engineered_ceiling",
    "max_abs_nac_over_hbar_dt",
    "fraction_above_0p8_hbar_dt",
    "fraction_above_0p9_hbar_dt",
    "sample_sum_meV",
    "time_integral_meV_fs",
]


@dataclass
class AuditResult:
    directory: Path
    nframes: int
    nstates: int
    dt_fs: float
    dt_source: str
    nac_unit: str
    threshold_mev: float
    policy: Optional[NacPolicy] = None
    params: Dict[str, Any] = field(default_factory=dict)
    checks: List[Dict[str, Any]] = field(default_factory=list)
    energies: Dict[str, Any] = field(default_factory=dict)
    couplings: Dict[str, Any] = field(default_factory=dict)
    dephasing: Optional[Dict[str, Any]] = None
    initial_conditions: Optional[Dict[str, Any]] = None
    pairs: List[PairStat] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "directory": str(self.directory),
            "nframes": self.nframes,
            "nstates": self.nstates,
            "dt_fs": self.dt_fs,
            "dt_source": self.dt_source,
            "nac_unit": self.nac_unit,
            "threshold_meV": self.threshold_mev,
            "nac_policy": (
                self.policy.as_dict() if self.policy is not None else dict(UNDECLARED)
            ),
            "inp": self.params,
            "checks": self.checks,
            "energies": self.energies,
            "couplings": self.couplings,
            "dephasing": self.dephasing,
            "initial_conditions": self.initial_conditions,
        }


def _check(name: str, status: str, detail: str) -> Dict[str, Any]:
    return {"check": name, "status": status, "detail": detail}


def audit_run(
    directory,
    nac_unit: str,
    dt_fs: Optional[float] = None,
    threshold_mev: float = 0.5,
    max_pairs: Optional[int] = None,
    policy: Optional[NacPolicy] = None,
    warn_fraction: float = DEFAULT_WARNING_FRACTION,
) -> AuditResult:
    """Audit one run directory.

    ``nac_unit`` must come from the code that produced NATXT.  It is never
    inferred from the magnitude of the values.  An explicit ``dt_fs``
    overrides POTIM from ``inp``.  ``policy`` is the *declared* upstream NAC
    handling rule for this dataset; without one, nothing about upstream
    handling is assumed.  ``warn_fraction`` is the fraction of samples inside
    the ``0.8 * hbar/dt`` region treated as unremarkable.
    """
    directory = Path(directory)
    if nac_unit not in NAC_UNITS:
        raise AuditError(f"nac_unit must be one of {NAC_UNITS}, got {nac_unit!r}")

    run = load_run_directory(directory)
    for required in ("EIGTXT", "NATXT"):
        if required not in run.present:
            raise AuditError(f"{directory}: {required} not found")

    energies = read_eigtxt(directory / "EIGTXT")
    nframes, nstates = energies.shape
    nac = read_natxt(directory / "NATXT", nstates)

    checks: List[Dict[str, Any]] = []
    if nac.shape[0] != nframes:
        raise AuditError(
            f"{directory}: EIGTXT has {nframes} frames, NATXT has {nac.shape[0]}"
        )
    checks.append(
        _check("frame_count", "ok", f"EIGTXT and NATXT both hold {nframes} frames")
    )

    declared = run.declared_nstates
    bmin = run.params.get("BMIN")
    if declared is None:
        checks.append(
            _check("band_window", "skipped", "inp absent or missing BMIN/BMAX")
        )
    elif declared != nstates:
        checks.append(
            _check(
                "band_window",
                "mismatch",
                f"inp declares BMAX-BMIN+1 = {declared} states, files hold {nstates}",
            )
        )
    else:
        checks.append(
            _check("band_window", "ok", f"inp BMIN/BMAX and files agree on {nstates} states")
        )

    nsw = run.params.get("NSW")
    if isinstance(nsw, int):
        status = "ok" if nframes in (nsw, nsw - 1) else "mismatch"
        checks.append(
            _check(
                "frame_count_vs_NSW",
                status,
                f"inp NSW = {nsw}; coupling files hold {nframes} frames "
                "(CA-NAC normally writes NSW-1)",
            )
        )

    if dt_fs is not None:
        step, source = float(dt_fs), "--dt-fs"
    elif run.dt_fs_from_inp is not None:
        step, source = run.dt_fs_from_inp, "inp POTIM"
    else:
        raise AuditError(
            f"{directory}: no timestep available; pass --dt-fs explicitly"
        )
    if step <= 0:
        raise AuditError(f"timestep must be positive, got {step}")

    nac_mev = nac_to_mev(nac, nac_unit, step)
    offdiag = ~np.eye(nstates, dtype=bool)
    diag_values = np.abs(nac_mev[:, np.eye(nstates, dtype=bool)])
    checks.append(
        _check(
            "nac_diagonal",
            "ok" if np.max(diag_values) == 0 else "nonzero",
            f"largest |NAC_ii| = {np.max(diag_values):.4g} meV",
        )
    )
    asym = np.abs(nac_mev + np.transpose(nac_mev, (0, 2, 1)))
    scale = float(np.max(np.abs(nac_mev))) or 1.0
    checks.append(
        _check(
            "nac_antisymmetry",
            "ok" if np.max(asym) <= 1e-6 * scale else "not antisymmetric",
            f"largest |NAC_ij + NAC_ji| = {np.max(asym):.4g} meV",
        )
    )

    gaps = energies[:, :, None] - energies[:, None, :]
    abs_nac = np.abs(nac_mev)

    # The coupling distribution is read against the energy scale set by the
    # electronic timestep, hbar/dt.  See the module docstring: approaching that
    # scale is a breakdown of the finite-difference NAC evaluation, not a
    # measurement of an enormous physical matrix element.
    offdiag_abs = abs_nac[:, offdiag]
    global_max = float(offdiag_abs.max())
    at_max = int(np.count_nonzero(offdiag_abs == global_max))
    limit_mev = hbar_over_dt_mev(step)
    limit_ev = hbar_over_dt_ev(step)
    n_samples = int(offdiag_abs.size)

    above = {
        fraction: int(np.count_nonzero(offdiag_abs > fraction * limit_mev))
        for fraction in TIMESTEP_LIMIT_FRACTIONS
    }
    fractions = {key: value / n_samples for key, value in above.items()}

    # A magnitude reached exactly by many samples at once did not arise from
    # the dynamics; some upstream step put it there.  Report that observation
    # without naming a cause the file cannot support.
    ceiling_mev = global_max if at_max > REPEATED_CEILING_MIN_SAMPLES else None
    ceiling_counts = (
        np.count_nonzero(abs_nac == ceiling_mev, axis=0)
        if ceiling_mev is not None
        else np.zeros((nstates, nstates), dtype=int)
    )

    lead = (
        f"for dt = {step:g} fs, hbar/dt = {limit_ev:.4g} eV "
        f"({limit_mev:.6g} meV); the largest off-diagonal |NAC| is "
        f"{global_max:.6g} meV = {global_max / limit_mev:.4g} x hbar/dt"
    )
    if fractions[0.8] > warn_fraction:
        checks.append(
            _check(
                "nac_timestep_limit",
                "near_timestep_limit",
                lead
                + f". {above[0.8]} of {n_samples} off-diagonal samples "
                f"({fractions[0.8]:.3%}) exceed 0.8 x hbar/dt and {above[0.9]} "
                f"({fractions[0.9]:.3%}) exceed 0.9 x hbar/dt. The coupling "
                "distribution reaches the numerical-safety region associated with "
                "the finite electronic timestep; couplings approaching this scale "
                "should be treated as numerically pathological rather than "
                "interpreted as arbitrarily large physical matrix elements",
            )
        )
    else:
        checks.append(
            _check(
                "nac_timestep_limit",
                "ok",
                lead
                + f", so no sample reaches the numerical-safety region "
                f"(0.8 x hbar/dt = {0.8 * limit_mev:.6g} meV)",
            )
        )

    if ceiling_mev is None:
        checks.append(
            _check(
                "nac_repeated_ceiling",
                "ok",
                f"the largest |NAC| ({global_max:.6g} meV) is reached by "
                f"{at_max} sample(s), which is not a repeated ceiling",
            )
        )
    else:
        detail = (
            f"{at_max} of {n_samples} off-diagonal samples sit at exactly "
            f"|NAC| = {ceiling_mev:.6g} meV ({ceiling_mev / 1000.0:.4g} eV). "
            "Repeated values at one exact magnitude are consistent with an "
            "intentionally imposed upstream NAC safety ceiling rather than with "
            "the dynamics"
        )
        detail += _ceiling_policy_detail(policy, ceiling_mev, step)
        checks.append(_check("nac_repeated_ceiling", "engineered_ceiling", detail))

    energy_summary = {
        "unit": "eV",
        "per_state_mean": energies.mean(axis=0).tolist(),
        "per_state_std": energies.std(axis=0, ddof=1).tolist(),
        "per_state_min": energies.min(axis=0).tolist(),
        "per_state_max": energies.max(axis=0).tolist(),
        "bands": (
            [int(bmin) + i for i in range(nstates)] if isinstance(bmin, int) else None
        ),
    }

    coupling_summary = {
        "unit": "meV",
        "declared_input_unit": nac_unit,
        "mean_abs_offdiagonal": float(abs_nac[:, offdiag].mean()),
        "rms_offdiagonal": float(np.sqrt(np.mean(nac_mev[:, offdiag] ** 2))),
        "max_abs_offdiagonal": float(abs_nac[:, offdiag].max()),
        "threshold_meV": threshold_mev,
        "samples_above_threshold": int(np.count_nonzero(offdiag_abs > threshold_mev)),
        "offdiagonal_samples": n_samples,
        "timestep_limit": {
            "dt_fs": step,
            "hbar_over_dt_eV": limit_ev,
            "hbar_over_dt_meV": limit_mev,
            "global_max_nac_meV": global_max,
            "global_max_over_hbar_dt": global_max / limit_mev,
            "samples_above_0p8_hbar_dt": above[0.8],
            "fraction_above_0p8_hbar_dt": fractions[0.8],
            "samples_above_0p9_hbar_dt": above[0.9],
            "fraction_above_0p9_hbar_dt": fractions[0.9],
            "warning_fraction": warn_fraction,
            "meaning": (
                "hbar/dt is the energy scale set by the discrete electronic "
                "timestep. A coupling approaching it means the finite-difference "
                "evaluation of the NAC has broken down over one step, not that "
                "the physical matrix element is that large. It is a diagnostic "
                "scale: no sample is filtered, rescaled or rejected here"
            ),
        },
        "engineered_ceiling": {
            "detected": ceiling_mev is not None,
            "ceiling_meV": ceiling_mev,
            "ceiling_eV": None if ceiling_mev is None else ceiling_mev / 1000.0,
            "samples_at_ceiling": at_max if ceiling_mev is not None else 0,
            "fraction_at_ceiling": (
                at_max / n_samples if ceiling_mev is not None else 0.0
            ),
            "ceiling_over_hbar_dt": (
                None if ceiling_mev is None else ceiling_mev / limit_mev
            ),
            "statistics_are_a_lower_bound": bool(
                ceiling_mev is not None
                and policy is not None
                and policy.truncates_valid_values
            ),
            "interpretation": _ceiling_interpretation(policy, ceiling_mev),
        },
        "samples_at_global_max": at_max,
        "nac_policy": policy.as_dict() if policy is not None else dict(UNDECLARED),
    }

    pairs: List[PairStat] = []
    for i in range(nstates):
        for j in range(i + 1, nstates):
            series = nac_mev[:, i, j]
            gap = gaps[:, i, j]
            abs_series = np.abs(series)
            n_pair = int(abs_series.size)
            pairs.append(
                PairStat(
                    i=i,
                    j=j,
                    band_i=int(bmin) + i if isinstance(bmin, int) else None,
                    band_j=int(bmin) + j if isinstance(bmin, int) else None,
                    mean_gap_ev=float(np.mean(np.abs(gap))),
                    rms_gap_ev=float(np.sqrt(np.mean(gap**2))),
                    mean_abs_nac_mev=float(np.mean(abs_series)),
                    rms_nac_mev=float(np.sqrt(np.mean(series**2))),
                    p95_abs_nac_mev=float(np.percentile(abs_series, 95)),
                    max_abs_nac_mev=float(np.max(abs_series)),
                    samples_above_threshold=int(np.count_nonzero(abs_series > threshold_mev)),
                    samples_at_global_max=int(np.count_nonzero(abs_series == global_max)),
                    samples_at_engineered_ceiling=int(ceiling_counts[i, j]),
                    max_abs_nac_over_hbar_dt=float(np.max(abs_series) / limit_mev),
                    fraction_above_0p8_hbar_dt=float(
                        np.count_nonzero(abs_series > 0.8 * limit_mev) / n_pair
                    ),
                    fraction_above_0p9_hbar_dt=float(
                        np.count_nonzero(abs_series > 0.9 * limit_mev) / n_pair
                    ),
                    sample_sum_mev=float(np.sum(abs_series)),
                    time_integral_mev_fs=float(np.sum(abs_series) * step),
                )
            )
    pairs.sort(key=lambda p: p.mean_abs_nac_mev, reverse=True)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    dephasing = None
    if "DEPHTIME" in run.present:
        table = read_dephtime(directory / "DEPHTIME")
        if table.shape[0] != nstates:
            checks.append(
                _check(
                    "dephtime_shape",
                    "mismatch",
                    f"DEPHTIME is {table.shape[0]}x{table.shape[1]}, expected {nstates}",
                )
            )
        upper = table[np.triu_indices(table.shape[0], k=1)]
        dephasing = {
            "unit": "fs (as written by the producing code)",
            "min": float(np.min(upper)),
            "median": float(np.median(upper)),
            "max": float(np.max(upper)),
        }

    initial = None
    if "INICON" in run.present:
        table = read_inicon(directory / "INICON")
        initial = {
            "n_samples": int(table.shape[0]),
            "start_frame_min": float(np.min(table[:, 0])),
            "start_frame_max": float(np.max(table[:, 0])),
            "start_bands": sorted({float(v) for v in table[:, 1]}),
        }
        nsample = run.params.get("NSAMPLE")
        if isinstance(nsample, int):
            checks.append(
                _check(
                    "inicon_count",
                    "ok" if table.shape[0] == nsample else "mismatch",
                    f"INICON holds {table.shape[0]} rows, inp NSAMPLE = {nsample}",
                )
            )
        if np.max(table[:, 0]) > nframes:
            checks.append(
                _check(
                    "inicon_range",
                    "mismatch",
                    f"INICON start frame {int(np.max(table[:, 0]))} exceeds the "
                    f"{nframes} frames present",
                )
            )

    return AuditResult(
        directory=directory,
        nframes=int(nframes),
        nstates=int(nstates),
        dt_fs=step,
        dt_source=source,
        nac_unit=nac_unit,
        threshold_mev=threshold_mev,
        policy=policy,
        params=run.params,
        checks=checks,
        energies=energy_summary,
        couplings=coupling_summary,
        dephasing=dephasing,
        initial_conditions=initial,
        pairs=pairs,
    )
