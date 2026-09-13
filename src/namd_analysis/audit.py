"""EIGTXT / NATXT dimension checks and pair-resolved coupling statistics."""

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
from .units import NAC_UNITS, hbar_over_dt_mev, nac_to_mev


class AuditError(ValueError):
    """Raised when the run directory cannot be audited as given."""


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
) -> AuditResult:
    """Audit one run directory.

    ``nac_unit`` must come from the code that produced NATXT.  It is never
    inferred from the magnitude of the values.  An explicit ``dt_fs``
    overrides POTIM from ``inp``.
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

    # A coupling file whose largest magnitude is hit by many samples has been
    # capped somewhere upstream.  Report it; do not undo it and do not treat
    # the capped values as measurements of the coupling.
    offdiag_abs = abs_nac[:, offdiag]
    global_max = float(offdiag_abs.max())
    at_max = int(np.count_nonzero(offdiag_abs == global_max))
    if at_max > 2:
        checks.append(
            _check(
                "nac_clipping",
                "capped",
                f"{at_max} of {offdiag_abs.size} off-diagonal samples sit exactly at "
                f"|NAC| = {global_max:.6g} meV, so the file was capped before it was "
                "written; the affected samples are a bound, not a value",
            )
        )
    else:
        checks.append(
            _check(
                "nac_clipping",
                "ok",
                f"the largest |NAC| ({global_max:.6g} meV) is reached by {at_max} sample(s)",
            )
        )

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
        "hbar_over_dt_meV": hbar_over_dt_mev(step),
        "hbar_over_dt_note": (
            "diagnostic scale only; it is not a bound on the coupling and no "
            "sample is filtered by it"
        ),
        "threshold_meV": threshold_mev,
        "samples_above_threshold": int(np.count_nonzero(offdiag_abs > threshold_mev)),
        "offdiagonal_samples": int(offdiag_abs.size),
        "samples_at_global_max": at_max,
        "global_max_is_a_cap": at_max > 2,
    }

    pairs: List[PairStat] = []
    for i in range(nstates):
        for j in range(i + 1, nstates):
            series = nac_mev[:, i, j]
            gap = gaps[:, i, j]
            abs_series = np.abs(series)
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
        params=run.params,
        checks=checks,
        energies=energy_summary,
        couplings=coupling_summary,
        dephasing=dephasing,
        initial_conditions=initial,
        pairs=pairs,
    )
