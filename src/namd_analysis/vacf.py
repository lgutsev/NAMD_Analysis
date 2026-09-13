"""Velocity autocorrelation and phonon spectral density from a trajectory.

Two conventions matter here and both are explicit:

* Velocities are built from *Cartesian* displacements.  XDATCAR stores
  fractional coordinates, so each displacement is wrapped into the minimum
  image and multiplied by the cell before differencing.  Differencing raw
  fractional coordinates mixes the three cell axes and leaves a jump of order
  one whenever an atom crosses a boundary.
* The default spectrum is the cosine transform of the VACF, the usual
  vibrational density of states.  ``power`` reproduces the squared-modulus
  convention used by the earlier scripts in this project, which is a
  different quantity with a different line shape.

Nothing here assigns modes to motions; peak positions and widths are
descriptive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .io.xdatcar import Trajectory
from .units import CM1_PER_INV_FS

WINDOWS = ("hann", "hamming", "none")
CONVENTIONS = ("cosine", "power")
ATOM_REDUCTIONS = ("mean", "sum")

#: Standard atomic weights (u) for the elements this project uses.  Extend as
#: needed; an unknown symbol is an error rather than a guessed mass.
ATOMIC_MASSES: Dict[str, float] = {
    "H": 1.008, "D": 2.014, "Li": 6.94, "B": 10.81, "C": 12.011, "N": 14.007,
    "O": 15.999, "F": 18.998, "Na": 22.990, "Mg": 24.305, "Al": 26.982,
    "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "K": 39.098,
    "Ca": 40.078, "Ti": 47.867, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845,
    "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723,
    "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904, "Rb": 85.468,
    "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Ag": 107.868, "Cd": 112.414,
    "In": 114.818, "Sn": 118.710, "Sb": 121.760, "Te": 127.60, "I": 126.904,
    "Cs": 132.905, "Ba": 137.327, "La": 138.905, "Hf": 178.49, "Ta": 180.948,
    "W": 183.84, "Au": 196.967, "Hg": 200.592, "Tl": 204.38, "Pb": 207.2,
    "Bi": 208.980,
}


class VacfError(ValueError):
    """Raised when a VACF cannot be computed as requested."""


def masses_for(symbols: Sequence[str]) -> np.ndarray:
    missing = sorted({s for s in symbols if s not in ATOMIC_MASSES})
    if missing:
        raise VacfError(
            f"no mass on file for element(s) {missing}; add them to ATOMIC_MASSES "
            "or run without --mass-weight"
        )
    return np.array([ATOMIC_MASSES[s] for s in symbols], dtype=float)


def cartesian_velocities(
    trajectory: Trajectory, dt_fs: float, unwrap: bool = True
) -> np.ndarray:
    """Finite-difference velocities in Angstrom/fs, shape ``(nframes-1, natoms, 3)``.

    With ``unwrap`` (the default) each fractional displacement is reduced to
    its minimum image before conversion to Cartesian, which removes the
    boundary-crossing jumps.  Disabling it reproduces an unwrapped raw
    difference and is only useful for comparison against older output.
    """
    if dt_fs <= 0:
        raise VacfError("dt_fs must be positive")
    if trajectory.nframes < 3:
        raise VacfError("at least three frames are needed to form velocities")

    positions = trajectory.positions
    deltas = positions[1:] - positions[:-1]
    if trajectory.direct:
        if unwrap:
            deltas = deltas - np.round(deltas)
        cells = trajectory.lattices[:-1]
        cartesian = np.einsum("fai,fij->faj", deltas, cells)
    else:
        if unwrap:
            # Cartesian frames are already unwrapped by VASP; nothing to fold.
            pass
        cartesian = deltas
    return cartesian / dt_fs


def remove_center_of_mass_motion(
    velocities: np.ndarray, masses: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Subtract the (mass-weighted) centre-of-mass velocity frame by frame.

    Rigid translation of the cell is not a vibration.  Left in, it adds a
    non-decaying term to the VACF that appears in the spectrum as spurious
    intensity at the lowest resolvable frequencies, exactly where the
    low-frequency dynamic-disorder metrics are read off.
    """
    if masses is None:
        weights = np.ones(velocities.shape[1])
    else:
        weights = masses
    com = np.einsum("fai,a->fi", velocities, weights) / weights.sum()
    speed = np.linalg.norm(com, axis=1)
    stats = {
        "com_speed_mean_ang_per_fs": float(np.mean(speed)),
        "com_speed_max_ang_per_fs": float(np.max(speed)),
        "atom_speed_rms_ang_per_fs": float(
            np.sqrt(np.mean(np.sum(velocities**2, axis=2)))
        ),
    }
    return velocities - com[:, None, :], stats


@dataclass
class VacfResult:
    lags_fs: np.ndarray
    values: np.ndarray
    normalized: np.ndarray
    n_origins: np.ndarray
    mass_weighted: bool
    atom_reduction: str
    dt_fs: float
    n_segments: int
    segment_values: Optional[np.ndarray] = None
    motion: Dict[str, Any] = field(default_factory=dict)

    @property
    def c0(self) -> float:
        return float(self.values[0])

    def diagnostics(self) -> Dict[str, Any]:
        norm = self.normalized
        zero_crossing = np.flatnonzero(norm <= 0.0)
        first_zero = (
            float(self.lags_fs[int(zero_crossing[0])]) if zero_crossing.size else None
        )
        integrate = getattr(np, "trapezoid", None) or np.trapz
        if first_zero is not None:
            cut = int(zero_crossing[0]) + 1
            correlation_time = float(integrate(norm[:cut], dx=self.dt_fs))
        else:
            correlation_time = float(integrate(norm, dx=self.dt_fs))
        return {
            "vacf_zero_lag": self.c0,
            "vacf_zero_lag_unit": (
                "u Angstrom^2/fs^2" if self.mass_weighted else "Angstrom^2/fs^2"
            ),
            "first_zero_crossing_fs": first_zero,
            "correlation_time_fs": correlation_time,
            "correlation_time_note": (
                "integral of the normalized VACF to its first zero crossing; "
                "it is not an exponential lifetime"
            ),
            "max_lag_fs": float(self.lags_fs[-1]),
            "n_segments": self.n_segments,
            **self.motion,
        }


def compute_vacf(
    velocities: np.ndarray,
    max_lag: int,
    dt_fs: float,
    masses: Optional[np.ndarray] = None,
    atom_reduction: str = "mean",
) -> VacfResult:
    """Unbiased VACF via the Wiener-Khinchin theorem.

    Lag ``k`` is averaged over the ``n - k`` available time origins, the same
    estimator as the direct double loop but computed with FFTs.
    """
    if atom_reduction not in ATOM_REDUCTIONS:
        raise VacfError(f"atom_reduction must be one of {ATOM_REDUCTIONS}")
    velocities = np.asarray(velocities, dtype=float)
    if velocities.ndim != 3:
        raise VacfError("velocities must have shape (nsteps, natoms, 3)")
    nsteps, natoms, _ = velocities.shape
    if max_lag < 2:
        raise VacfError("max_lag must be at least 2")
    if max_lag > nsteps:
        raise VacfError(
            f"max_lag {max_lag} exceeds the {nsteps} velocity steps available"
        )

    size = 1
    while size < 2 * nsteps:
        size *= 2
    spectrum = np.fft.rfft(velocities, n=size, axis=0)
    correlation = np.fft.irfft(
        (spectrum * np.conjugate(spectrum)).real, n=size, axis=0
    )[:max_lag]

    origins = (nsteps - np.arange(max_lag)).astype(float)
    per_atom = correlation.sum(axis=2) / origins[:, None]

    if masses is not None:
        if masses.shape[0] != natoms:
            raise VacfError(
                f"{masses.shape[0]} masses for {natoms} atoms"
            )
        weighted = per_atom * masses[None, :]
        values = (
            weighted.sum(axis=1) / masses.sum()
            if atom_reduction == "mean"
            else weighted.sum(axis=1)
        )
    else:
        values = (
            per_atom.mean(axis=1) if atom_reduction == "mean" else per_atom.sum(axis=1)
        )

    if values[0] == 0:
        raise VacfError("VACF(0) is zero; the trajectory has no motion")
    return VacfResult(
        lags_fs=np.arange(max_lag) * dt_fs,
        values=values,
        normalized=values / values[0],
        n_origins=origins,
        mass_weighted=masses is not None,
        atom_reduction=atom_reduction,
        dt_fs=dt_fs,
        n_segments=1,
    )


def compute_vacf_segmented(
    velocities: np.ndarray,
    max_lag: int,
    dt_fs: float,
    segment_length: int,
    masses: Optional[np.ndarray] = None,
    atom_reduction: str = "mean",
) -> VacfResult:
    """Average VACFs computed on consecutive non-overlapping segments.

    Segments from one trajectory are not independent samples; the spread
    between them measures how much the correlation function drifts along the
    run, not the statistical error of an ensemble.
    """
    nsteps = velocities.shape[0]
    if segment_length <= max_lag:
        raise VacfError("segment_length must exceed max_lag")
    if segment_length > nsteps:
        raise VacfError(
            f"segment_length {segment_length} exceeds the {nsteps} velocity steps"
        )
    segments = []
    for start in range(0, nsteps - segment_length + 1, segment_length):
        chunk = velocities[start : start + segment_length]
        segments.append(
            compute_vacf(chunk, max_lag, dt_fs, masses, atom_reduction).values
        )
    stack = np.asarray(segments)
    values = stack.mean(axis=0)
    return VacfResult(
        lags_fs=np.arange(max_lag) * dt_fs,
        values=values,
        normalized=values / values[0],
        n_origins=(segment_length - np.arange(max_lag)).astype(float),
        mass_weighted=masses is not None,
        atom_reduction=atom_reduction,
        dt_fs=dt_fs,
        n_segments=len(segments),
        segment_values=stack,
    )


def _window(name: str, size: int) -> np.ndarray:
    if name == "hann":
        return np.hanning(size)
    if name == "hamming":
        return np.hamming(size)
    if name == "none":
        return np.ones(size)
    raise VacfError(f"window must be one of {WINDOWS}")


def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing in bin units, matching ``gaussian_filter1d``.

    numpy's ``symmetric`` padding is what scipy calls ``reflect`` (its default);
    numpy's own ``reflect`` is scipy's ``mirror`` and drops the edge sample.
    The difference only shows at the ends of the array, which for a spectrum is
    the lowest-frequency bins -- exactly where the dynamic-disorder metrics are
    read off.
    """
    if sigma <= 0:
        return values
    radius = int(4 * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, radius, mode="symmetric")
    return np.convolve(padded, kernel, mode="valid")


@dataclass
class Spectrum:
    frequency_cm1: np.ndarray
    intensity: np.ndarray
    convention: str
    window: str
    smooth_sigma: float
    label: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


def spectral_density(
    vacf: VacfResult,
    convention: str = "cosine",
    window: str = "hann",
    smooth_sigma: float = 0.0,
    label: str = "",
) -> Spectrum:
    """Fourier transform the VACF onto a cm^-1 grid.

    ``cosine`` returns 2 dt Re[FFT(C w)], the vibrational density of states.
    ``power`` returns |dt FFT(C w)|^2, the convention of the earlier scripts.
    The zero-frequency bin is dropped, as in that earlier output.
    """
    if convention not in CONVENTIONS:
        raise VacfError(f"convention must be one of {CONVENTIONS}")
    values = vacf.values * _window(window, vacf.values.size)
    transform = np.fft.rfft(values)
    dt = vacf.dt_fs
    if convention == "cosine":
        intensity = 2.0 * dt * transform.real
    else:
        intensity = np.abs(dt * transform) ** 2
    frequency = np.fft.rfftfreq(values.size, d=dt) * CM1_PER_INV_FS
    intensity = gaussian_smooth(intensity, smooth_sigma)
    keep = frequency > 0
    return Spectrum(
        frequency_cm1=frequency[keep],
        intensity=intensity[keep],
        convention=convention,
        window=window,
        smooth_sigma=smooth_sigma,
        label=label,
        meta={
            "dt_fs": dt,
            "max_lag_fs": float(vacf.lags_fs[-1]),
            "resolution_cm1": float(frequency[1] - frequency[0]),
            "nyquist_cm1": float(frequency[-1]),
            "mass_weighted": vacf.mass_weighted,
            "n_segments": vacf.n_segments,
        },
    )


def trajectory_spectrum(
    trajectory: Trajectory,
    dt_fs: float,
    max_lag: int,
    mass_weight: bool = False,
    unwrap: bool = True,
    segment_length: Optional[int] = None,
    convention: str = "cosine",
    window: str = "hann",
    smooth_sigma: float = 0.0,
    atom_reduction: str = "mean",
    remove_com: bool = True,
    label: str = "",
) -> Tuple[VacfResult, Spectrum]:
    """Full path from a trajectory to a spectral density."""
    masses = masses_for(trajectory.atom_symbols()) if mass_weight else None
    velocities = cartesian_velocities(trajectory, dt_fs, unwrap=unwrap)
    motion: Dict[str, Any] = {}
    if remove_com:
        velocities, motion = remove_center_of_mass_motion(velocities, masses)
    else:
        _, motion = remove_center_of_mass_motion(velocities, masses)
    motion["center_of_mass_motion_removed"] = remove_com
    if segment_length:
        result = compute_vacf_segmented(
            velocities, max_lag, dt_fs, segment_length, masses, atom_reduction
        )
    else:
        result = compute_vacf(velocities, max_lag, dt_fs, masses, atom_reduction)
    result.motion = motion
    spectrum = spectral_density(
        result,
        convention=convention,
        window=window,
        smooth_sigma=smooth_sigma,
        label=label,
    )
    return result, spectrum
