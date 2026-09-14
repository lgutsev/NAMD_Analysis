"""Physical constants and unit conversions used across the package.

Every conversion here is explicit on purpose: NAC files carry no unit
metadata, so the unit must come from the producing code, not from the
magnitude of the numbers.
"""

from __future__ import annotations

#: Reduced Planck constant in eV fs.
HBAR_EV_FS = 0.6582119569509066

#: 1 fs^-1 expressed in cm^-1 (c = 29979245800 cm/s).
CM1_PER_INV_FS = 1.0e15 / 29979245800.0  # 33356.409519815204

#: Femtoseconds per nanosecond.
FS_PER_NS = 1.0e6

#: NAC units this package can convert from.
NAC_UNITS = ("eV", "meV", "fs^-1")


def nac_to_mev(values, unit: str, dt_fs: float):
    """Convert a NAC array to meV.

    ``eV``/``meV`` are treated as a coupling energy already multiplied by
    hbar.  ``fs^-1`` is a time-derivative coupling d_ij = <i|d/dt|j>, which
    becomes an energy as hbar * d_ij.  ``dt_fs`` is accepted for symmetry
    with callers and is unused for the supported units; it is *not* used to
    silently rescale anything.
    """
    if unit == "eV":
        return values * 1000.0
    if unit == "meV":
        return values * 1.0
    if unit == "fs^-1":
        return values * HBAR_EV_FS * 1000.0
    raise ValueError(f"unsupported NAC unit {unit!r}; choose one of {NAC_UNITS}")


def hbar_over_dt_ev(dt_fs: float) -> float:
    """hbar/dt in eV: the energy scale set by the electronic timestep.

    For dt = 1 fs this is 0.658 eV.  A coupling approaching this scale is a
    signal that the finite-difference evaluation of the NAC has broken down
    over one step, not a measurement of an arbitrarily large physical matrix
    element.  It is a diagnostic scale: nothing here filters, rescales or
    rejects a sample because of it.
    """
    if dt_fs <= 0:
        raise ValueError("dt_fs must be positive")
    return HBAR_EV_FS / dt_fs


def hbar_over_dt_mev(dt_fs: float) -> float:
    """hbar/dt in meV.  See :func:`hbar_over_dt_ev`."""
    return hbar_over_dt_ev(dt_fs) * 1000.0


def inv_fs_to_cm1(freq_inv_fs):
    """Convert a frequency in fs^-1 to cm^-1."""
    return freq_inv_fs * CM1_PER_INV_FS
