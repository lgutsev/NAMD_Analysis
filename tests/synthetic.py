"""Synthetic inputs with known answers, shared by the tests."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from namd_analysis.units import CM1_PER_INV_FS, FS_PER_NS

INP_TEMPLATE = """&NAMDPARA
  BMIN       = {bmin}
  BMAX       = {bmax}

  NSAMPLE    = {nsample}
  NTRAJ      = 1000
  NSW        = {nsw}
  NELM       = 10

  TEMP       = 300
  NAMDTIME   = 1000
  POTIM      = {potim}

  ALGO       = "DISH"
  LHOLE      = .FALSE.
/
"""


def write_run_directory(
    root: Path,
    nframes: int = 200,
    nstates: int = 3,
    bmin: int = 100,
    coupling_ev: float = 0.01,
    cap_ev: Optional[float] = None,
    gap_ev: float = 0.5,
) -> Path:
    """A minimal CA-NAC style run directory with a known coupling."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    energies = np.zeros((nframes, nstates))
    for state in range(nstates):
        energies[:, state] = state * gap_ev + 0.01 * rng.standard_normal(nframes)
    np.savetxt(root / "EIGTXT", energies)

    nac = np.zeros((nframes, nstates, nstates))
    for i in range(nstates):
        for j in range(i + 1, nstates):
            series = coupling_ev * np.sin(
                np.linspace(0, 8 * np.pi, nframes) + i + j
            )
            if cap_ev is not None:
                series = np.clip(series, -cap_ev, cap_ev)
            nac[:, i, j] = series
            nac[:, j, i] = -series
    np.savetxt(root / "NATXT", nac.reshape(nframes, nstates * nstates))

    (root / "inp").write_text(
        INP_TEMPLATE.format(
            bmin=bmin, bmax=bmin + nstates - 1, nsample=4, nsw=nframes + 1, potim=1.0
        ),
        encoding="utf-8",
    )
    inicon = np.column_stack(
        [np.array([10, 20, 30, 40]), np.full(4, bmin + nstates - 1)]
    )
    np.savetxt(root / "INICON", inicon, fmt="%d")
    dephtime = np.full((nstates, nstates), 5.0)
    np.fill_diagonal(dephtime, 0.0)
    np.savetxt(root / "DEPHTIME", dephtime)
    return root


def write_legacy_fit(root: Path, tau_ns: float, r_squared: float) -> Path:
    path = root / "fitting_results_final.txt"
    path.write_text(
        f"Fitted parameter (A): {tau_ns:.4f} ns\n"
        f"R² value: {r_squared:.4f}\n\n"
        "Time (ns), Fitted CBM Population\n"
        "0.000000, 1.00000000\n",
        encoding="utf-8",
    )
    return path


def write_shprop_set(
    root: Path,
    n_files: int = 4,
    nsteps: int = 400,
    dt_fs: float = 1000.0,
    tau_fs: float = 120000.0,
    nstates: int = 2,
    noise: float = 0.0,
    seed: int = 1,
) -> list:
    """SHPROP files whose two populations sum to one at every step.

    Column 0 is time in fs, column 1 stands in for the energy column written
    by Hefei-NAMD, and the remaining columns are populations.
    """
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    time = np.arange(nsteps) * dt_fs
    paths = []
    for index in range(n_files):
        cbm = np.exp(-time / tau_fs)
        if noise:
            cbm = cbm + noise * rng.standard_normal(nsteps)
            cbm = np.clip(cbm, 0.0, 1.0)
        columns = [time, np.full(nsteps, -1.5)]
        if nstates == 2:
            columns += [1.0 - cbm, cbm]
        else:
            share = (1.0 - cbm) / (nstates - 1)
            columns += [share] * (nstates - 1) + [cbm]
        table = np.column_stack(columns)
        path = root / f"SHPROP.{index + 1}"
        np.savetxt(path, table)
        paths.append(path)
    return paths


def write_kinetic_shprop_set(
    root: Path,
    rate_matrix,
    p0,
    n_files: int = 4,
    nsteps: int = 400,
    dt_fs: float = 1000.0,
    noise: float = 0.0,
    seed: int = 5,
) -> list:
    """SHPROP files whose populations follow a known master equation.

    One population column per state, so a state map can address them directly:
    column 0 is time in fs, column 1 stands in for the energy column, and the
    rest are populations of the states in ``rate_matrix`` order.
    """
    from scipy.linalg import expm

    root.mkdir(parents=True, exist_ok=True)
    rate_matrix = np.asarray(rate_matrix, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    nstates = rate_matrix.shape[0]
    time_fs = np.arange(nsteps) * dt_fs
    dt_ns = dt_fs / FS_PER_NS

    step = expm(rate_matrix * dt_ns)
    exact = np.empty((nsteps, nstates))
    exact[0] = p0
    for index in range(1, nsteps):
        exact[index] = step @ exact[index - 1]

    rng = np.random.default_rng(seed)
    paths = []
    for index in range(n_files):
        populations = exact.copy()
        if noise:
            populations = populations + noise * rng.standard_normal(populations.shape)
            populations = np.clip(populations, 0.0, None)
            totals = populations.sum(axis=1, keepdims=True)
            populations = populations / np.where(totals > 0, totals, 1.0)
        table = np.column_stack([time_fs, np.full(nsteps, -1.5), populations])
        path = root / f"SHPROP.{index + 1}"
        np.savetxt(path, table)
        paths.append(path)
    return paths


def kinetic_config(names, recombined=None) -> dict:
    """A complete state map for ``write_kinetic_shprop_set`` output."""
    columns = list(range(2, 2 + len(names)))
    return {
        "name": "synthetic_kinetics",
        "time_column": 0,
        "time_unit": "fs",
        "population_columns": columns,
        "groups": {name: [column] for name, column in zip(names, columns)},
        "complete_population": True,
        "recombined_group": recombined,
    }


def write_xdatcar(
    path: Path,
    frequencies_cm1: Sequence[float] = (100.0,),
    nframes: int = 2000,
    dt_fs: float = 1.0,
    amplitude_ang: float = 0.05,
    cell_ang: float = 10.0,
    drift_ang_per_fs: float = 0.0,
    species: Sequence[str] = ("C",),
    counts: Sequence[int] = (2,),
) -> Path:
    """An XDATCAR whose atoms oscillate at exactly ``frequencies_cm1``.

    A non-zero ``drift_ang_per_fs`` pushes the atoms across the cell boundary
    so that minimum-image unwrapping is actually exercised.
    """
    natoms = int(sum(counts))
    time = np.arange(nframes) * dt_fs
    positions = np.zeros((nframes, natoms, 3))
    base = np.linspace(0.2, 0.8, natoms)
    for atom in range(natoms):
        frequency = frequencies_cm1[atom % len(frequencies_cm1)]
        omega = 2 * np.pi * frequency / CM1_PER_INV_FS  # rad/fs
        displacement = amplitude_ang * np.sin(omega * time + atom)
        cartesian = base[atom] * cell_ang + displacement + drift_ang_per_fs * time
        positions[:, atom, 0] = (cartesian / cell_ang) % 1.0
        positions[:, atom, 1] = base[atom]
        positions[:, atom, 2] = base[atom]

    lines = [
        "synthetic",
        "1.0",
        f"  {cell_ang:.8f}   0.00000000   0.00000000",
        f"  0.00000000   {cell_ang:.8f}   0.00000000",
        f"  0.00000000   0.00000000   {cell_ang:.8f}",
        "  " + "  ".join(species),
        "  " + "  ".join(str(c) for c in counts),
    ]
    for frame in range(nframes):
        lines.append(f"Direct configuration=  {frame + 1}")
        for atom in range(natoms):
            x, y, z = positions[frame, atom]
            lines.append(f"  {x:.8f}  {y:.8f}  {z:.8f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_spectrum_file(
    path: Path, peaks_cm1: Sequence[float], width_cm1: float = 10.0, scale: float = 1e22
) -> Path:
    frequency = np.arange(1, 1000) * 1.0
    intensity = np.zeros_like(frequency)
    for peak in peaks_cm1:
        intensity += np.exp(-0.5 * ((frequency - peak) / width_cm1) ** 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        np.column_stack([frequency, intensity * scale]),
        header="Frequency(cm^-1)\tSpectral_Density",
        comments="",
    )
    return path
