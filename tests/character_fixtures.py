"""Builders for realistic PROCAR / SHPROP fixtures.

The PROCAR builder deliberately reproduces the dialects VASP actually writes,
because the risk this package has to defend against is not a malformed file --
that fails loudly -- but a *valid* file in a dialect the reader misinterprets
into plausible numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

#: Orbital column layouts VASP writes for different LORBIT settings.
ORBITALS = {
    "spd": ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2", "tot"],
    "sp": ["s", "py", "pz", "px", "tot"],
    "s": ["s", "tot"],
}


def _ion_table(
    ion_totals: Sequence[float],
    orbitals: Sequence[str],
    scientific: bool = False,
    pad: str = "   ",
    include_tot_row: bool = True,
    decimals: int = 6,
) -> str:
    """One ionic projection table: header, one row per ion, then a tot row.

    ``decimals=3`` reproduces VASP's F7.3 printing: each value is rounded on
    its own, while the tot row carries the sum of the *unrounded* values, as
    VASP accumulates it before printing.
    """
    columns = list(orbitals)
    header = f"{pad}ion" + "".join(f"{pad}{name}" for name in columns) + "\n"
    lines = [header]
    for index, total in enumerate(ion_totals, start=1):
        # Spread the total across the non-tot columns so 'tot' is the sum.
        share = total / max(1, len(columns) - 1)
        values = [share] * (len(columns) - 1) + [total]
        if scientific:
            rendered = "".join(f"{pad}{value:.6E}" for value in values)
        else:
            rendered = "".join(f"{pad}{value:.{decimals}f}" for value in values)
        lines.append(f"{pad}{index}{rendered}\n")
    if include_tot_row:
        grand = float(np.sum(ion_totals))
        share = grand / max(1, len(columns) - 1)
        values = [share] * (len(columns) - 1) + [grand]
        rendered = "".join(f"{pad}{value:.{decimals}f}" for value in values)
        lines.append(f"{pad}tot{rendered}\n")
    return "".join(lines)


def write_procar(
    path,
    band_ion_totals: Dict[int, Sequence[float]],
    orbitals: str = "s",
    nkpoints: int = 1,
    spin_blocks: int = 1,
    tables_per_band: int = 1,
    scientific: bool = False,
    pad: str = "   ",
    newline: str = "\n",
    bom: bool = False,
    declared_bands: Optional[int] = None,
    declared_ions: Optional[int] = None,
    truncate_after: Optional[int] = None,
    duplicate_ion: bool = False,
    drop_tot_row: bool = False,
    decimals: int = 6,
) -> Path:
    """Write a PROCAR in a chosen dialect.

    ``tables_per_band`` > 1 emulates LORBIT=12 (charge then phase) or SOC
    (total, mx, my, mz): only the first table is the scalar charge projection.
    """
    path = Path(path)
    columns = ORBITALS[orbitals]
    nions = len(next(iter(band_ion_totals.values())))
    bands = sorted(band_ion_totals)

    chunks: List[str] = ["PROCAR lm decomposed\n"]
    chunks.append(
        f"# of k-points:  {nkpoints}         # of bands:  "
        f"{declared_bands if declared_bands is not None else len(bands)}"
        f"         # of ions:  {declared_ions if declared_ions is not None else nions}\n\n"
    )
    for spin in range(1, spin_blocks + 1):
        if spin_blocks > 1:
            chunks.append(f" spin component {spin}\n")
        for kpoint in range(1, nkpoints + 1):
            chunks.append(
                f" k-point {kpoint} :    0.00000000 0.00000000 0.00000000     weight = 1.00000000\n\n"
            )
            for band in bands:
                chunks.append(f" band {band} # energy   -5.00000000 # occ.  2.00000000\n\n")
                totals = list(band_ion_totals[band])
                for table in range(tables_per_band):
                    # Later tables are phase/magnetisation components: different
                    # numbers, and reading them instead of the first would be a
                    # silent scientific error.
                    values = totals if table == 0 else [-v / 2.0 for v in totals]
                    body = _ion_table(
                        values,
                        columns,
                        scientific=scientific,
                        pad=pad,
                        include_tot_row=not drop_tot_row,
                        decimals=decimals,
                    )
                    if duplicate_ion and table == 0:
                        rows = body.splitlines(keepends=True)
                        rows.insert(2, rows[1])
                        body = "".join(rows)
                    chunks.append(body)
                    chunks.append("\n")

    text = "".join(chunks)
    if truncate_after is not None:
        text = "\n".join(text.splitlines()[:truncate_after]) + "\n"
    if newline != "\n":
        text = text.replace("\n", newline)
    data = text.encode("utf-8")
    if bom:
        data = b"\xef\xbb\xbf" + data
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def write_shprop(
    path,
    namdtini: int,
    populations: np.ndarray,
    bmin: int = 10,
    bmax: int = 11,
    nsw: Optional[int] = None,
    dt_fs: float = 1000.0,
) -> Path:
    """A SHPROP history: header metadata, then time, energy and populations."""
    path = Path(path)
    populations = np.asarray(populations, dtype=float)
    ntime = populations.shape[0]
    nsw = nsw if nsw is not None else ntime + 1
    header = (
        f"# BMIN = {bmin}\n"
        f"# BMAX = {bmax}\n"
        f"# NSW = {nsw}\n"
        f"# NAMDTINI = {namdtini}\n"
        "# POTIM = 1.0\n"
    )
    time = np.arange(ntime) * dt_fs
    table = np.column_stack([time, np.full(ntime, -1.5), populations])
    body = "\n".join(" ".join(f"{value:.10f}" for value in row) for row in table) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + body, encoding="utf-8")
    return path


def write_manifest(
    path, frames: Dict[int, Path], cycle_length: Optional[int] = None
) -> Path:
    """An explicit frame -> PROCAR projection manifest."""
    path = Path(path)
    payload: Dict[str, object] = {
        "frames": [
            {"frame": int(frame), "procar": str(Path(procar).resolve())}
            for frame, procar in sorted(frames.items())
        ]
    }
    if cycle_length is not None:
        payload["cycle_length"] = int(cycle_length)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_atom_groups(
    path,
    groups: Dict[str, Sequence[int]],
    complete_atoms: bool = True,
    min_projection_weight: float = 0.5,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "groups": {name: list(atoms) for name, atoms in groups.items()},
                "complete_atoms": complete_atoms,
                "min_projection_weight": min_projection_weight,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def write_state_map(path, bmin: int = 10, bmax: int = 11, groups=None) -> Path:
    """State map with one population column per basis state, columns 2..N+1."""
    path = Path(path)
    nstates = bmax - bmin + 1
    columns = list(range(2, 2 + nstates))
    if groups is None:
        groups = {f"band{bmin + i}": [columns[i]] for i in range(nstates)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "name": "character_fixture",
                "time_column": 0,
                "time_unit": "fs",
                "population_columns": columns,
                "groups": {k: list(v) for k, v in groups.items()},
                "complete_population": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


#: A PROCAR whose content would raise if it were ever parsed.  Used to prove
#: that frames outside the required set are never opened for projections.
POISON = "this is not a PROCAR and parsing it must fail\n"


def write_poison_procar(path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(POISON, encoding="utf-8")
    return path


def swap_campaign(root: Path, ntime: int = 4, cycle_length: int = 4):
    """A two-history campaign with a deliberate BCF <-> PCBM character swap.

    Bands 10 and 11; ions 1-2 are BCF, ions 3-4 are PCBM.  Frames 1 and 2 give
    band 10 BCF character and band 11 PCBM character; frames 3 and 4 swap them.
    Two SHPROP histories start at different NAMDTINI, so they visit the two
    halves of the cycle in opposite order -- which is exactly the situation that
    averaging-before-projecting would get wrong.
    """
    root = Path(root)
    pure_bcf = [1.0, 1.0, 0.0, 0.0]
    pure_pcbm = [0.0, 0.0, 1.0, 1.0]
    frames: Dict[int, Path] = {}
    for frame in (1, 2):
        frames[frame] = write_procar(
            root / f"PROCAR.{frame}", {10: pure_bcf, 11: pure_pcbm}
        )
    for frame in (3, 4):
        frames[frame] = write_procar(
            root / f"PROCAR.{frame}", {10: pure_pcbm, 11: pure_bcf}
        )

    # Two histories with different populations, so averaging first would differ.
    pops_a = np.zeros((ntime, 2))
    pops_a[:, 0] = np.linspace(0.9, 0.1, ntime)
    pops_a[:, 1] = 1.0 - pops_a[:, 0]
    pops_b = np.zeros((ntime, 2))
    pops_b[:, 0] = np.linspace(0.2, 0.8, ntime)
    pops_b[:, 1] = 1.0 - pops_b[:, 0]

    shprop_a = write_shprop(root / "SHPROP.1", 1, pops_a, nsw=cycle_length + 1)
    shprop_b = write_shprop(root / "SHPROP.3", 3, pops_b, nsw=cycle_length + 1)

    manifest = write_manifest(root / "projection.json", frames, cycle_length=cycle_length)
    atom_groups = write_atom_groups(
        root / "atoms.json", {"BCF": [1, 2], "PCBM": [3, 4]}
    )
    state_map = write_state_map(root / "state_map.json")
    return {
        "shprop": [shprop_a, shprop_b],
        "populations": {"SHPROP.1": pops_a, "SHPROP.3": pops_b},
        "manifest": manifest,
        "atom_groups": atom_groups,
        "state_map": state_map,
        "frames": frames,
        "cycle_length": cycle_length,
    }


def expected_swap_populations(campaign) -> np.ndarray:
    """Hand-computed projection-weighted ensemble mean for ``swap_campaign``.

    Frames 1,2 -> band10 is BCF, band11 is PCBM.  Frames 3,4 -> reversed.
    History 1 starts at NAMDTINI=1 and visits frames 1,2,3,4.
    History 3 starts at NAMDTINI=3 and visits frames 3,4,1,2.
    """
    pops_a = campaign["populations"]["SHPROP.1"]
    pops_b = campaign["populations"]["SHPROP.3"]
    ntime = pops_a.shape[0]
    period = campaign["cycle_length"]

    def frames_for(start):
        tion = np.arange(1, ntime + 1)
        frames = np.mod(tion + start - 1, period)
        frames[frames == 0] = period
        return frames

    def project(pops, start):
        out = np.zeros((ntime, 2))
        for t, frame in enumerate(frames_for(start)):
            if frame in (1, 2):
                out[t, 0] = pops[t, 0]  # BCF from band 10
                out[t, 1] = pops[t, 1]  # PCBM from band 11
            else:
                out[t, 0] = pops[t, 1]
                out[t, 1] = pops[t, 0]
        return out

    return 0.5 * (project(pops_a, 1) + project(pops_b, 3))
