"""Readers for the full-space validation: VASP volumetric files and Bader output.

``read_volumetric`` reads the first density block of a CHGCAR-format file
(CHGCAR, PARCHG, AECCAR*).  VASP stores ``rho * V_cell`` on the grid, x
fastest, so the integral over a region is the sum of its grid values divided
by the number of grid points.  Only the first block is read: a non-spin
PARCHG has one, and the character analysis refuses spin-polarized input.

``read_bader_acf`` reads the ``ACF.dat`` of Henkelman's ``bader`` program.
Run as ``bader PARCHG -ref CHGCAR_sum``, its ``CHARGE`` column is the band
density integrated over each atom's Bader basin of the reference density,
and ``VACUUM CHARGE`` is the part assigned to no atom.  Both are kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .xdatcar import _apply_scale


class VolumetricFormatError(ValueError):
    """Raised when a volumetric or Bader file cannot be read without guessing."""


@dataclass
class Volumetric:
    """The first density block of a CHGCAR-format file, and its structure."""

    path: Path
    lattice: np.ndarray  # (3, 3) Angstrom, rows are lattice vectors
    species: List[str]
    counts: List[int]
    fractional: np.ndarray  # (natoms, 3)
    data: np.ndarray  # (nx, ny, nz), rho * V_cell as stored

    @property
    def natoms(self) -> int:
        return int(sum(self.counts))

    @property
    def grid(self) -> tuple:
        return tuple(int(n) for n in self.data.shape)

    def integral(self) -> float:
        """The integrated density, in the file's electron units."""
        return float(self.data.sum() / self.data.size)


def _floats(text: str, what: str, path: Path, lineno: int) -> List[float]:
    try:
        return [float(item) for item in text.split()]
    except ValueError:
        raise VolumetricFormatError(f"{path}: line {lineno}: cannot read {what} from {text.strip()!r}") from None


#: Characters read per block while streaming the grid.
_BLOCK = 1 << 24


def read_volumetric(path) -> Volumetric:
    path = Path(path)
    with path.open("r", errors="replace") as handle:

        def take(what: str) -> str:
            line = handle.readline()
            if not line:
                raise VolumetricFormatError(f"{path}: file ends while reading {what}")
            return line

        take("the comment line")
        scale = _floats(take("the scale factor"), "the scale factor", path, 2)
        if len(scale) != 1:
            raise VolumetricFormatError(f"{path}: line 2 must hold one scale factor")
        cell = np.array(
            [_floats(take("a lattice vector"), "a lattice vector", path, 3 + i) for i in range(3)],
            dtype=float,
        )
        if cell.shape != (3, 3):
            raise VolumetricFormatError(f"{path}: lines 3-5 must hold three 3-vectors")
        lattice = _apply_scale(cell, scale[0], path, 2)
        line = take("species or counts")
        tokens = line.split()
        species: List[str] = []
        if tokens and not tokens[0].lstrip("-").isdigit():
            species = tokens
            line = take("atom counts")
            tokens = line.split()
        try:
            counts = [int(item) for item in tokens]
        except ValueError:
            raise VolumetricFormatError(f"{path}: cannot read atom counts from {line.strip()!r}") from None
        if species and len(species) != len(counts):
            raise VolumetricFormatError(
                f"{path}: {len(species)} species names but {len(counts)} counts"
            )
        mode = take("the coordinate mode").strip()
        if mode[:1].lower() == "s":
            mode = take("the coordinate mode").strip()
        natoms = sum(counts)
        positions = np.array(
            [_floats(take("an atom position"), "an atom position", path, 0)[:3] for _ in range(natoms)],
            dtype=float,
        )
        if mode[:1].lower() in ("c", "k"):
            fractional = positions @ np.linalg.inv(lattice)
        elif mode[:1].lower() == "d":
            fractional = positions
        else:
            raise VolumetricFormatError(f"{path}: unknown coordinate mode {mode!r}")
        grid_line = handle.readline()
        while grid_line and not grid_line.strip():
            grid_line = handle.readline()
        try:
            nx, ny, nz = (int(item) for item in grid_line.split())
        except ValueError:
            raise VolumetricFormatError(
                f"{path}: expected the grid dimensions NGX NGY NGZ, got {grid_line.strip()!r}"
            ) from None
        needed = nx * ny * nz
        values = np.empty(needed, dtype=float)
        filled = 0
        carry = ""
        # Stream in blocks, splitting only at line ends; stop at the declared
        # count, so whatever follows the first grid (augmentation occupancies,
        # a second spin block) is never parsed.
        while filled < needed:
            block = handle.read(_BLOCK)
            if block:
                text = carry + block
                cut = text.rfind("\n")
                if cut < 0:
                    carry = text
                    continue
                chunk, carry = text[:cut], text[cut:]
            else:
                chunk, carry = carry, ""
            tokens = chunk.split()
            take_n = min(len(tokens), needed - filled)
            if take_n:
                try:
                    values[filled : filled + take_n] = np.asarray(tokens[:take_n], dtype=float)
                except ValueError:
                    raise VolumetricFormatError(
                        f"{path}: a non-numeric value inside the density grid after "
                        f"{filled} of {needed} values"
                    ) from None
                filled += take_n
            if not block:
                break
        if filled < needed:
            raise VolumetricFormatError(
                f"{path}: the density grid holds {filled} values but {nx}x{ny}x{nz} = "
                f"{needed} were declared; the file is truncated"
            )
    data = values.reshape((nz, ny, nx)).transpose(2, 1, 0)
    return Volumetric(
        path=path, lattice=lattice, species=species, counts=counts,
        fractional=fractional % 1.0, data=data,
    )


@dataclass
class BaderCharges:
    path: Path
    charges: np.ndarray  # per atom, in ACF order (1-based atom index = position + 1)
    vacuum_charge: float
    vacuum_volume: Optional[float]
    total_reported: Optional[float]

    @property
    def total(self) -> float:
        """Atom basins plus vacuum: everything the file assigns."""
        return float(np.sum(self.charges) + self.vacuum_charge)


_FOOTER = re.compile(r"^\s*(VACUUM CHARGE|VACUUM VOLUME|NUMBER OF ELECTRONS)\s*:\s*(\S+)", re.I)


def read_bader_acf(path, natoms: Optional[int] = None) -> BaderCharges:
    """Per-atom basin integrals and the vacuum remainder from ``ACF.dat``."""
    path = Path(path)
    charges: List[float] = []
    footer = {}
    header_seen = False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise VolumetricFormatError(f"cannot read {path}: {exc}") from exc
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or set(stripped) <= {"-"}:
            continue
        if stripped.startswith("#"):
            upper = stripped.upper()
            if "CHARGE" not in upper:
                raise VolumetricFormatError(f"{path}: line {lineno}: header without a CHARGE column")
            header_seen = True
            continue
        match = _FOOTER.match(line)
        if match:
            footer[match.group(1).upper()] = float(match.group(2))
            continue
        fields = stripped.split()
        try:
            index = int(fields[0])
            charge = float(fields[4])
        except (ValueError, IndexError):
            raise VolumetricFormatError(
                f"{path}: line {lineno}: expected '# X Y Z CHARGE ...', got {stripped!r}"
            ) from None
        if index != len(charges) + 1:
            raise VolumetricFormatError(
                f"{path}: line {lineno}: atom {index} out of order (expected {len(charges) + 1})"
            )
        charges.append(charge)
    if not header_seen or not charges:
        raise VolumetricFormatError(f"{path}: no Bader atom table found")
    if "VACUUM CHARGE" not in footer:
        raise VolumetricFormatError(
            f"{path}: no VACUUM CHARGE line; the file is truncated or not an ACF.dat"
        )
    if natoms is not None and len(charges) != natoms:
        raise VolumetricFormatError(
            f"{path}: {len(charges)} atoms, but the atom partition covers {natoms}"
        )
    return BaderCharges(
        path=path,
        charges=np.asarray(charges, dtype=float),
        vacuum_charge=footer["VACUUM CHARGE"],
        vacuum_volume=footer.get("VACUUM VOLUME"),
        total_reported=footer.get("NUMBER OF ELECTRONS"),
    )
