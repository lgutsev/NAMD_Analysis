"""XDATCAR reader, including the repeated-header form written under NPT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


class XdatcarFormatError(ValueError):
    """Raised when an XDATCAR cannot be read without guessing."""


@dataclass
class Trajectory:
    """Frames of a VASP trajectory.

    ``positions`` are fractional when ``direct`` is True.  ``lattices`` holds
    one scaled 3x3 cell per frame, so a variable-cell run is preserved rather
    than collapsed onto the first cell.
    """

    positions: np.ndarray  # (nframes, natoms, 3)
    lattices: np.ndarray  # (nframes, 3, 3) in Angstrom
    species: List[str]
    counts: List[int]
    direct: bool
    comment: str
    path: Optional[Path] = None

    @property
    def nframes(self) -> int:
        return int(self.positions.shape[0])

    @property
    def natoms(self) -> int:
        return int(self.positions.shape[1])

    @property
    def variable_cell(self) -> bool:
        return not bool(np.allclose(self.lattices, self.lattices[0]))

    def atom_symbols(self) -> List[str]:
        symbols: List[str] = []
        for symbol, count in zip(self.species, self.counts):
            symbols.extend([symbol] * count)
        return symbols


def _floats(line: str) -> Optional[List[float]]:
    try:
        return [float(token) for token in line.split()]
    except ValueError:
        return None


def _apply_scale(cell: np.ndarray, scale: float, path, lineno: int) -> np.ndarray:
    """Apply the VASP scaling factor on line 2 of a POSCAR-style header.

    A positive value multiplies the lattice vectors.  A **negative** value is
    VASP's other convention: its magnitude is the target cell volume in cubic
    Angstrom, and the vectors are scaled uniformly to reach it.  Multiplying by
    a negative scale instead -- which is what happens if the convention is
    ignored -- flips the cell and changes its volume by an arbitrary factor,
    silently rescaling every velocity and spectral intensity derived from it.
    """
    if scale > 0:
        return scale * cell
    volume = abs(float(np.linalg.det(cell)))
    if scale == 0.0 or volume == 0.0:
        raise XdatcarFormatError(
            f"{path}: line {lineno} gives scale {scale} with cell volume {volume}; "
            "the lattice is degenerate"
        )
    return cell * (abs(scale) / volume) ** (1.0 / 3.0)


def _is_header_start(lines: List[str], index: int) -> bool:
    """True when a full 7-line POSCAR-style header begins at ``index``."""
    if index + 6 >= len(lines):
        return False
    scale = _floats(lines[index + 1])
    if scale is None or len(scale) != 1:
        return False
    for offset in (2, 3, 4):
        vector = _floats(lines[index + offset])
        if vector is None or len(vector) != 3:
            return False
    counts = _floats(lines[index + 6])
    return counts is not None and all(float(c).is_integer() for c in counts)


def read_xdatcar(path) -> Trajectory:
    """Read an XDATCAR into a :class:`Trajectory`."""
    path = Path(path)
    with path.open("r", errors="replace") as handle:
        lines = [line.rstrip("\n") for line in handle]
    if len(lines) < 8:
        raise XdatcarFormatError(f"{path}: file is too short to be an XDATCAR")

    comment = lines[0].strip()
    if not _is_header_start(lines, 0):
        raise XdatcarFormatError(f"{path}: no POSCAR-style header at the top of the file")

    def parse_header(index: int):
        scale = float(lines[index + 1].split()[0])
        cell = np.array(
            [[float(v) for v in lines[index + offset].split()] for offset in (2, 3, 4)],
            dtype=float,
        )
        cell = _apply_scale(cell, scale, path, index + 2)
        species_line = lines[index + 5].split()
        counts_line = [int(float(v)) for v in lines[index + 6].split()]
        if all(_floats(token) is not None for token in species_line):
            raise XdatcarFormatError(
                f"{path}: line {index + 6} has no element symbols; VASP 4 style "
                "XDATCAR without a species line is not supported"
            )
        return cell, species_line, counts_line

    lattice, species, counts = parse_header(0)
    natoms = sum(counts)
    if natoms <= 0:
        raise XdatcarFormatError(f"{path}: atom counts sum to {natoms}")

    frames: List[np.ndarray] = []
    lattices: List[np.ndarray] = []
    direct: Optional[bool] = None

    index = 7
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        lowered = line.lower()
        if lowered.startswith("direct") or lowered.startswith("cartesian"):
            frame_direct = lowered.startswith("direct")
            if direct is None:
                direct = frame_direct
            elif direct != frame_direct:
                raise XdatcarFormatError(
                    f"{path}: frame at line {index + 1} switches between Direct and "
                    "Cartesian coordinates"
                )
            block = []
            for offset in range(1, natoms + 1):
                if index + offset >= len(lines):
                    raise XdatcarFormatError(
                        f"{path}: truncated frame starting at line {index + 1}"
                    )
                values = _floats(lines[index + offset])
                if values is None or len(values) < 3:
                    raise XdatcarFormatError(
                        f"{path}: line {index + offset + 1} is not a coordinate triple"
                    )
                block.append(values[:3])
            frames.append(np.asarray(block, dtype=float))
            lattices.append(lattice)
            index += natoms + 1
            continue
        if _is_header_start(lines, index):
            lattice, species_next, counts_next = parse_header(index)
            if counts_next != counts:
                raise XdatcarFormatError(
                    f"{path}: atom counts change at line {index + 1}"
                )
            species = species_next
            index += 7
            continue
        index += 1

    if not frames:
        raise XdatcarFormatError(f"{path}: no coordinate frames found")

    return Trajectory(
        positions=np.asarray(frames),
        lattices=np.asarray(lattices),
        species=species,
        counts=counts,
        direct=bool(direct),
        comment=comment,
        path=path,
    )
