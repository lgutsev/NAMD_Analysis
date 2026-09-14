"""Readers for CA-NAC / Hefei-NAMD run directories."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .namelist import basis_size, read_namelist
from .tables import TableFormatError, read_numeric_table

#: Files a run directory may contain; presence is reported, never assumed.
KNOWN_FILES = (
    "inp",
    "EIGTXT",
    "NATXT",
    "INICON",
    "DEPHTIME",
    "energy.dat",
    "fitting_results_final.txt",
)

_HEADER = re.compile(r"^\s*[#!]\s*([^=]+?)\s*=\s*(.*?)\s*$")


def _header_value(text: str) -> Any:
    value = text.strip()
    upper = value.upper()
    if upper in {"T", ".TRUE.", "TRUE"}:
        return True
    if upper in {"F", ".FALSE.", "FALSE"}:
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value.replace("D", "E").replace("d", "e"))
    except ValueError:
        return value


def read_table_metadata(path) -> Dict[str, Any]:
    """Read ``# KEY = VALUE`` / ``! KEY = VALUE`` metadata without touching data.

    Hefei-NAMD writes the sampling origin (``NAMDTINI``), basis window and
    timestep into SHPROP headers.  The ordinary numeric-table reader strips
    those comments deliberately, so frame-aware analyses use this companion
    reader rather than trying to recover the information from the populations.
    Duplicate keys must agree exactly.
    """
    path = Path(path)
    metadata: Dict[str, Any] = {}
    with path.open("r", errors="replace") as handle:
        for raw in handle:
            stripped = raw.lstrip()
            if not stripped.startswith(("#", "!")):
                if stripped.strip():
                    break
                continue
            match = _HEADER.match(raw)
            if match is None:
                continue
            key = match.group(1).strip().upper()
            value = _header_value(match.group(2))
            if key in metadata and metadata[key] != value:
                raise TableFormatError(
                    f"{path}: metadata key {key!r} occurs with conflicting values"
                )
            metadata[key] = value
    return metadata


@dataclass
class ShpropData:
    """A SHPROP numeric table together with the metadata needed for alignment."""

    path: Path
    table: np.ndarray
    metadata: Dict[str, Any]


def read_eigtxt(path) -> np.ndarray:
    """State energies, shape ``(nframes, nstates)``, in the file's own unit (eV)."""
    return read_numeric_table(path)


def read_natxt(path, nstates: Optional[int] = None) -> np.ndarray:
    """Real NAC matrices, shape ``(nframes, nstates, nstates)``.

    Each row holds one flattened N x N matrix.  Complex or packed triangular
    layouts are rejected rather than guessed at.
    """
    flat = read_numeric_table(path)
    ncols = flat.shape[1]
    side = int(round(np.sqrt(ncols)))
    if side * side != ncols:
        raise TableFormatError(
            f"{path}: {ncols} columns is not a perfect square; "
            "packed or complex NAC layouts are not supported"
        )
    if nstates is not None and side != nstates:
        raise TableFormatError(
            f"{path}: NAC implies {side} states, EIGTXT has {nstates}"
        )
    return flat.reshape(flat.shape[0], side, side)


def read_inicon(path) -> np.ndarray:
    """Initial conditions, shape ``(nsample, 2)`` = (start frame, start band)."""
    return read_numeric_table(path)


def read_dephtime(path) -> np.ndarray:
    """Pure-dephasing times, shape ``(nstates, nstates)``, in the file's unit (fs)."""
    table = read_numeric_table(path)
    if table.shape[0] != table.shape[1]:
        raise TableFormatError(f"{path}: DEPHTIME is not square {table.shape}")
    return table


def read_shprop(path) -> np.ndarray:
    """One SHPROP file as a 2-D array; columns are interpreted by configuration.

    Hefei-NAMD writes time in column 0 and an energy in column 1, followed by
    one population column per basis state, but this reader does not enforce
    that: column meaning is declared by the user's JSON configuration.
    """
    return read_numeric_table(path)


def read_shprop_with_metadata(path) -> ShpropData:
    """Read a SHPROP table while preserving its Hefei-NAMD header metadata."""
    path = Path(path)
    return ShpropData(path=path, table=read_shprop(path), metadata=read_table_metadata(path))


def read_legacy_fit(path) -> Dict[str, Any]:
    """Parse ``fitting_results_final.txt`` written by the legacy plotting script."""
    result: Dict[str, Any] = {"tau_ns": None, "r_squared": None}
    with Path(path).open("r", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("Fitted parameter"):
                result["tau_ns"] = float(line.split(":")[1].split()[0])
            elif line.startswith("R"):
                try:
                    result["r_squared"] = float(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    pass
            if line.startswith("Time"):
                break
    return result


@dataclass
class RunDirectory:
    """A single CA-NAC / Hefei-NAMD run directory."""

    path: Path
    present: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    legacy_fit: Optional[Dict[str, Any]] = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def declared_nstates(self) -> Optional[int]:
        return basis_size(self.params)

    @property
    def dt_fs_from_inp(self) -> Optional[float]:
        potim = self.params.get("POTIM")
        return float(potim) if isinstance(potim, (int, float)) else None


def load_run_directory(path) -> RunDirectory:
    """Inspect a run directory without reading the large tables."""
    path = Path(path)
    run = RunDirectory(path=path)
    run.present = [name for name in KNOWN_FILES if (path / name).is_file()]
    if "inp" in run.present:
        run.params = read_namelist(path / "inp")
    if "fitting_results_final.txt" in run.present:
        try:
            run.legacy_fit = read_legacy_fit(path / "fitting_results_final.txt")
        except (OSError, ValueError):
            run.legacy_fit = None
    return run


def looks_like_run_directory(path) -> bool:
    """True when a directory holds the minimum evidence of a NAMD run."""
    path = Path(path)
    return (path / "EIGTXT").is_file() or (
        (path / "inp").is_file() and (path / "NATXT").is_file()
    )
