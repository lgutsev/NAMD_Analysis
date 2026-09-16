"""Readers for CA-NAC / Hefei-NAMD run directories."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .namelist import basis_size, read_namelist
from .tables import TableFormatError, clean_numeric_line, read_numeric_table

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
    """Read a SHPROP table while preserving its Hefei-NAMD header metadata.

    Materializes the whole table.  An archived history can be hundreds of
    megabytes, so frame-aware analysis uses :func:`shprop_structure` and
    :func:`iter_shprop_chunks` instead; this remains for the fixed-column
    commands and for tests that want the table in one piece.
    """
    path = Path(path)
    return ShpropData(path=path, table=read_shprop(path), metadata=read_table_metadata(path))


#: Rows per chunk when nothing else is specified.  At eight columns this is a
#: few megabytes of float64, which is small next to one PROCAR and large
#: enough that per-chunk overhead does not dominate.
DEFAULT_CHUNK_ROWS = 100_000


@dataclass
class ShpropStructure:
    """What a SHPROP file is, without its numbers.

    Everything here is obtained in a single streaming pass that never holds
    more than one line, so the cost is independent of file size in memory and
    linear in it on disk.  This is what preflight reads: it is enough to plan
    the frame alignment and to judge a candidate column layout, and it is not
    enough to compute a population.
    """

    path: Path
    metadata: Dict[str, Any]
    n_rows: int
    n_columns: int
    first_row: np.ndarray
    last_row: np.ndarray
    bytes_on_disk: int

    def time_span(self, time_column: int = 0) -> Dict[str, Any]:
        return {
            "first": float(self.first_row[time_column]),
            "last": float(self.last_row[time_column]),
        }


def _numeric_fields(raw: str) -> Optional[List[str]]:
    """Split one line into numeric fields, or ``None`` if it carries no data.

    Comment stripping and Fortran ``D`` exponents are handled exactly as in
    :func:`~namd_analysis.io.tables.read_numeric_table`; the two must agree or
    streaming and whole-file reads could disagree about a file's contents.
    """
    text = clean_numeric_line(raw)
    return text.split() if text else None


def shprop_structure(path) -> ShpropStructure:
    """Scan a SHPROP file for its shape and endpoints without building a table.

    Counts rows, fixes the column count from the first numeric row, and keeps
    only the first and last rows.  Ragged rows are still an error here, because
    a file whose width changes partway cannot be chunked either; what is *not*
    done is validating every value, which is the full reader's job.
    """
    path = Path(path)
    metadata = read_table_metadata(path)
    width: Optional[int] = None
    n_rows = 0
    first: Optional[np.ndarray] = None
    last_fields: Optional[List[str]] = None

    def _row(fields: List[str], lineno: Any) -> np.ndarray:
        try:
            return np.asarray([float(field) for field in fields], dtype=float)
        except ValueError as exc:
            raise TableFormatError(f"{path}:{lineno}: non-numeric field ({exc})") from exc

    with path.open("r", errors="replace") as handle:
        for lineno, raw in enumerate(handle, start=1):
            fields = _numeric_fields(raw)
            if fields is None:
                continue
            if width is None:
                width = len(fields)
            elif len(fields) != width:
                raise TableFormatError(
                    f"{path}:{lineno}: {len(fields)} columns, expected {width}"
                )
            if first is None:
                first = _row(fields, lineno)
            n_rows += 1
            last_fields = fields
            last_lineno = lineno
    if width is None or first is None or last_fields is None:
        raise TableFormatError(f"{path}: no numeric rows found")
    last = _row(last_fields, last_lineno)
    return ShpropStructure(
        path=path,
        metadata=metadata,
        n_rows=n_rows,
        n_columns=width,
        first_row=first,
        last_row=last,
        bytes_on_disk=path.stat().st_size,
    )


def iter_shprop_chunks(path, chunk_rows: int = DEFAULT_CHUNK_ROWS):
    """Yield ``(row_offset, chunk)`` over a SHPROP table, bounded in memory.

    ``chunk`` is a ``(rows, ncols)`` float array holding at most ``chunk_rows``
    rows.  The intermediate Python buffer is a flat list of floats rather than
    a list of per-row lists: for a file with millions of rows, the per-list
    object overhead of the ordinary reader is what exhausts memory long before
    the resulting array would.

    Validation matches :func:`~namd_analysis.io.tables.read_numeric_table`
    exactly -- ragged rows and non-finite values are errors, never truncation
    or repair -- but is applied chunk by chunk.
    """
    path = Path(path)
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be at least 1")
    width: Optional[int] = None
    buffer: List[float] = []
    rows_in_buffer = 0
    offset = 0
    any_row = False

    def _flush(offset: int, buffer: List[float], rows: int, width: int):
        array = np.asarray(buffer, dtype=float).reshape(rows, width)
        if not np.all(np.isfinite(array)):
            bad = int(np.count_nonzero(~np.isfinite(array)))
            raise TableFormatError(
                f"{path}: {bad} non-finite value(s) in rows "
                f"{offset}..{offset + rows - 1}"
            )
        return array

    with path.open("r", errors="replace") as handle:
        for lineno, raw in enumerate(handle, start=1):
            fields = _numeric_fields(raw)
            if fields is None:
                continue
            if width is None:
                width = len(fields)
            elif len(fields) != width:
                raise TableFormatError(
                    f"{path}:{lineno}: {len(fields)} columns, expected {width}"
                )
            try:
                buffer.extend(float(field) for field in fields)
            except ValueError as exc:
                raise TableFormatError(
                    f"{path}:{lineno}: non-numeric field ({exc}); "
                    "complex or packed formats are not supported"
                ) from exc
            rows_in_buffer += 1
            any_row = True
            if rows_in_buffer == chunk_rows:
                yield offset, _flush(offset, buffer, rows_in_buffer, width)
                offset += rows_in_buffer
                buffer = []
                rows_in_buffer = 0
    if rows_in_buffer:
        yield offset, _flush(offset, buffer, rows_in_buffer, width)
    elif not any_row:
        raise TableFormatError(f"{path}: no numeric rows found")


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
