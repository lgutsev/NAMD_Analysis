"""Tolerant readers for the headerless numeric tables used by Hefei-NAMD."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_D_EXPONENT = re.compile(r"(?<=[0-9.])[dD](?=[-+]?[0-9])")


class TableFormatError(ValueError):
    """Raised when a numeric table cannot be read without guessing."""


def clean_numeric_line(line: str) -> str:
    """Strip comments and normalize Fortran ``D`` exponents on one line.

    Shared with the streaming SHPROP reader so that a chunked read and a
    whole-file read can never disagree about what a line contains.
    """
    for marker in ("#", "!"):
        idx = line.find(marker)
        if idx >= 0:
            line = line[:idx]
    return _D_EXPONENT.sub("E", line).strip()


#: Historical private spelling, kept so existing call sites read unchanged.
_clean = clean_numeric_line


def read_numeric_table(path, expect_columns: Optional[int] = None) -> np.ndarray:
    """Read a whitespace-separated numeric table into a 2-D float array.

    Blank lines, ``#``/``!`` comments and Fortran ``D`` exponents are handled.
    Ragged rows are an error, never a silent truncation.
    """
    path = Path(path)
    rows: List[List[float]] = []
    width: Optional[int] = None
    with path.open("r", errors="replace") as handle:
        for lineno, raw in enumerate(handle, start=1):
            text = _clean(raw)
            if not text:
                continue
            fields = text.split()
            try:
                values = [float(field) for field in fields]
            except ValueError as exc:
                raise TableFormatError(
                    f"{path}:{lineno}: non-numeric field ({exc}); "
                    "complex or packed formats are not supported"
                ) from exc
            if width is None:
                width = len(values)
            elif len(values) != width:
                raise TableFormatError(
                    f"{path}:{lineno}: {len(values)} columns, expected {width}"
                )
            rows.append(values)
    if not rows:
        raise TableFormatError(f"{path}: no numeric rows found")
    array = np.asarray(rows, dtype=float)
    if expect_columns is not None and array.shape[1] != expect_columns:
        raise TableFormatError(
            f"{path}: {array.shape[1]} columns, expected {expect_columns}"
        )
    if not np.all(np.isfinite(array)):
        bad = int(np.count_nonzero(~np.isfinite(array)))
        raise TableFormatError(f"{path}: {bad} non-finite value(s)")
    return array


def read_xy_with_header(path) -> Tuple[np.ndarray, Optional[str]]:
    """Read a two-column table that may carry one non-numeric header line.

    Returns ``(array, header_text_or_None)``.  Used for the
    ``spectral_density_*.txt`` files, whose first line is a column legend.
    """
    path = Path(path)
    header = None
    with path.open("r", errors="replace") as handle:
        lines = handle.readlines()
    for index, raw in enumerate(lines):
        text = _clean(raw)
        if not text:
            continue
        try:
            [float(field) for field in text.split()]
        except ValueError:
            header = text
            lines = lines[index + 1 :]
        break
    rows = []
    width = None
    for lineno, raw in enumerate(lines, start=1):
        text = _clean(raw)
        if not text:
            continue
        values = [float(field) for field in text.split()]
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise TableFormatError(f"{path}: ragged row {lineno}")
        rows.append(values)
    if not rows:
        raise TableFormatError(f"{path}: no numeric rows found")
    return np.asarray(rows, dtype=float), header
