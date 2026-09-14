"""Strict reader for the ionic projection weights needed for state character.

The character analysis needs only the per-ion ``tot`` projection for each band.
It intentionally does not try to be a general PROCAR parser: multiple k-points
or multiple spin components are rejected rather than silently mixed.  For SOC
PROCAR files the first ionic table after each band is the scalar charge
projection and is the one read here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


class ProcarFormatError(ValueError):
    """Raised when a PROCAR cannot be interpreted without guessing."""


_COUNTS = re.compile(
    r"#\s*of\s*k-points\s*:\s*(\d+)\s+"
    r"#\s*of\s*bands\s*:\s*(\d+)\s+"
    r"#\s*of\s*ions\s*:\s*(\d+)",
    re.IGNORECASE,
)
_BAND = re.compile(r"^\s*band\s+(\d+)\b", re.IGNORECASE)
_SPIN = re.compile(r"^\s*spin\s+component\s+(\d+)\b", re.IGNORECASE)


@dataclass
class ProcarProjection:
    """Per-ion scalar projection weights for every parsed VASP band."""

    path: Path
    bands: np.ndarray
    ion_totals: np.ndarray
    nions: int

    def band_index(self) -> Dict[int, int]:
        return {int(band): i for i, band in enumerate(self.bands)}


def read_procar_ion_totals(path) -> ProcarProjection:
    """Read the per-ion ``tot`` column from a single-k-point PROCAR.

    Returns ``ion_totals[band_index, ion_index]``.  Ion indices in PROCAR are
    one-based; the returned array is naturally zero-based.  The parser rejects
    multiple k-points and multiple spin components because the NAMD basis must
    be aligned to one unambiguous set of adiabatic states.
    """
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()
    counts = None
    for line in lines[:30]:
        match = _COUNTS.search(line)
        if match:
            counts = tuple(int(match.group(i)) for i in range(1, 4))
            break
    if counts is None:
        raise ProcarFormatError(f"{path}: missing PROCAR k-point/band/ion header")
    nkpoints, nbands_declared, nions = counts
    if nkpoints != 1:
        raise ProcarFormatError(
            f"{path}: {nkpoints} k-points found; character analysis requires "
            "one k-point and will not average or select one implicitly"
        )

    spin_components = {
        int(match.group(1))
        for line in lines
        for match in [_SPIN.match(line)]
        if match is not None
    }
    if len(spin_components) > 1:
        raise ProcarFormatError(
            f"{path}: multiple spin components {sorted(spin_components)} found; "
            "selecting or combining them must be explicit upstream"
        )

    bands: List[int] = []
    weights: List[List[float]] = []
    i = 0
    while i < len(lines):
        band_match = _BAND.match(lines[i])
        if band_match is None:
            i += 1
            continue
        band = int(band_match.group(1))

        header_index = None
        for j in range(i + 1, min(i + 20, len(lines))):
            text = lines[j].strip().lower()
            if text.startswith("band ") or text.startswith("k-point"):
                break
            fields = text.split()
            if fields and fields[0] == "ion" and "tot" in fields:
                header_index = j
                break
        if header_index is None:
            raise ProcarFormatError(
                f"{path}: band {band} has no ionic projection header with a tot column"
            )

        fields = lines[header_index].split()
        tot_column = [field.lower() for field in fields].index("tot")
        # Data rows include an ion index before the projection columns, whereas
        # the header's first field is the label ``ion``.  Thus the numeric field
        # position of ``tot`` is the same index in the split row.
        band_weights: List[float] = []
        for ion_offset in range(1, nions + 1):
            row_index = header_index + ion_offset
            if row_index >= len(lines):
                raise ProcarFormatError(f"{path}: band {band} ends inside its ion table")
            row = lines[row_index].split()
            if not row:
                raise ProcarFormatError(f"{path}: blank ion row in band {band}")
            try:
                ion = int(row[0])
            except ValueError as exc:
                raise ProcarFormatError(
                    f"{path}: expected ion row {ion_offset} in band {band}, got {lines[row_index]!r}"
                ) from exc
            if ion != ion_offset:
                raise ProcarFormatError(
                    f"{path}: band {band} ion rows are not 1..{nions} in order"
                )
            if tot_column >= len(row):
                raise ProcarFormatError(
                    f"{path}: band {band} ion {ion} lacks the declared tot column"
                )
            try:
                value = float(row[tot_column].replace("D", "E").replace("d", "e"))
            except ValueError as exc:
                raise ProcarFormatError(
                    f"{path}: non-numeric tot projection in band {band}, ion {ion}"
                ) from exc
            if not np.isfinite(value) or value < -1.0e-12:
                raise ProcarFormatError(
                    f"{path}: invalid tot projection {value!r} in band {band}, ion {ion}"
                )
            band_weights.append(max(0.0, value))

        bands.append(band)
        weights.append(band_weights)
        i = header_index + nions + 1

    if not bands:
        raise ProcarFormatError(f"{path}: no bands with ionic projections found")
    if len(set(bands)) != len(bands):
        raise ProcarFormatError(
            f"{path}: duplicate band numbers found; this usually means multiple "
            "k-points/spins or an unsupported PROCAR layout"
        )
    if len(bands) != nbands_declared:
        raise ProcarFormatError(
            f"{path}: parsed {len(bands)} bands but header declares {nbands_declared}"
        )

    return ProcarProjection(
        path=path,
        bands=np.asarray(bands, dtype=int),
        ion_totals=np.asarray(weights, dtype=float),
        nions=nions,
    )
