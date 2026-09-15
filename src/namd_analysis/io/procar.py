"""Strict reader for the ionic projection weights needed for state character.

The character analysis needs only the per-ion ``tot`` projection for each band
in the NAMD basis window.  It intentionally does not try to be a general PROCAR
parser: multiple k-points or multiple spin components are rejected rather than
silently mixed.  For a PROCAR that writes several ionic tables per band -- SOC
(total, mx, my, mz) or LORBIT=12 (charge then phase) -- the first table after
the band line is the scalar charge projection and is the one read here; the
rest are skipped, because the band scanner only resumes on the next ``band``
line.

The file is read as a stream, one line at a time, and only the requested bands
are retained.  A campaign PROCAR can be hundreds of megabytes and only a
handful of its bands are ever in the NAMD window, so neither the file nor the
unused bands are held in memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

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
_KPOINT = re.compile(r"^\s*k-point\s+(\d+)\b", re.IGNORECASE)

#: How far past a ``band`` line the ionic header may sit before the layout is
#: considered unreadable.
_HEADER_LOOKAHEAD = 20

#: Lines from the top of the file within which the counts header must appear.
_COUNTS_LOOKAHEAD = 30


@dataclass
class ProcarProjection:
    """Per-ion scalar projection weights for the parsed VASP bands."""

    path: Path
    bands: np.ndarray
    ion_totals: np.ndarray
    nions: int
    #: Bands present in the file, whether or not their weights were retained.
    bands_seen: int = 0
    #: Bands declared by the counts header.
    bands_declared: int = 0

    def band_index(self) -> Dict[int, int]:
        return {int(band): i for i, band in enumerate(self.bands)}


def _numeric(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def read_procar_ion_totals(
    path, bands: Optional[Sequence[int]] = None
) -> ProcarProjection:
    """Read the per-ion ``tot`` column from a single-k-point PROCAR.

    ``ion_totals[band_index, ion_index]``; PROCAR ion indices are one-based and
    the returned array is zero-based.  Multiple k-points and multiple spin
    components are rejected: the NAMD basis must map onto one unambiguous set
    of adiabatic states, and averaging or picking one implicitly would change
    the science without saying so.

    ``bands`` restricts which VASP band numbers are retained.  Bands outside it
    are still walked, so the band count and ion-table structure are still
    checked, but their projections are neither parsed nor stored.  Passing the
    NAMD window keeps a large PROCAR from being materialised in memory.
    """
    path = Path(path)
    wanted = None if bands is None else {int(band) for band in bands}
    if wanted is not None and not wanted:
        raise ProcarFormatError(f"{path}: an empty band selection was requested")

    counts = None
    nkpoints = nbands_declared = nions = 0
    spin_components: set = set()
    kept_bands: List[int] = []
    kept_weights: List[List[float]] = []
    seen_bands: List[int] = []

    # Streaming state machine.  ``pending`` holds the band whose ionic header
    # we are still looking for; ``table`` holds the band whose rows we are
    # consuming.
    pending: Optional[int] = None
    pending_distance = 0
    table_band: Optional[int] = None
    tot_column = -1
    rows_read = 0
    row_values: List[float] = []
    keep_current = False

    with path.open("r", errors="replace") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.rstrip("\r\n")

            if counts is None:
                match = _COUNTS.search(line)
                if match:
                    counts = tuple(int(match.group(i)) for i in range(1, 4))
                    nkpoints, nbands_declared, nions = counts
                    if nkpoints != 1:
                        raise ProcarFormatError(
                            f"{path}: {nkpoints} k-points found; character analysis "
                            "requires one k-point and will not average or select one "
                            "implicitly. Re-run the projection with a single k-point."
                        )
                    if nions <= 0 or nbands_declared <= 0:
                        raise ProcarFormatError(
                            f"{path}: header declares {nbands_declared} bands and "
                            f"{nions} ions; both must be positive"
                        )
                    continue
                if lineno > _COUNTS_LOOKAHEAD:
                    raise ProcarFormatError(
                        f"{path}: missing PROCAR k-point/band/ion header in the first "
                        f"{_COUNTS_LOOKAHEAD} lines. This does not look like a PROCAR."
                    )
                continue

            spin_match = _SPIN.match(line)
            if spin_match is not None:
                spin_components.add(int(spin_match.group(1)))
                if len(spin_components) > 1:
                    raise ProcarFormatError(
                        f"{path}: line {lineno}: multiple spin components "
                        f"{sorted(spin_components)} found; character analysis will not "
                        "select or combine them implicitly. Split the PROCAR, or "
                        "supply a non-spin-polarised projection."
                    )
                continue

            if table_band is not None:
                # Consuming the ionic table of ``table_band``.
                fields = line.split()
                if not fields:
                    raise ProcarFormatError(
                        f"{path}: line {lineno}: blank ion row inside band "
                        f"{table_band} (expected ion {rows_read + 1} of {nions})"
                    )
                try:
                    ion = int(fields[0])
                except ValueError as exc:
                    raise ProcarFormatError(
                        f"{path}: line {lineno}: expected ion {rows_read + 1} of "
                        f"{nions} in band {table_band}, got {line.strip()!r}"
                    ) from exc
                if ion != rows_read + 1:
                    raise ProcarFormatError(
                        f"{path}: line {lineno}: band {table_band} ion rows must run "
                        f"1..{nions} in order; expected ion {rows_read + 1}, found {ion}"
                    )
                if keep_current:
                    if tot_column >= len(fields):
                        raise ProcarFormatError(
                            f"{path}: line {lineno}: band {table_band} ion {ion} has "
                            f"{len(fields)} columns but the header puts 'tot' at column "
                            f"{tot_column}"
                        )
                    try:
                        value = _numeric(fields[tot_column])
                    except ValueError as exc:
                        raise ProcarFormatError(
                            f"{path}: line {lineno}: non-numeric tot projection "
                            f"{fields[tot_column]!r} in band {table_band}, ion {ion}"
                        ) from exc
                    if not np.isfinite(value) or value < -1.0e-12:
                        raise ProcarFormatError(
                            f"{path}: line {lineno}: invalid tot projection {value!r} "
                            f"in band {table_band}, ion {ion}; expected a finite "
                            "non-negative number"
                        )
                    row_values.append(max(0.0, value))
                rows_read += 1
                if rows_read == nions:
                    if keep_current:
                        kept_bands.append(table_band)
                        kept_weights.append(row_values)
                    table_band = None
                    row_values = []
                    rows_read = 0
                continue

            band_match = _BAND.match(line)
            if band_match is not None:
                if pending is not None:
                    raise ProcarFormatError(
                        f"{path}: line {lineno}: band {pending} is followed by band "
                        f"{int(band_match.group(1))} before any ionic projection "
                        "header with a 'tot' column"
                    )
                pending = int(band_match.group(1))
                pending_distance = 0
                seen_bands.append(pending)
                continue

            if pending is not None:
                pending_distance += 1
                stripped = line.strip()
                if _KPOINT.match(line):
                    raise ProcarFormatError(
                        f"{path}: line {lineno}: band {pending} is followed by a new "
                        "k-point before any ionic projection header"
                    )
                fields = stripped.split()
                lowered = [field.lower() for field in fields]
                if lowered and lowered[0] == "ion" and "tot" in lowered:
                    tot_column = lowered.index("tot")
                    # A data row carries the ion index where the header carries
                    # the literal 'ion', so the field position is unchanged.
                    table_band = pending
                    keep_current = wanted is None or pending in wanted
                    pending = None
                    rows_read = 0
                    row_values = []
                    continue
                if pending_distance > _HEADER_LOOKAHEAD:
                    raise ProcarFormatError(
                        f"{path}: band {pending} has no ionic projection header with a "
                        f"'tot' column within {_HEADER_LOOKAHEAD} lines (searched from "
                        f"line {lineno - pending_distance})"
                    )

    if counts is None:
        raise ProcarFormatError(
            f"{path}: missing PROCAR k-point/band/ion header; the file may be empty "
            "or truncated"
        )
    if table_band is not None:
        raise ProcarFormatError(
            f"{path}: file ends inside the ion table of band {table_band} after "
            f"{rows_read} of {nions} ion rows"
        )
    if pending is not None:
        raise ProcarFormatError(
            f"{path}: file ends after band {pending} with no ionic projection table"
        )
    if not seen_bands:
        raise ProcarFormatError(f"{path}: no bands with ionic projections found")
    if len(set(seen_bands)) != len(seen_bands):
        duplicates = sorted({b for b in seen_bands if seen_bands.count(b) > 1})[:5]
        raise ProcarFormatError(
            f"{path}: duplicate band numbers {duplicates}; this usually means several "
            "k-points or spin blocks, or a PROCAR layout this reader does not support"
        )
    if len(seen_bands) != nbands_declared:
        raise ProcarFormatError(
            f"{path}: parsed {len(seen_bands)} bands but the header declares "
            f"{nbands_declared}; the file is truncated or its layout is unsupported"
        )
    if wanted is not None:
        missing = sorted(wanted - set(kept_bands))
        if missing:
            raise ProcarFormatError(
                f"{path}: required VASP bands {missing} are not in this PROCAR "
                f"(it holds bands {min(seen_bands)}..{max(seen_bands)})"
            )

    return ProcarProjection(
        path=path,
        bands=np.asarray(kept_bands, dtype=int),
        ion_totals=np.asarray(kept_weights, dtype=float),
        nions=nions,
        bands_seen=len(seen_bands),
        bands_declared=nbands_declared,
    )


def procar_structure(path) -> Dict[str, object]:
    """Cheap header-only look at one PROCAR, for preflight.

    Reads the counts header and the first band's ionic header and stops.  It
    does not validate the whole file; that is what the full reader is for.
    """
    path = Path(path)
    nkpoints = nbands = nions = None
    spins: set = set()
    orbital_columns: Optional[List[str]] = None
    with path.open("r", errors="replace") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.rstrip("\r\n")
            if nkpoints is None:
                match = _COUNTS.search(line)
                if match:
                    nkpoints, nbands, nions = (int(match.group(i)) for i in range(1, 4))
                elif lineno > _COUNTS_LOOKAHEAD:
                    raise ProcarFormatError(
                        f"{path}: missing PROCAR k-point/band/ion header in the first "
                        f"{_COUNTS_LOOKAHEAD} lines"
                    )
                continue
            spin_match = _SPIN.match(line)
            if spin_match is not None:
                spins.add(int(spin_match.group(1)))
                continue
            fields = [field.lower() for field in line.split()]
            if fields and fields[0] == "ion" and "tot" in fields:
                orbital_columns = fields
                break
    if nkpoints is None:
        raise ProcarFormatError(f"{path}: missing PROCAR k-point/band/ion header")
    return {
        "path": str(path),
        "n_kpoints": nkpoints,
        "n_bands": nbands,
        "n_ions": nions,
        "spin_components_seen": sorted(spins),
        "ion_header_columns": orbital_columns,
        "single_kpoint": nkpoints == 1,
        "single_spin_block": len(spins) <= 1,
    }


def iter_band_numbers(path, limit: Optional[int] = None) -> Iterable[int]:
    """Yield the band numbers present in a PROCAR without reading projections.

    ``limit`` stops the scan once that many band lines have been seen, which
    for a single-k-point file is the whole band list.
    """
    seen = 0
    with Path(path).open("r", errors="replace") as handle:
        for raw in handle:
            match = _BAND.match(raw)
            if match is not None:
                yield int(match.group(1))
                seen += 1
                if limit is not None and seen >= limit:
                    return


def procar_band_numbers(path, limit: Optional[int] = None) -> List[int]:
    """The band numbers a PROCAR actually labels its blocks with.

    VASP numbers bands 1..NBANDS, but the numbering is read rather than
    assumed: comparing a required band *number* against a band *count* would
    be wrong for any file that does not follow that convention.
    """
    return list(iter_band_numbers(path, limit=limit))
