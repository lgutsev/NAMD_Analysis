"""What the PROCAR weights are, and how precisely they were written down.

Two questions Issue #4 turns on, answered from files rather than memory.

**Which projection produced the weights.**  For ``LORBIT >= 10`` VASP
projects each band onto the PAW projector functions and *ignores* ``RWIGS``;
for ``LORBIT < 10`` it integrates inside Wigner-Seitz spheres of radius
``RWIGS``.  So an RWIGS sweep is a sensitivity test of the second kind of
PROCAR and a no-op on the first.  The method is read from the INCAR, OUTCAR
or ``vasprun.xml`` of a production frame; a PROCAR alone cannot tell
``LORBIT=1`` from ``11`` (both lm-decomposed) or ``0`` from ``10``, so from a
PROCAR alone the answer is "undetermined", said as such.

**How much weight printing lost.**  VASP prints each ion's ``tot`` to three
decimals and prints the band's grand total on a ``tot`` row that it summed
*before* rounding.  A band spread thinly over many ions loses weight when
each small per-ion value is rounded: an ion at 0.0004 prints as 0.000.  The
character analysis sums the printed per-ion values, so part of a delocalized
band's low ``captured_projection`` can be a printing effect rather than
interstitial density.  The printed grand total measures that directly.  If
the file's ``tot`` row was instead summed after rounding, the two agree to
within one rounding step and the comparison is simply uninformative, which is
also reported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .io.procar import ProcarFormatError, procar_structure, read_procar_ion_totals
from .io.vasp_settings import (
    VaspSettingsError,
    incar_lorbit,
    incar_rwigs,
    outcar_settings,
    read_incar,
    vasprun_lorbit,
)

PAW_PROJECTORS = "paw_projectors"
RWIGS_SPHERES = "rwigs_spheres"
UNDETERMINED = "undetermined"
CONFLICTING = "conflicting"

METHOD_MEANING = {
    PAW_PROJECTORS: (
        "LORBIT >= 10: each band is projected onto the PAW projector functions "
        "of each ion. VASP ignores RWIGS for LORBIT >= 10, so rerunning with "
        "different RWIGS would reproduce these PROCARs unchanged; an RWIGS sweep "
        "is NOT a sensitivity test of these weights. What the projection misses "
        "(1 - total_projection) lies outside the projector regions, and only a "
        "full-space partition of the band density can say where"
    ),
    RWIGS_SPHERES: (
        "LORBIT < 10: each band is integrated inside a Wigner-Seitz sphere of "
        "radius RWIGS around each ion. The weights depend on RWIGS, so an RWIGS "
        "sweep is a meaningful, if partial, sensitivity test of them"
    ),
    UNDETERMINED: (
        "LORBIT could not be read. A PROCAR alone does not determine it "
        "(LORBIT 1 and 11 write the same lm-decomposed layout, 0 and 10 the same "
        "l-decomposed one). Supply the INCAR, OUTCAR or vasprun.xml of a "
        "production frame. Until then, whether an RWIGS sweep tests these "
        "weights is unknown, and no statement here depends on assuming it"
    ),
    CONFLICTING: (
        "the supplied files disagree about LORBIT, so they do not all describe "
        "the run that wrote the PROCARs. Resolve which files belong to the "
        "production run before relying on either"
    ),
}

_LM_COLUMNS = {"py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2", "x2-y2", "fy3x2"}


def _procar_layout(path: Path) -> Dict[str, Any]:
    try:
        structure = procar_structure(path)
    except ProcarFormatError as exc:
        return {"path": str(path), "readable": False, "error": str(exc)}
    with path.open("r", errors="replace") as handle:
        first_line = handle.readline().strip()
    columns = structure.get("ion_header_columns") or []
    orbital = [c for c in columns if c not in ("ion", "tot")]
    if any(c in _LM_COLUMNS for c in orbital):
        layout = "lm-decomposed"
        consistent = [1, 11, 12, 13, 14]
    elif orbital:
        layout = "l-decomposed"
        consistent = [0, 10]
    else:
        layout = "unknown"
        consistent = []
    return {
        "path": str(path),
        "readable": True,
        "first_line": first_line,
        "ion_header_columns": columns,
        "orbital_layout": layout,
        "lorbit_values_consistent_with_layout": consistent,
        "n_kpoints": structure["n_kpoints"],
        "n_bands": structure["n_bands"],
        "n_ions": structure["n_ions"],
    }


def projection_method(
    procar: Optional[Path] = None,
    incar: Optional[Path] = None,
    outcar: Optional[Path] = None,
    vasprun: Optional[Path] = None,
) -> Dict[str, Any]:
    """Resolve LORBIT from whatever files are supplied, and say what it implies.

    OUTCAR and ``vasprun.xml`` record what VASP ran with; the INCAR records
    what was asked for.  All supplied sources are read and must agree.
    """
    sources: Dict[str, Optional[int]] = {}
    details: Dict[str, Any] = {}
    errors: List[str] = []
    rwigs: Dict[str, Any] = {}
    if incar is not None:
        try:
            tags = read_incar(incar)
            sources["INCAR"] = incar_lorbit(tags)
            rwigs["INCAR"] = incar_rwigs(tags)
            details["INCAR"] = str(incar)
        except VaspSettingsError as exc:
            errors.append(str(exc))
    if outcar is not None:
        try:
            settings = outcar_settings(outcar)
            sources["OUTCAR"] = settings["LORBIT"]
            rwigs["OUTCAR"] = settings["RWIGS"]
            details["OUTCAR"] = str(outcar)
        except VaspSettingsError as exc:
            errors.append(str(exc))
    if vasprun is not None:
        try:
            sources["vasprun.xml"] = vasprun_lorbit(vasprun)
            details["vasprun.xml"] = str(vasprun)
        except VaspSettingsError as exc:
            errors.append(str(exc))

    found = {name: value for name, value in sources.items() if value is not None}
    values = sorted(set(found.values()))
    if len(values) > 1:
        method, lorbit = CONFLICTING, None
    elif values:
        lorbit = values[0]
        method = PAW_PROJECTORS if lorbit >= 10 else RWIGS_SPHERES
    else:
        method, lorbit = UNDETERMINED, None

    layout = _procar_layout(Path(procar)) if procar is not None else None
    layout_check = None
    if layout is not None and layout.get("readable") and lorbit is not None:
        allowed = layout["lorbit_values_consistent_with_layout"]
        layout_check = {
            "consistent": (lorbit in allowed) if allowed else None,
            "note": (
                "the PROCAR's orbital columns only narrow LORBIT to a set; they "
                "confirm or contradict the resolved value, they never decide it"
            ),
        }

    return {
        "method": method,
        "LORBIT": lorbit,
        "LORBIT_by_source": sources,
        "sources_read": details,
        "RWIGS_by_source": rwigs,
        "rwigs_sweep_tests_these_weights": (
            False if method == PAW_PROJECTORS else True if method == RWIGS_SPHERES else None
        ),
        "meaning": METHOD_MEANING[method],
        "procar_layout": layout,
        "procar_layout_check": layout_check,
        "errors": errors,
    }


def find_vasp_files(directory: Path) -> Dict[str, Optional[Path]]:
    """INCAR, OUTCAR, vasprun.xml and PROCAR in one frame directory, if present."""
    directory = Path(directory)
    found: Dict[str, Optional[Path]] = {}
    for key, name in (
        ("incar", "INCAR"),
        ("outcar", "OUTCAR"),
        ("vasprun", "vasprun.xml"),
        ("procar", "PROCAR"),
    ):
        candidate = directory / name
        found[key] = candidate if candidate.is_file() else None
    return found


RESOLUTION_HEADER = [
    "frame",
    "band",
    "scope",
    "n_ions",
    "sum_of_printed_values",
    "n_ions_printed_zero",
    "rounding_lower_bound",
    "rounding_upper_bound",
    "printed_band_total",
    "printed_total_minus_sum",
    "print_half_step",
]


def procar_resolution_check(
    path: Path,
    bands: Sequence[int],
    groups: Mapping[str, Sequence[int]],
    frame: Optional[int] = None,
) -> Dict[str, Any]:
    """How much of each band's printed weight rounding could have moved.

    For every requested band: per declared group and for the whole band, the
    sum of the printed per-ion ``tot`` values, how many ions printed as zero,
    and the interval the true sum must lie in given that each printed value is
    within half a print step of the truth (and no true value is negative).
    Then the grand total VASP printed on the ``tot`` row, and its difference
    from the sum of the printed per-ion values.
    """
    path = Path(path)
    projection = read_procar_ion_totals(path, bands=[int(b) for b in bands])
    decimals = projection.tot_decimals
    half = 0.5 * 10.0 ** (-decimals) if decimals is not None else None
    lookup = projection.band_index()
    records: List[Dict[str, Any]] = []
    rows: List[List[Any]] = []
    for band in bands:
        values = projection.ion_totals[lookup[int(band)]]
        scopes = [(name, np.asarray(atoms, dtype=int) - 1) for name, atoms in groups.items()]
        scopes.append(("all_ions", np.arange(projection.nions)))
        band_record: Dict[str, Any] = {"band": int(band), "scopes": {}}
        for name, index in scopes:
            if index.size and index.max() >= projection.nions:
                raise ProcarFormatError(
                    f"{path}: group {name!r} references ion {int(index.max()) + 1} but "
                    f"the PROCAR has {projection.nions}"
                )
            part = values[index]
            summed = float(np.sum(part))
            zero = int(np.count_nonzero(part == 0.0))
            if half is not None:
                lower = float(np.sum(np.maximum(part - half, 0.0)))
                upper = float(np.sum(part + half))
            else:
                lower = upper = None
            band_record["scopes"][name] = {
                "n_ions": int(index.size),
                "sum_of_printed_values": summed,
                "n_ions_printed_zero": zero,
                "rounding_lower_bound": lower,
                "rounding_upper_bound": upper,
            }
            printed = projection.tot_rows.get(int(band)) if name == "all_ions" else None
            difference = None if printed is None else printed - summed
            rows.append(
                [frame, int(band), name, int(index.size), summed, zero, lower, upper,
                 printed, difference, half]
            )
        printed = projection.tot_rows.get(int(band))
        summed = band_record["scopes"]["all_ions"]["sum_of_printed_values"]
        band_record["printed_band_total"] = printed
        band_record["printed_total_minus_sum"] = None if printed is None else printed - summed
        if printed is None:
            reading = "the file has no tot row for this band, so the rounding loss is not measured"
        elif half is None:
            reading = "per-ion values are in scientific notation; print rounding is negligible"
        elif abs(printed - summed) <= half * 1.0001:
            reading = (
                "the printed total agrees with the sum of the printed per-ion values to "
                "within one print step: either little weight was lost to rounding, or "
                "this file's tot row was summed after rounding. The comparison cannot "
                "tell those apart; the rounding bounds still apply"
            )
        else:
            reading = (
                "the printed total differs from the sum of the printed per-ion values by "
                "more than one print step, so the tot row was summed before rounding and "
                "the difference is the net weight per-ion rounding moved"
            )
        band_record["reading"] = reading
        records.append(band_record)
    return {
        "frame": frame,
        "path": str(path),
        "per_ion_decimals": decimals,
        "print_half_step": half,
        "bands": records,
        "rows": rows,
    }
