"""An independent, full-space check of PROCAR fragment fractions.

The PROCAR weights see only the part of a band inside the PAW projector
regions; ``captured_projection`` says how much that was, and normalization
decides where the rest goes.  This module prepares, and then reads back, a
measurement that does not decide it: the band's own density
``|psi_i(r)|^2`` (a band-resolved ``PARCHG``) integrated over a fixed
real-space partition that covers the whole cell -- Bader basins of the total
charge density, and optionally the nearest-atom (Voronoi) partition -- then
summed into the declared fragments.

Nothing here runs VASP.  :func:`write_campaign` writes the inputs and a
batch script; :func:`compare` reads what the run produced and sets the
full-space fractions beside the PROCAR ones, the raw weights and the
allocation bracket from :mod:`namd_analysis.projection_robustness`.

The comparison states numbers.  It does not choose among "comparably mixed",
"mixed with a different magnitude" and "mixing largely disappears"; its
``readings`` block lists, for each of those, the numbers that would support
it, evaluated on the data, and leaves the choice to the reader.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .io.volumetric import Volumetric
from .projection_robustness import (
    FrameMark,
    ProjectionTable,
    RobustnessAudit,
    mixing_levels,
)


class FullspaceError(ValueError):
    """Raised when the validation would have to guess."""


# --------------------------------------------------------------------------
# Choosing frames
# --------------------------------------------------------------------------


@dataclass
class ValidationFrame:
    frame: int
    reasons: List[str] = field(default_factory=list)
    source_dir: Optional[str] = None


def select_frames(
    result: RobustnessAudit,
    band: int,
    windows: Sequence[Any] = (),
    include: Sequence[FrameMark] = (),
    n_worst: int = 3,
    n_mixed: int = 2,
    n_controls: int = 1,
    n_typical: int = 1,
) -> List[ValidationFrame]:
    """Frames for the validation, each with every reason it was chosen.

    Rules, all on ``band`` and all deterministic (ties by frame number):

    * ``worst_capture``: the ``n_worst`` lowest captured projections;
    * ``procar_mixed``: the ``n_mixed`` largest normalized ``min(w_a, w_b)``,
      only where it is positive;
    * ``high_capture_control``: the ``n_controls`` highest captured projections;
    * ``typical_capture``: the frame closest to the median capture;
    * ``window:<name>``: in each window, the frame with the largest normalized
      ``min(w_a, w_b)``, or the window's middle frame when none is positive;
    * each ``include`` mark, under its own label.
    """
    table = result.table
    bi = table.band_position(band)
    frames = table.frames
    captured = table.captured[:, bi]
    mixing = result.arrays.levels["normalized"][:, bi]
    chosen: Dict[int, ValidationFrame] = {}

    def add(frame: int, reason: str) -> None:
        entry = chosen.setdefault(int(frame), ValidationFrame(int(frame)))
        if reason not in entry.reasons:
            entry.reasons.append(reason)

    for i in np.lexsort((frames, captured))[:n_worst]:
        add(frames[i], "worst_capture")
    for i in np.lexsort((frames, -mixing))[:n_mixed]:
        if mixing[i] > 0.0:
            add(frames[i], "procar_mixed")
    for i in np.lexsort((frames, -captured))[:n_controls]:
        add(frames[i], "high_capture_control")
    if n_typical:
        median = float(np.median(captured))
        for i in np.lexsort((frames, np.abs(captured - median)))[:n_typical]:
            add(frames[i], "typical_capture")
    for window in windows:
        inside = np.flatnonzero((frames >= window.first) & (frames <= window.last))
        if inside.size == 0:
            raise FullspaceError(f"window {window.name} holds no frame of the projection table")
        best = inside[np.lexsort((frames[inside], -mixing[inside]))[0]]
        if mixing[best] <= 0.0:
            middle = (window.first + window.last) / 2.0
            best = inside[np.lexsort((frames[inside], np.abs(frames[inside] - middle)))[0]]
        add(frames[best], f"window:{window.name}" + (" (control)" if window.role == "control" else ""))
    present = set(int(f) for f in frames)
    for mark in include:
        for frame in mark.frames:
            if frame not in present:
                raise FullspaceError(f"included frame {frame} ({mark.label}) is not in the table")
            add(frame, mark.label)
    return [chosen[f] for f in sorted(chosen)]


SELECTION_RULES = {
    "worst_capture": "lowest captured_projection on the focus band",
    "procar_mixed": "largest normalized min(w_a, w_b) on the focus band (PROCAR reports mixing)",
    "high_capture_control": "highest captured_projection on the focus band",
    "typical_capture": "captured_projection closest to the focus band's median",
    "window:<name>": (
        "inside that window, the largest normalized min(w_a, w_b); the middle frame "
        "when none is positive"
    ),
}


def procar_reference(
    table: ProjectionTable, frames: Sequence[int], bands: Sequence[int], pair: Tuple[str, str]
) -> Dict[str, Dict[str, Any]]:
    """The production PROCAR numbers each validation sample will be compared to."""
    lookup = {int(f): i for i, f in enumerate(table.frames)}
    raw = table.raw_weights()
    reference: Dict[str, Dict[str, Any]] = {}
    for frame in frames:
        fi = lookup[int(frame)]
        for band in bands:
            bi = table.band_position(band)
            captured = float(table.captured[fi, bi])
            reference[f"{int(frame)}:{int(band)}"] = {
                "frame": int(frame),
                "band": int(band),
                "captured_projection": captured,
                "total_projection": (
                    None if table.total is None else float(table.total[fi, bi])
                ),
                "raw": {g: float(raw[fi, bi, gi]) for gi, g in enumerate(table.groups)},
                "normalized": {
                    g: float(table.weights[fi, bi, gi]) for gi, g in enumerate(table.groups)
                },
            }
    return reference


# --------------------------------------------------------------------------
# Inputs for the run
# --------------------------------------------------------------------------

#: Tags removed from the production INCAR before either step, because they
#: belong to the dynamics or to another post-processing mode.
REMOVED_TAGS = ("LPARD", "IBAND", "EINT", "NBMOD", "KPUSE", "LSEPB", "LSEPK")

SCF_OVERRIDES = {
    "ISTART": ("0", "start from scratch: the run must not depend on a WAVECAR it was not given"),
    "ICHARG": ("2", "superposition of atomic charges, consistent with ISTART = 0"),
    "NSW": ("0", "a single-point calculation on this frame's geometry"),
    "IBRION": ("-1", "no ionic update"),
    "LWAVE": (".TRUE.", "the WAVECAR is the input of the PARCHG step"),
    "LCHARG": (".TRUE.", "the valence CHGCAR is written for reference"),
    "LAECHG": (".TRUE.", "AECCAR0 + AECCAR2 define the Bader basins"),
}

PARCHG_OVERRIDES = {
    "ISTART": ("1", "read the WAVECAR the SCF step wrote"),
    "NSW": ("0", "no ionic update"),
    "IBRION": ("-1", "no ionic update"),
    "LPARD": (".TRUE.", "write band-decomposed densities"),
    "KPUSE": ("1", "the single k-point"),
    "LSEPB": (".TRUE.", "one PARCHG per band"),
    "LSEPK": (".FALSE.", "one k-point; nothing to separate"),
    "LWAVE": (".FALSE.", "keep the SCF WAVECAR unchanged"),
    "LCHARG": (".FALSE.", "keep the SCF CHGCAR unchanged"),
    "LAECHG": (".FALSE.", "keep the SCF AECCAR files unchanged"),
}


def derive_incars(
    base: Mapping[str, str], bands: Sequence[int]
) -> Tuple[Dict[str, str], Dict[str, str], List[Dict[str, Any]]]:
    """The SCF and PARCHG INCARs, and a record of every change from ``base``.

    Everything else -- functional, cutoff, precision, smearing, NBANDS, and
    LORBIT in the SCF step -- is the production run's own, so the SCF
    step reproduces the production projection and its PROCAR can be checked
    against the one the character analysis read.
    """
    base = {str(k).upper(): str(v) for k, v in base.items()}
    if "ISPIN" in base and base["ISPIN"].split()[0] not in ("1",):
        raise FullspaceError(
            f"the production INCAR has ISPIN = {base['ISPIN']}; the character analysis "
            "rejects spin-polarized projections and this validation would not match it"
        )
    if "NBANDS" in base:
        try:
            nbands = int(base["NBANDS"].split()[0])
        except ValueError:
            raise FullspaceError(f"NBANDS = {base['NBANDS']!r} is not an integer") from None
        if max(bands) > nbands:
            raise FullspaceError(
                f"band {max(bands)} exceeds the production NBANDS = {nbands}"
            )
    changes: List[Dict[str, Any]] = []
    scf = {k: v for k, v in base.items() if k not in REMOVED_TAGS}
    for tag in REMOVED_TAGS:
        if tag in base:
            changes.append({"step": "both", "tag": tag, "from": base[tag], "to": None,
                            "why": "belongs to another post-processing mode"})
    for tag, (value, why) in SCF_OVERRIDES.items():
        if base.get(tag) != value:
            changes.append({"step": "scf", "tag": tag, "from": base.get(tag), "to": value,
                            "why": why})
        scf[tag] = value
    parchg = {k: v for k, v in scf.items() if k != "LORBIT"}
    if "LORBIT" in scf:
        changes.append({"step": "parchg", "tag": "LORBIT", "from": scf["LORBIT"], "to": None,
                        "why": "the SCF step's PROCAR is the one kept; this step writes none"})
    for tag, (value, why) in PARCHG_OVERRIDES.items():
        if scf.get(tag) != value:
            changes.append({"step": "parchg", "tag": tag, "from": scf.get(tag), "to": value,
                            "why": why})
        parchg[tag] = value
    parchg["IBAND"] = " ".join(str(int(b)) for b in bands)
    changes.append({"step": "parchg", "tag": "IBAND", "from": None, "to": parchg["IBAND"],
                    "why": "the bands whose densities are partitioned"})
    parchg.pop("ICHARG", None)
    return scf, parchg, changes


JOB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=fullspace_{band}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks}
#SBATCH --time={walltime}
#SBATCH --array=1-{n_frames}
#SBATCH --output=fullspace_%A_%a.out
#SBATCH --error=fullspace_%A_%a.err
#
# Full-space validation of the PROCAR fragment fractions of band {band}.
# Generated by namd-analysis character-fullspace-prepare; see
# validation_manifest.json for why each frame was chosen and every INCAR
# change from the production run.
#
# One array task per frame:
#   1. single-point SCF with the production settings (INCAR.scf), keeping
#      WAVECAR, CHGCAR, AECCAR0/2 and the PROCAR (saved as PROCAR.rerun, so
#      the rerun can be checked against the production projection);
#   2. band-decomposed densities from that WAVECAR (INCAR.parchg);
#   3. Bader basins of AECCAR0 + AECCAR2, and each band's PARCHG integrated
#      over them: ACF_<band>.dat.
#
# Nothing is inferred about the cluster. Set before submitting:
#   VASP_CMD    the command that runs VASP here, e.g. "srun vasp_gam"
#   BADER_CMD   default: bader            (Henkelman group code)
#   CHGSUM_CMD  default: chgsum.pl        (VTST scripts)
# and load whatever modules those need.

set -euo pipefail
: "${{VASP_CMD:?set VASP_CMD to the command that runs VASP, e.g. 'srun vasp_gam'}}"
BADER_CMD=${{BADER_CMD:-bader}}
CHGSUM_CMD=${{CHGSUM_CMD:-chgsum.pl}}
BANDS=({bands})

# The campaign directory is fixed at preparation time, so the script can be
# submitted from anywhere.
cd {campaign}
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" frames.tsv)
FRAME=$(printf '%s' "$LINE" | cut -f1)
SRC=$(printf '%s' "$LINE" | cut -f2)
WORK="frame_${{FRAME}}"
cd "$WORK"

for f in POSCAR POTCAR; do
  [ -s "$SRC/$f" ] || {{ echo "missing $SRC/$f" >&2; exit 3; }}
  cp "$SRC/$f" .
done
if [ -s "$SRC/KPOINTS" ]; then cp "$SRC/KPOINTS" .; fi

# 1. SCF, production settings
cp INCAR.scf INCAR
$VASP_CMD > vasp_scf.log 2>&1
grep -q "aborting loop because EDIFF is reached" OUTCAR || {{
  echo "frame $FRAME: SCF did not report EDIFF convergence" >&2; exit 4; }}
if [ -s PROCAR ]; then
  cp PROCAR PROCAR.rerun
else
  echo "frame $FRAME: no PROCAR written (LORBIT unset?); the rerun check will be reported as not possible" >&2
fi
cp OUTCAR OUTCAR.scf
cp EIGENVAL EIGENVAL.scf 2>/dev/null || true

# 2. band-decomposed densities
cp INCAR.parchg INCAR
$VASP_CMD > vasp_parchg.log 2>&1

# 3. Bader basins of the all-electron reference, band densities integrated in them
$CHGSUM_CMD AECCAR0 AECCAR2 > chgsum.log 2>&1
[ -s CHGCAR_sum ] || {{ echo "frame $FRAME: CHGCAR_sum not written" >&2; exit 5; }}
for B in "${{BANDS[@]}}"; do
  PADDED=$(printf '%04d' "$B")
  shopt -s nullglob
  FOUND=(PARCHG.${{PADDED}}.*)
  shopt -u nullglob
  if [ "${{#FOUND[@]}}" -ne 1 ]; then
    echo "frame $FRAME band $B: expected one PARCHG.${{PADDED}}.*, found ${{#FOUND[@]}}" >&2
    exit 6
  fi
  cp "${{FOUND[0]}}" "PARCHG_${{B}}"
  # -vac off assigns every grid point to an atom basin, so the partition is
  # complete; a vacuum threshold would set part of the band aside.
  $BADER_CMD "PARCHG_${{B}}" -ref CHGCAR_sum -vac off > "bader_${{B}}.log" 2>&1
  mv ACF.dat "ACF_${{B}}.dat"
  mv BCF.dat "BCF_${{B}}.dat" 2>/dev/null || true
  mv AVF.dat "AVF_${{B}}.dat" 2>/dev/null || true
done
touch DONE
"""


def render_job(
    band: int,
    bands: Sequence[int],
    n_frames: int,
    account: Optional[str],
    partition: Optional[str],
    nodes: int,
    ntasks: int,
    walltime: str,
    campaign: str = ".",
) -> str:
    return JOB_TEMPLATE.format(
        campaign=shlex.quote(str(campaign)),
        band=int(band),
        account=account or "REQUIRED_EDIT_account",
        partition=partition or "REQUIRED_EDIT_partition",
        nodes=int(nodes),
        ntasks=int(ntasks),
        walltime=walltime,
        n_frames=int(n_frames),
        bands=" ".join(str(int(b)) for b in bands),
    )


# --------------------------------------------------------------------------
# Partitioning a band density
# --------------------------------------------------------------------------


def fragment_fractions(
    per_atom: np.ndarray,
    groups: Mapping[str, Sequence[int]],
    unassigned: float = 0.0,
) -> Dict[str, Any]:
    """Sum per-atom integrals into fragments, as fractions of the whole band.

    ``unassigned`` (Bader's vacuum charge) is part of the band and part of the
    denominator; it is reported, never dropped.
    """
    per_atom = np.asarray(per_atom, dtype=float)
    covered = sorted(atom for atoms in groups.values() for atom in atoms)
    if covered and covered[-1] > per_atom.size:
        raise FullspaceError(
            f"the atom partition references atom {covered[-1]} but the density "
            f"partition has {per_atom.size} atoms"
        )
    total = float(np.sum(per_atom) + unassigned)
    if not np.isfinite(total) or total <= 0.0:
        raise FullspaceError(f"the partitioned band integrates to {total}; nothing to normalize")
    sums = {name: float(np.sum(per_atom[np.asarray(atoms, dtype=int) - 1]))
            for name, atoms in groups.items()}
    undeclared = float(np.sum(per_atom)) - sum(sums.values())
    return {
        "integral": total,
        "fractions": {name: value / total for name, value in sums.items()},
        "undeclared_atoms_fraction": undeclared / total,
        "unassigned_fraction": float(unassigned) / total,
    }


def nearest_atom_integrals(volume: Volumetric, chunk: int = 1 << 20) -> np.ndarray:
    """Integrate a density over the nearest-atom (Voronoi) cells, periodically.

    Each grid point goes to the atom nearest in Cartesian distance among the
    27 periodic images, which is exact for any cell whose shape does not
    reach past its neighbours (true for the slab cells here).  Returns the
    per-atom integral in the file's electron units.
    """
    from scipy.spatial import cKDTree

    lattice = volume.lattice
    natoms = volume.natoms
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
                      dtype=float)
    images = (volume.fractional[None, :, :] + shifts[:, None, :]).reshape(-1, 3) @ lattice
    owner = np.tile(np.arange(natoms), len(shifts))
    tree = cKDTree(images)
    nx, ny, nz = volume.grid
    values = volume.data.reshape(-1)  # C order over (nx, ny, nz)
    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    fractional = np.stack([ix.reshape(-1) / nx, iy.reshape(-1) / ny, iz.reshape(-1) / nz], axis=1)
    totals = np.zeros(natoms, dtype=float)
    for start in range(0, fractional.shape[0], chunk):
        points = fractional[start : start + chunk] @ lattice
        _, index = tree.query(points)
        totals += np.bincount(owner[index], weights=values[start : start + chunk],
                              minlength=natoms)
    return totals / values.size


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

COMPARISON_HEADER_FIXED = [
    "frame",
    "band",
    "method",
    "reasons",
    "fullspace_integral",
    "unassigned_fraction",
    "undeclared_atoms_fraction",
    "procar_captured_projection",
    "procar_uncaptured_weight",
]

COMPARISON_TAIL = [
    "pair",
    "fullspace_pair_min",
    "procar_pair_min_normalized",
    "procar_pair_min_worst_case",
    "procar_pair_min_best_case",
    "fullspace_over_normalized_pair_min",
    "fullspace_within_allocation_bracket",
    "rerun_max_abs_raw_difference",
    "rerun_captured_difference",
]


def comparison_header(groups: Sequence[str]) -> List[str]:
    return (
        list(COMPARISON_HEADER_FIXED)
        + [f"fullspace_{g}" for g in groups]
        + [f"procar_normalized_{g}" for g in groups]
        + [f"procar_raw_{g}" for g in groups]
        + [f"uncaptured_allocated_to_{g}" for g in groups]
        + [f"uncaptured_proportional_share_{g}" for g in groups]
        + list(COMPARISON_TAIL)
    )


def compare_sample(
    reference: Mapping[str, Any],
    fullspace: Mapping[str, Any],
    pair: Tuple[str, str],
    bracket_tolerance: float = 0.0,
) -> Dict[str, Any]:
    """One (frame, band, method): full-space against PROCAR, in every form."""
    groups = list(reference["raw"])
    captured = float(reference["captured_projection"])
    uncaptured = 1.0 - captured
    raw = reference["raw"]
    normalized = reference["normalized"]
    f = fullspace["fractions"]
    missing = [g for g in groups if g not in f]
    if missing:
        raise FullspaceError(f"the full-space partition has no fragment {missing}")
    a, b = pair
    levels = mixing_levels(np.array(raw[a]), np.array(raw[b]), np.array(captured))
    full_min = min(f[a], f[b])
    normalized_min = float(levels["normalized"])
    allocated = {
        g: ((f[g] - raw[g]) / uncaptured) if uncaptured > 0 else None for g in groups
    }
    proportional = {g: raw[g] / captured for g in groups}
    within = (
        all(raw[g] - bracket_tolerance <= f[g] <= raw[g] + uncaptured + bracket_tolerance
            for g in groups)
        if uncaptured >= 0 else None
    )
    return {
        "fullspace": dict(f),
        "unassigned_fraction": fullspace.get("unassigned_fraction", 0.0),
        "undeclared_atoms_fraction": fullspace.get("undeclared_atoms_fraction", 0.0),
        "integral": fullspace.get("integral"),
        "procar_normalized": dict(normalized),
        "procar_raw": dict(raw),
        "captured_projection": captured,
        "uncaptured_weight": uncaptured,
        "uncaptured_allocated": allocated,
        "uncaptured_proportional_share": proportional,
        "pair": [a, b],
        "fullspace_pair_min": float(full_min),
        "procar_pair_min_normalized": normalized_min,
        "procar_pair_min_worst_case": float(levels["worst_case"]),
        "procar_pair_min_best_case": float(levels["best_case"]),
        "fullspace_over_normalized_pair_min": (
            float(full_min / normalized_min) if normalized_min > 0 else None
        ),
        "fullspace_within_allocation_bracket": within,
    }


def rerun_check(
    procar_path: Path, reference: Mapping[str, Any], groups: Mapping[str, Sequence[int]]
) -> Dict[str, Any]:
    """Does the validation's own SCF reproduce the production projection?

    If it does not, band ``i`` of the rerun may not be the state the
    character analysis called band ``i`` (a near-degeneracy can reorder
    bands), and its PARCHG is compared with the wrong PROCAR row.  Reported,
    never corrected.
    """
    from .io.procar import read_procar_ion_totals

    band = int(reference["band"])
    projection = read_procar_ion_totals(procar_path, bands=[band])
    values = projection.ion_totals[0]
    raw = {name: float(np.sum(values[np.asarray(atoms, dtype=int) - 1]))
           for name, atoms in groups.items()}
    captured = sum(raw.values())
    diffs = {g: raw[g] - float(reference["raw"][g]) for g in reference["raw"] if g in raw}
    return {
        "procar": str(procar_path),
        "rerun_raw": raw,
        "rerun_captured_projection": captured,
        "max_abs_raw_difference": float(max(abs(v) for v in diffs.values())) if diffs else None,
        "captured_difference": captured - float(reference["captured_projection"]),
    }


def readings(
    records: Sequence[Mapping[str, Any]],
    band: int,
    method: str,
    thresholds: Sequence[float],
) -> Dict[str, Any]:
    """The numbers bearing on each of the three possible statements.

    For each threshold tau, over the validation frames of ``band`` that
    PROCAR calls mixed (normalized ``min(w_a, w_b) >= tau``): how many are
    also mixed in full space at the same tau, and the ratio of the full-space
    pair minimum to the normalized one (median and range).  Frames PROCAR
    does not call mixed are counted separately, so appearance of mixing in
    full space is visible too.  No statement is selected.
    """
    rows = [r for r in records if r["band"] == band and r["method"] == method]
    by_tau = []
    for tau in thresholds:
        procar_mixed = [r for r in rows if r["procar_pair_min_normalized"] >= tau]
        procar_not = [r for r in rows if r["procar_pair_min_normalized"] < tau]
        ratios = [r["fullspace_over_normalized_pair_min"] for r in procar_mixed
                  if r["fullspace_over_normalized_pair_min"] is not None]
        by_tau.append(
            {
                "threshold": float(tau),
                "n_frames_procar_mixed": len(procar_mixed),
                "n_procar_mixed_also_fullspace_mixed": sum(
                    1 for r in procar_mixed if r["fullspace_pair_min"] >= tau
                ),
                "ratio_fullspace_to_normalized_median": (
                    float(np.median(ratios)) if ratios else None
                ),
                "ratio_fullspace_to_normalized_min": float(min(ratios)) if ratios else None,
                "ratio_fullspace_to_normalized_max": float(max(ratios)) if ratios else None,
                "n_frames_procar_not_mixed": len(procar_not),
                "n_procar_not_mixed_but_fullspace_mixed": sum(
                    1 for r in procar_not if r["fullspace_pair_min"] >= tau
                ),
            }
        )
    return {
        "band": int(band),
        "method": method,
        "n_frames": len(rows),
        "by_threshold": by_tau,
        "statements": {
            "comparably_mixed": (
                "supported where PROCAR-mixed frames stay mixed in full space at the "
                "same threshold and the full-space/normalized ratio stays near one"
            ),
            "mixed_but_magnitude_changes": (
                "supported where PROCAR-mixed frames stay mixed at a lower threshold "
                "but the ratio departs substantially from one"
            ),
            "mixing_largely_disappears": (
                "supported where few PROCAR-mixed frames remain mixed in full space "
                "at any threshold in the grid, and the ratio is small"
            ),
            "note": (
                "the conditions are stated, not applied: which statement the numbers "
                "support is for the reader to decide, and 'near one' and "
                "'substantially' are deliberately not given a number here"
            ),
        },
    }
