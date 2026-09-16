"""Generate a character-analysis configuration from a real campaign directory.

Hand-writing ``state_map.json``, ``atom_groups.json`` and
``projection_manifest.json`` for a several-hundred-frame campaign is where
mistakes enter that no later check can catch: a column block off by one still
sums to one, and a frame directory silently missing still produces a number.

This module reads what is on disk and proposes a configuration, with one rule
throughout: **it refuses ambiguity rather than resolving it**.  Where more than
one reading of a SHPROP table fits, where a frame directory tree is irregular,
where a preset's ion count disagrees with the PROCAR, it says so and stops.
Every value it writes carries a ``source`` saying whether it was inferred from
the files, taken from a preset a human established, or given on the command
line.

It infers no subsystem names and no atom boundaries.  Those come from a preset
or from the user; element symbols and contiguity are not evidence of physical
partition.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .character import (
    CharacterError,
    _expand_atom_spec,
    resolve_namdtini,
)
from .io.hefei import ShpropStructure, iter_shprop_chunks, shprop_structure
from .io.procar import ProcarFormatError, procar_structure
from .populations import CONSERVATION_ATOL
from .presets import CampaignPreset, PresetError, load_preset

#: Provenance labels.  Every generated value carries exactly one.
FROM_SHPROP = "inferred_from_shprop_table"
FROM_HEADERS = "inferred_from_shprop_headers"
FROM_DIRECTORY = "projection_directory_coverage"
FROM_CLI = "explicit_cli"
FROM_PRESET = "preset"

#: Rows sampled from the head of each history when testing candidate column
#: layouts.  Bounded so that a 900 MB file costs the same as a small one.
COLUMN_SAMPLE_ROWS = 2000


class PrepareError(ValueError):
    """Raised when preparation would have to guess."""


def _provenance(value: Any, source: str, detail: str) -> Dict[str, Any]:
    return {"value": value, "source": source, "detail": detail}


# --------------------------------------------------------------------------
# SHPROP inspection
# --------------------------------------------------------------------------


@dataclass
class ShpropSurvey:
    """What the SHPROP files agree on, and where they do not."""

    structures: List[ShpropStructure]
    bmin: int
    bmax: int
    n_states: int
    n_columns: int
    n_rows: int
    namdtini: List[int]
    nsw: List[Optional[int]]
    header_periods: List[int]
    #: Exact VASP band numbers and where they came from.  The basis is
    #: provenance, not something a SHPROP table reveals.
    band_numbers: List[int] = field(default_factory=list)
    band_numbers_source: str = "unresolved"
    #: One provenance label per history, parallel to ``namdtini``.
    namdtini_sources: List[str] = field(default_factory=list)

    @property
    def paths(self) -> List[Path]:
        return [record.path for record in self.structures]


def _header_int(record: ShpropStructure, key: str) -> int:
    value = record.metadata.get(key)
    if isinstance(value, bool):
        raise PrepareError(f"{record.path}: header {key} is a boolean, not an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise PrepareError(
        f"{record.path}: SHPROP header has no integer {key}. Preparation reads "
        "the basis window and sampling origin from the header and will not infer "
        "them from filenames or table shapes."
    )


def survey_shprop(
    paths: Sequence, preset: Optional[CampaignPreset] = None
) -> ShpropSurvey:
    """Read headers and shapes for every history, refusing disagreement.

    No table is materialized: each file costs one streaming scan and a
    constant amount of memory, whatever its size.

    ``preset`` supplies the exact VASP band numbers for a registered campaign.
    They are needed because a production SHPROP is a plain numeric table with no
    BMIN/BMAX at all, and the basis cannot be recovered from one; the same
    precedence the analysis uses applies here, via
    :func:`~namd_analysis.character.resolve_band_numbers`.
    """
    paths = [Path(path) for path in paths]
    if not paths:
        raise PrepareError("no SHPROP files were supplied")
    structures = [shprop_structure(path) for path in paths]

    widths: Dict[int, List[str]] = {}
    heights: Dict[int, List[str]] = {}
    for record in structures:
        widths.setdefault(record.n_columns, []).append(record.path.name)
        heights.setdefault(record.n_rows, []).append(record.path.name)
    if len(widths) != 1:
        parts = [f"{cols} columns in {sorted(names)}" for cols, names in widths.items()]
        raise PrepareError(
            "SHPROP files have different column counts: "
            + "; ".join(parts)
            + ". A single column layout cannot describe them."
        )
    if len(heights) != 1:
        parts = [f"{rows} rows in {sorted(names)}" for rows, names in heights.items()]
        raise PrepareError(
            "SHPROP files have different row counts: "
            + "; ".join(parts)
            + ". They cannot be averaged on one time grid, and no interpolation "
            "or truncation is performed."
        )

    bands = _survey_bands(structures, preset)
    bmin, bmax = bands[0], bands[-1]

    namdtini: List[int] = []
    namdtini_sources: List[str] = []
    for record in structures:
        try:
            start, source = resolve_namdtini(record)
        except CharacterError as exc:
            raise PrepareError(str(exc)) from exc
        namdtini.append(start)
        namdtini_sources.append(source)

    nsw: List[Optional[int]] = []
    for record in structures:
        raw = record.metadata.get("NSW")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            nsw.append(None)
        else:
            nsw.append(int(raw))
    periods = sorted({value - 1 for value in nsw if value is not None})
    return ShpropSurvey(
        structures=structures,
        bmin=bmin,
        bmax=bmax,
        n_states=len(bands),
        n_columns=next(iter(widths)),
        n_rows=next(iter(heights)),
        namdtini=namdtini,
        nsw=nsw,
        header_periods=periods,
        band_numbers=bands,
        band_numbers_source=_band_source(preset, structures),
        namdtini_sources=namdtini_sources,
    )


def _band_source(
    preset: Optional[CampaignPreset], structures: Sequence[ShpropStructure]
) -> str:
    if preset is not None and preset.band_numbers:
        return f"preset_{preset.preset}_{preset.campaign}"
    return "SHPROP_BMIN_BMAX_metadata"


def _survey_bands(
    structures: Sequence[ShpropStructure], preset: Optional[CampaignPreset]
) -> List[int]:
    """Exact VASP bands for the survey, by the same precedence as the analysis."""
    if preset is not None and preset.band_numbers:
        return list(preset.band_numbers)
    windows: Dict[Tuple[int, int], List[str]] = {}
    for record in structures:
        low = record.metadata.get("BMIN")
        high = record.metadata.get("BMAX")
        if not isinstance(low, int) or not isinstance(high, int):
            raise PrepareError(
                f"{record.path}: this SHPROP carries no integer BMIN/BMAX, and the "
                "exact VASP band numbers are not recoverable from a plain numeric "
                "table. Use a registered campaign preset (--preset/--campaign), or "
                "supply a state map declaring 'band_numbers'."
            )
        if high < low:
            raise PrepareError(
                f"{record.path}: optional metadata has BMAX={high} < BMIN={low}"
            )
        windows.setdefault((low, high), []).append(record.path.name)
    if len(windows) != 1:
        parts = [f"{lo}:{hi} in {sorted(names)}" for (lo, hi), names in windows.items()]
        raise PrepareError(
            "SHPROP files carry different optional BMIN/BMAX windows: "
            + "; ".join(parts)
            + ". One configuration covers one basis; declare band_numbers explicitly "
            "or split the campaign by window."
        )
    (low, high) = next(iter(windows))
    return list(range(low, high + 1))


@dataclass
class ColumnInference:
    """A candidate column layout and the evidence for or against it."""

    time_column: int
    population_columns: List[int]
    accepted: bool
    reasons: List[str] = field(default_factory=list)
    min_value: float = 0.0
    max_value: float = 0.0
    min_row_sum: float = 0.0
    max_row_sum: float = 0.0


def _sample_rows(path: Path, rows: int) -> np.ndarray:
    for _offset, chunk in iter_shprop_chunks(path, rows):
        return chunk
    raise PrepareError(f"{path}: no numeric rows to sample")


def infer_columns(
    survey: ShpropSurvey,
    sample_rows: int = COLUMN_SAMPLE_ROWS,
    atol: float = CONSERVATION_ATOL,
) -> Tuple[List[int], int, Dict[str, Any]]:
    """Find the one contiguous population block that fits, or refuse.

    Column *count* alone is not evidence when more than one block of the right
    width exists, so each candidate is tested against real values: populations
    lie in [0,1] and, for a complete basis, sum to one across the block.  A
    candidate that includes the energy column fails both.  If two candidates
    survive, the layout is genuinely ambiguous from the file and preparation
    stops rather than picking one.
    """
    n_states = survey.n_states
    n_columns = survey.n_columns
    if n_columns <= n_states:  # noqa: SIM102 - kept separate for the message
        raise PrepareError(
            f"SHPROP files have {n_columns} columns but the header basis "
            f"{survey.bmin}:{survey.bmax} needs {n_states} population columns plus "
            "a time column. The header and the table disagree; neither is assumed "
            "to be right."
        )

    samples = [_sample_rows(record.path, sample_rows) for record in survey.structures]
    time_column = 0
    for record, sample in zip(survey.structures, samples):
        times = sample[:, time_column]
        if times.size > 1 and np.any(np.diff(times) <= 0):
            raise PrepareError(
                f"{record.path}: column {time_column} does not increase strictly, so "
                "it is not the time column. Preparation does not search for one."
            )

    candidates: List[ColumnInference] = []
    for start in range(0, n_columns - n_states + 1):
        block = list(range(start, start + n_states))
        if time_column in block:
            continue
        candidate = ColumnInference(
            time_column=time_column, population_columns=block, accepted=True
        )
        lo, hi = np.inf, -np.inf
        smin, smax = np.inf, -np.inf
        for sample in samples:
            values = sample[:, block]
            lo = min(lo, float(values.min()))
            hi = max(hi, float(values.max()))
            sums = values.sum(axis=1)
            smin = min(smin, float(sums.min()))
            smax = max(smax, float(sums.max()))
        candidate.min_value, candidate.max_value = float(lo), float(hi)
        candidate.min_row_sum, candidate.max_row_sum = float(smin), float(smax)
        if lo < -atol or hi > 1.0 + atol:
            candidate.accepted = False
            candidate.reasons.append(
                f"values span [{lo:.6g}, {hi:.6g}], outside [0,1]"
            )
        if abs(smin - 1.0) > atol or abs(smax - 1.0) > atol:
            candidate.accepted = False
            candidate.reasons.append(
                f"row sums span [{smin:.6g}, {smax:.6g}], not one"
            )
        if candidate.accepted:
            candidate.reasons.append(
                f"values within [0,1] and row sums one to {atol:g} over "
                f"{sum(s.shape[0] for s in samples)} sampled row(s)"
            )
        candidates.append(candidate)

    accepted = [c for c in candidates if c.accepted]
    sampled = int(samples[0].shape[0])
    rationale = {
        "n_columns": n_columns,
        "n_states_from_header": n_states,
        "sampled_rows_per_file": sampled,
        "total_rows_per_file": int(survey.n_rows),
        "sample_covers_whole_file": sampled >= survey.n_rows,
        "sampling_caveat": (
            "candidates are tested against the first "
            f"{sampled} of {survey.n_rows} row(s), so that preparation costs the "
            "same on a 900 MB history as on a small one. A file whose populations "
            "leave [0,1] or stop summing to one only later in the trajectory would "
            "not be caught here -- but it is caught by the analysis, which "
            "validates every row of every chunk and refuses to produce a number"
        ),
        "candidates": [
            {
                "population_columns": c.population_columns,
                "accepted": c.accepted,
                "min_value": c.min_value,
                "max_value": c.max_value,
                "min_row_sum": c.min_row_sum,
                "max_row_sum": c.max_row_sum,
                "reasons": c.reasons,
            }
            for c in candidates
        ],
        "rule": (
            "a candidate is one contiguous block of BMAX-BMIN+1 columns that "
            "excludes the time column; it is accepted only if every sampled value "
            "lies in [0,1] and every sampled row sums to one. Exactly one "
            "candidate must survive"
        ),
    }
    if not accepted:
        raise PrepareError(
            f"no contiguous block of {n_states} columns in a {n_columns}-column "
            "SHPROP table behaves like a complete population: "
            + "; ".join(
                f"{c.population_columns}: {', '.join(c.reasons)}" for c in candidates
            )
            + ". Write the state map by hand, or check that the header basis "
            "matches the table."
        )
    if len(accepted) > 1:
        raise PrepareError(
            f"{len(accepted)} column layouts fit equally well "
            + f"({[c.population_columns for c in accepted]}); the table alone does "
            "not say which is right, so nothing is chosen. Declare "
            "population_columns explicitly in a state map."
        )
    chosen = accepted[0]
    rationale["chosen"] = chosen.population_columns
    return chosen.population_columns, time_column, rationale


# --------------------------------------------------------------------------
# Projection directory discovery
# --------------------------------------------------------------------------

_NUMERIC_DIR = re.compile(r"^\d+$")


@dataclass
class FrameDiscovery:
    """Numbered frame directories found under a projection directory."""

    root: Path
    frames: Dict[int, Path]
    first: int
    last: int
    missing: List[int]
    duplicates: Dict[int, List[str]]
    numeric_dirs_without_procar: List[str]
    padding: Optional[int]
    regular: bool
    irregular_reason: Optional[str]

    @property
    def count(self) -> int:
        return len(self.frames)

    def summary(self) -> Dict[str, Any]:
        return {
            "projection_directory": str(self.root),
            "discovered_frames": self.count,
            "first_frame": self.first,
            "last_frame": self.last,
            "contiguous": not self.missing,
            "missing_frames": self.missing[:50],
            "n_missing_frames": len(self.missing),
            "duplicate_spellings": {
                str(k): v for k, v in sorted(self.duplicates.items())
            },
            "numeric_dirs_without_procar": self.numeric_dirs_without_procar[:20],
            "zero_padding_width": self.padding,
            "layout_is_regular": self.regular,
            "irregular_reason": self.irregular_reason,
        }


def discover_frames(projection_dir, procar_name: str = "PROCAR") -> FrameDiscovery:
    """Map numbered subdirectories to frame numbers, reporting every anomaly.

    A directory is a frame only if its name is entirely digits and it contains
    the PROCAR.  Two spellings of the same number (``1`` and ``0001``) are a
    duplicate, not a choice to be made quietly: which one the analysis would
    have read depends on directory order, so it is an error.
    """
    root = Path(projection_dir)
    if not root.is_dir():
        raise PrepareError(f"{root}: projection directory does not exist")
    frames: Dict[int, Path] = {}
    spellings: Dict[int, List[str]] = {}
    without: List[str] = []
    paddings: set = set()
    unpadded = True
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or not _NUMERIC_DIR.match(entry.name):
            continue
        number = int(entry.name)
        procar = entry / procar_name
        if not procar.is_file():
            without.append(entry.name)
            continue
        spellings.setdefault(number, []).append(entry.name)
        frames[number] = procar
        paddings.add(len(entry.name))
        if entry.name != str(number):
            unpadded = False
    if not frames:
        raise PrepareError(
            f"{root}: no numbered subdirectory contains a {procar_name}. "
            f"{len(without)} numeric directory/directories lack one"
            + (f" (first: {without[:5]})" if without else "")
            + ". Preparation looks only for all-digit directory names."
        )
    duplicates = {n: names for n, names in spellings.items() if len(names) > 1}
    if duplicates:
        shown = "; ".join(f"frame {n} as {names}" for n, names in sorted(duplicates.items()))
        raise PrepareError(
            f"{root}: the same frame number appears under more than one directory "
            f"spelling ({shown}). Which PROCAR would be read depends on directory "
            "order, so this is not resolved automatically."
        )
    numbers = sorted(frames)
    first, last = numbers[0], numbers[-1]
    missing = [n for n in range(first, last + 1) if n not in frames]
    # An unpadded tree has mixed name widths (1..9 then 10..) and is still
    # perfectly regular: "{frame}" reproduces every name exactly. Only a
    # *zero-padded* tree needs one shared width.
    if unpadded:
        padding = 1
    else:
        padding = next(iter(paddings)) if len(paddings) == 1 else None
    regular = True
    reason: Optional[str] = None
    if padding is None:
        regular = False
        reason = (
            f"directory names use mixed widths {sorted(paddings)} and are not the "
            "plain decimal spelling, so no single pattern reproduces them"
        )
    elif missing:
        regular = False
        reason = f"{len(missing)} frame(s) are missing between {first} and {last}"
    elif padding not in (1, len(str(last))) and padding < len(str(last)):
        regular = False
        reason = f"zero padding of {padding} cannot represent frame {last}"
    return FrameDiscovery(
        root=root,
        frames=frames,
        first=first,
        last=last,
        missing=missing,
        duplicates=duplicates,
        numeric_dirs_without_procar=without,
        padding=padding,
        regular=regular,
        irregular_reason=reason,
    )


def build_manifest_payload(
    discovery: FrameDiscovery,
    cycle_length: Optional[int],
    procar_name: str = "PROCAR",
    absolute: bool = True,
) -> Dict[str, Any]:
    """A pattern manifest for a regular tree, explicit entries otherwise."""
    root = discovery.root.resolve() if absolute else discovery.root
    payload: Dict[str, Any]
    if discovery.regular and discovery.padding is not None:
        width = discovery.padding
        field = "{frame}" if width == 1 else f"{{frame:0{width}d}}"
        payload = {
            "procar_pattern": f"{root.as_posix()}/{field}/{procar_name}",
            "first_frame": discovery.first,
            "last_frame": discovery.last,
            "frame_step": 1,
        }
    else:
        payload = {
            "frames": [
                {"frame": number, "procar": discovery.frames[number].resolve().as_posix()}
                for number in sorted(discovery.frames)
            ]
        }
    if cycle_length is not None:
        payload["cycle_length"] = int(cycle_length)
    return payload


# --------------------------------------------------------------------------
# Cycle length
# --------------------------------------------------------------------------


def propose_cycle_length(
    survey: ShpropSurvey,
    discovery: FrameDiscovery,
    frame_mode: str,
    explicit: Optional[int] = None,
) -> Dict[str, Any]:
    """Offer a period, and say loudly when the two sources disagree.

    Directory coverage and ``NSW - 1`` are independent statements about the
    same number.  When they agree, the proposal is safe.  When they do not,
    nothing is chosen: the engine that wrote the headers is the authority on
    its own bookkeeping, and this package cannot see it.
    """
    coverage = None
    if discovery.first == 1 and not discovery.missing:
        coverage = discovery.last
    headers = list(survey.header_periods)
    record: Dict[str, Any] = {
        "frame_mode": frame_mode,
        "available_frames_first": discovery.first,
        "available_frames_last": discovery.last,
        "available_frame_count": discovery.count,
        "contiguous_from_one": coverage is not None,
        "coverage_derived_cycle_length": coverage,
        "nsw_values": sorted({v for v in survey.nsw if v is not None}),
        "header_derived_periods_nsw_minus_1": headers,
        "explicit_cycle_length": explicit,
    }
    if frame_mode != "dish-cyclic":
        record.update(
            selected=None,
            source="not_applicable",
            agrees=None,
            note="linear alignment has no period; cycle_length is not written",
        )
        return record
    if explicit is not None:
        record.update(
            selected=int(explicit),
            source=FROM_CLI,
            agrees=(headers == [explicit]) if headers else None,
            note=(
                "taken from --cycle-length; it overrides both NSW-1 and directory "
                "coverage, and the disagreement below is recorded, not resolved"
                if headers and headers != [explicit]
                else "taken from --cycle-length and consistent with the headers"
            ),
        )
        return record
    if len(headers) > 1:
        record.update(
            selected=None,
            source="ambiguous",
            agrees=False,
            note=(
                f"the histories imply different periods from their own NSW headers "
                f"({headers}); each would wrap on a different cycle. Split the "
                "campaign by period, or pass --cycle-length if you know which is "
                "correct."
            ),
        )
        return record
    header = headers[0] if headers else None
    if coverage is not None and header is not None and coverage != header:
        record.update(
            selected=None,
            source="ambiguous",
            agrees=False,
            note=(
                f"the projection directory covers frames 1..{coverage} while the "
                f"SHPROP headers imply NSW-1 = {header}. These are independent "
                "statements about the same period and they disagree. Nothing is "
                "chosen: pass --cycle-length to state which is right, after "
                "checking whether frames are missing from the archive or whether "
                "the header describes a different run length."
            ),
        )
        return record
    if coverage is not None:
        record.update(
            selected=int(coverage),
            source=FROM_DIRECTORY,
            agrees=(header == coverage) if header is not None else None,
            note=(
                f"frames 1..{coverage} are present with no gaps"
                + (
                    f" and NSW-1 = {header} agrees"
                    if header is not None
                    else " and no NSW header was available to compare"
                )
            ),
        )
        return record
    record.update(
        selected=header,
        source=FROM_HEADERS if header is not None else "unresolved",
        agrees=None,
        note=(
            f"the projection directory starts at frame {discovery.first} or has gaps, "
            "so coverage cannot propose a period"
            + (f"; NSW-1 = {header} is used" if header is not None else " and no NSW header was available")
        ),
    )
    return record


# --------------------------------------------------------------------------
# Structure / PROCAR cross-checks
# --------------------------------------------------------------------------


def representative_procar(discovery: FrameDiscovery) -> Dict[str, Any]:
    """Header-only probe of the first available frame's PROCAR."""
    number = min(discovery.frames)
    path = discovery.frames[number]
    try:
        structure = procar_structure(path)
    except ProcarFormatError as exc:
        raise PrepareError(f"{path}: {exc}") from exc
    return {"frame": number, "path": str(path), **structure}


def poscar_ion_count(path) -> Optional[Dict[str, Any]]:
    """Total ion count and element blocks from a POSCAR/CONTCAR, if present.

    Element counts are read because they are a cross-check on the ion total.
    They are *not* used to name or bound a subsystem: a contiguous run of one
    element is not evidence that the run is one physical group.
    """
    path = Path(path)
    if not path.is_file():
        return None
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 7:
        return None
    for index in (5, 6):
        if index >= len(lines):
            return None
        fields = lines[index].split()
        if fields and all(field.isdigit() for field in fields):
            counts = [int(field) for field in fields]
            symbols = lines[index - 1].split() if index == 6 else []
            return {
                "path": str(path),
                "n_ions": sum(counts),
                "element_counts": counts,
                "element_symbols": symbols if len(symbols) == len(counts) else [],
                "note": (
                    "element blocks are reported as a cross-check on the ion total "
                    "only; a contiguous run of one element is not evidence of a "
                    "physical subsystem boundary"
                ),
            }
    return None


def find_structure_file(projection_dir, discovery: FrameDiscovery) -> Optional[Path]:
    """Locate a POSCAR/CONTCAR beside the campaign or its first frame."""
    candidates: List[Path] = []
    root = Path(projection_dir)
    first = discovery.frames[min(discovery.frames)].parent
    for base in (root, first):
        for name in ("CONTCAR", "POSCAR"):
            candidates.append(base / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


# --------------------------------------------------------------------------
# Atom groups
# --------------------------------------------------------------------------


def validate_atom_groups(
    payload: Dict[str, Any], procar_ions: int, origin: str
) -> Dict[str, Any]:
    """Check a partition against the structure it claims to describe.

    A partition that was right for one campaign and is silently applied to
    another is the failure this guards against, so the ion count is compared
    before anything is written, and a mismatch is fatal rather than a warning.
    """
    groups = payload.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise PrepareError(f"{origin}: atom groups must be a nonempty object")
    expanded = {name: _expand_atom_spec(spec) for name, spec in groups.items()}
    seen: Dict[int, str] = {}
    for name, atoms in expanded.items():
        if not atoms:
            raise PrepareError(f"{origin}: atom group {name!r} is empty")
        for atom in atoms:
            if atom <= 0:
                raise PrepareError(
                    f"{origin}: group {name!r} has ion index {atom}; PROCAR ion "
                    "indices are one-based positive integers"
                )
            if atom in seen:
                raise PrepareError(
                    f"{origin}: PROCAR ion {atom} appears in both {seen[atom]!r} "
                    f"and {name!r}; groups must be disjoint"
                )
            seen[atom] = name
    assigned = sorted(seen)
    highest = assigned[-1]
    if highest > procar_ions:
        raise PrepareError(
            f"{origin}: references PROCAR ion {highest}, but the representative "
            f"PROCAR has only {procar_ions} ions. The partition does not describe "
            "this structure; it is not trimmed to fit."
        )
    complete = bool(payload.get("complete_atoms", True))
    unassigned = sorted(set(range(1, procar_ions + 1)) - set(assigned))
    if complete and unassigned:
        raise PrepareError(
            f"{origin}: complete_atoms is true, but {len(unassigned)} of the "
            f"{procar_ions} PROCAR ions are unassigned (first: {unassigned[:10]}). "
            "Either complete the partition or set complete_atoms false."
        )
    return {
        "origin": origin,
        "group_names": list(expanded),
        "ions_per_group": {name: len(atoms) for name, atoms in expanded.items()},
        "assigned_ions": len(assigned),
        "procar_ions": procar_ions,
        "lowest_ion": assigned[0],
        "highest_ion": highest,
        "complete_atoms": complete,
        "unassigned_ions": unassigned[:20],
        "n_unassigned_ions": len(unassigned),
        "covers_exactly": complete and not unassigned and assigned[0] == 1,
    }


# --------------------------------------------------------------------------
# Slurm
# --------------------------------------------------------------------------


@dataclass
class SlurmOptions:
    """Resources for the generated batch script; all overridable."""

    account: Optional[str] = None
    partition: Optional[str] = None
    nodes: int = 1
    tasks: int = 1
    cpus: int = 1
    memory: str = "32G"
    time: str = "04:00:00"
    job_name: str = "character"


def render_sbatch(
    slurm: SlurmOptions,
    shprop_paths: Sequence[Path],
    state_map: Path,
    atom_groups: Path,
    manifest: Path,
    frame_mode: str,
    out_dir: str,
    extra_args: Sequence[str] = (),
) -> str:
    """A batch script that checks before it computes, and says what it ran.

    Preflight runs first and a non-zero exit aborts the job, because the whole
    point of preflight is to fail in seconds rather than after hours of PROCAR
    parsing.  ``set -euo pipefail`` makes that abort automatic rather than
    dependent on the next command noticing.
    """
    directives = [
        f"#SBATCH --job-name={slurm.job_name}",
        f"#SBATCH --nodes={slurm.nodes}",
        f"#SBATCH --ntasks={slurm.tasks}",
        f"#SBATCH --cpus-per-task={slurm.cpus}",
        f"#SBATCH --mem={slurm.memory}",
        f"#SBATCH --time={slurm.time}",
        "#SBATCH --output=character_%j.out",
        "#SBATCH --error=character_%j.err",
    ]
    if slurm.account:
        directives.insert(1, f"#SBATCH --account={slurm.account}")
    if slurm.partition:
        directives.insert(1, f"#SBATCH --partition={slurm.partition}")

    files = " \\\n    ".join(f'"{path.as_posix()}"' for path in shprop_paths)
    common = [
        f'  --files {files}',
        f'  --config "{state_map.as_posix()}"',
        f'  --projection-manifest "{manifest.as_posix()}"',
        f'  --atom-groups "{atom_groups.as_posix()}"',
        f"  --frame-mode {frame_mode}",
    ]
    for arg in extra_args:
        common.append(f"  {arg}")
    preflight_body = " \\\n".join(common)
    run_body = " \\\n".join(common + [f'  --out "{out_dir}"'])

    lines = [
        "#!/bin/bash",
        *directives,
        "",
        "# Generated by namd-analysis character-prepare. Review before submitting:",
        "# the resources below are a starting proposal, not a measurement.",
        "set -euo pipefail",
        "",
        'echo "host:      $(hostname)"',
        'echo "date:      $(date -Is)"',
        'echo "workdir:   $(pwd)"',
        'echo "slurm job: ${SLURM_JOB_ID:-<none>}"',
        'echo "binary:    $(command -v namd-analysis)"',
        "python -c 'import namd_analysis; print(\"version:   \" + namd_analysis.__version__)'",
        "python - <<'PROVENANCE'",
        "import pathlib, subprocess, namd_analysis",
        "root = pathlib.Path(namd_analysis.__file__).resolve().parents[2]",
        "if (root / '.git').exists():",
        "    try:",
        "        commit = subprocess.check_output(",
        "            ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()",
        "        dirty = subprocess.check_output(",
        "            ['git', '-C', str(root), 'status', '--porcelain'], text=True).strip()",
        "        print('commit:    ' + commit + (' (dirty)' if dirty else ''))",
        "    except Exception as exc:",
        "        print('commit:    unavailable (' + str(exc) + ')')",
        "else:",
        "    print('commit:    not an editable checkout')",
        "PROVENANCE",
        "",
        'echo "--- inputs ---"',
    ]
    for path in shprop_paths:
        lines.append(f'ls -l "{path.as_posix()}"')
    lines += [
        f'ls -l "{state_map.as_posix()}" "{atom_groups.as_posix()}" "{manifest.as_posix()}"',
        "",
        'echo "--- preflight (cheap; a non-zero exit aborts this job) ---"',
        "namd-analysis character-preflight \\",
        preflight_body,
        "",
        'echo "--- full analysis ---"',
        "/usr/bin/time -v namd-analysis character-populations \\",
        run_body,
        "",
        'echo "--- generated files ---"',
        f'ls -l "{out_dir}"',
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


@dataclass
class PreparedCampaign:
    """Everything preparation decided, and the files it wrote."""

    report: Dict[str, Any]
    written: List[Path]
    state_map_path: Path
    atom_groups_path: Path
    manifest_path: Path
    sbatch_path: Optional[Path]
    unresolved: List[str]

    @property
    def ok(self) -> bool:
        return not self.unresolved


def prepare_campaign(
    shprop_paths: Sequence,
    projection_dir,
    out_dir,
    frame_mode: str,
    preset: Optional[str] = None,
    campaign: Optional[str] = None,
    cycle_length: Optional[int] = None,
    atom_groups_json: Optional[Any] = None,
    slurm: Optional[SlurmOptions] = None,
    write_sbatch: bool = False,
    results_dir: str = "results/character",
    procar_name: str = "PROCAR",
    extra_run_args: Sequence[str] = (),
) -> PreparedCampaign:
    """Read a campaign and write a configuration for it, or explain why not.

    Ordering matters: everything that could contradict something else is read
    before anything is written, so a campaign that cannot be prepared leaves no
    half-written configuration behind.
    """
    out = Path(out_dir)
    campaign_preset: Optional[CampaignPreset] = None
    if preset is not None:
        try:
            campaign_preset = load_preset(preset, campaign)
        except PresetError as exc:
            raise PrepareError(str(exc)) from exc
    survey = survey_shprop(shprop_paths, preset=campaign_preset)
    columns, time_column, column_rationale = infer_columns(survey)
    discovery = discover_frames(projection_dir, procar_name=procar_name)
    procar = representative_procar(discovery)
    cycle = propose_cycle_length(survey, discovery, frame_mode, cycle_length)

    unresolved: List[str] = []
    warnings: List[str] = []

    # --- state map -------------------------------------------------------
    if campaign_preset is not None:
        if list(campaign_preset.population_columns) != list(columns):
            raise PrepareError(
                f"preset {campaign_preset.name} declares population columns "
                f"{campaign_preset.population_columns}, but the SHPROP tables "
                f"support {columns}. The preset was written for a differently "
                "shaped run; nothing is reconciled automatically."
            )
        if campaign_preset.time_column != time_column:
            raise PrepareError(
                f"preset {campaign_preset.name} declares time column "
                f"{campaign_preset.time_column}, but column {time_column} is the "
                "one that increases in these files"
            )
        state_map_payload = campaign_preset.state_map_payload()
        state_map_source = f"{FROM_PRESET}_{preset}_{campaign_preset.campaign}"
        state_map_detail = (
            "group membership and state order come from the preset; the column "
            "block was independently inferred from the tables and matches"
        )
    else:
        state_map_payload = {
            "name": Path(projection_dir).name or "campaign",
            "time_column": time_column,
            "time_unit": "fs",
            "population_columns": list(columns),
            "groups": {},
            "complete_population": True,
            "recombined_group": None,
            "notes": (
                "population columns were inferred from the table; the grouping of "
                "those columns into named states was NOT, because a column's "
                "physical identity is a property of the run that wrote it"
            ),
        }
        unresolved.append(
            "state_map.json has no groups: the column block was inferred but which "
            "state each column holds is a property of the run, not of the file. "
            "Fill in 'groups' (and 'recombined_group') by hand, or use a preset."
        )
        state_map_source = FROM_SHPROP
        state_map_detail = "column block inferred; state identities left unresolved"

    # --- atom groups -----------------------------------------------------
    procar_ions = int(procar["n_ions"])
    if atom_groups_json is not None:
        given = Path(atom_groups_json)
        atom_payload = json.loads(given.read_text(encoding="utf-8"))
        atom_source = FROM_CLI
        atom_origin = str(given)
    elif campaign_preset is not None:
        if campaign_preset.n_ions != procar_ions:
            raise PrepareError(
                f"preset {campaign_preset.name} describes a {campaign_preset.n_ions}-ion "
                f"structure, but the representative PROCAR (frame {procar['frame']}) "
                f"has {procar_ions} ions. This is the wrong campaign for this preset; "
                "the partition is not rescaled or trimmed."
            )
        atom_payload = campaign_preset.atom_groups_payload()
        atom_source = f"{FROM_PRESET}_{preset}_{campaign_preset.campaign}"
        atom_origin = f"preset {campaign_preset.name}"
    else:
        raise PrepareError(
            "no atom grouping is available: pass --preset for a registered campaign "
            "or --atom-groups with a partition you have confirmed. Subsystem names "
            "and atom boundaries are never inferred from the structure, because "
            "element symbols and contiguity are not evidence of physical grouping."
        )
    atom_validation = validate_atom_groups(atom_payload, procar_ions, atom_origin)

    structure_path = find_structure_file(projection_dir, discovery)
    structure_info = poscar_ion_count(structure_path) if structure_path else None
    if structure_info and structure_info["n_ions"] != procar_ions:
        warnings.append(
            f"{structure_info['path']} describes {structure_info['n_ions']} ions but "
            f"the representative PROCAR has {procar_ions}; the PROCAR is what the "
            "analysis reads, and the disagreement is reported rather than resolved"
        )

    # --- manifest --------------------------------------------------------
    if discovery.missing:
        warnings.append(
            f"{len(discovery.missing)} frame director(ies) are missing between "
            f"{discovery.first} and {discovery.last} (first: {discovery.missing[:10]}); "
            "the manifest lists only what exists, and a history that visits a missing "
            "frame will fail preflight"
        )
    if cycle["source"] == "ambiguous":
        unresolved.append(cycle["note"])
    selected_period = cycle.get("selected")
    if selected_period is not None:
        # A period the archive cannot cover would fail later, during the first
        # real run, after every PROCAR header had already been read. Say it
        # here, with the count actually found.
        uncovered = [n for n in range(1, int(selected_period) + 1) if n not in discovery.frames]
        if uncovered:
            unresolved.append(
                f"cycle_length {selected_period} needs frames 1..{selected_period}, but "
                f"{len(uncovered)} of them have no PROCAR directory (first: "
                f"{uncovered[:10]}). The projection directory holds "
                f"{discovery.count} frame(s), {discovery.first}..{discovery.last}. "
                "Resolve which is right before running: either frames are missing "
                "from the archive, or the period is not what you passed."
            )
    manifest_payload = build_manifest_payload(
        discovery, cycle.get("selected"), procar_name=procar_name
    )

    # --- write -----------------------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    state_map_path = out / "state_map.json"
    atom_groups_path = out / "atom_groups.json"
    manifest_path = out / "projection_manifest.json"
    written: List[Path] = []
    for path, payload in (
        (state_map_path, state_map_payload),
        (atom_groups_path, atom_payload),
        (manifest_path, manifest_payload),
    ):
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written.append(path)

    sbatch_path: Optional[Path] = None
    if write_sbatch:
        options = slurm or SlurmOptions()
        script = render_sbatch(
            options,
            survey.paths,
            state_map_path.resolve(),
            atom_groups_path.resolve(),
            manifest_path.resolve(),
            frame_mode,
            results_dir,
            extra_args=extra_run_args,
        )
        sbatch_path = out / "run_character_test.sbatch"
        sbatch_path.write_text(script, encoding="utf-8")
        written.append(sbatch_path)

    report: Dict[str, Any] = {
        "command": "character-prepare",
        "frame_mode": _provenance(frame_mode, FROM_CLI, "never inferred from filenames"),
        "preset": _provenance(
            None if campaign_preset is None else f"{preset}/{campaign_preset.campaign}",
            FROM_CLI if campaign_preset is not None else "none",
            "declared campaign registry entry"
            if campaign_preset is not None
            else "no preset was given",
        ),
        "shprop": {
            "files": [str(p) for p in survey.paths],
            "n_files": len(survey.paths),
            "rows": survey.n_rows,
            "columns": survey.n_columns,
            "bytes": [r.bytes_on_disk for r in survey.structures],
            "BMIN": survey.bmin,
            "BMAX": survey.bmax,
            "basis_size": survey.n_states,
            "NAMDTINI": survey.namdtini,
            "NSW": survey.nsw,
            "header_derived_periods": survey.header_periods,
            "read_note": (
                "headers, shapes and a bounded sample of rows only; no SHPROP table "
                "was materialized"
            ),
        },
        "population_columns": _provenance(
            list(columns), FROM_SHPROP, "one contiguous block survived the value test"
        ),
        "population_column_rationale": column_rationale,
        "time_column": _provenance(
            time_column, FROM_SHPROP, "the column that increases strictly"
        ),
        "state_map": {
            "source": state_map_source,
            "detail": state_map_detail,
            "payload": state_map_payload,
        },
        "atom_groups": {
            "source": atom_source,
            "detail": f"partition taken from {atom_origin} and checked against the PROCAR",
            "validation": atom_validation,
            "payload": atom_payload,
        },
        "structure_file": structure_info,
        "projection": discovery.summary(),
        "representative_procar": procar,
        "cycle_length": cycle,
        "manifest": {
            "source": FROM_DIRECTORY,
            "detail": (
                "pattern generated from a regular zero-padded tree"
                if "procar_pattern" in manifest_payload
                else "explicit per-frame entries because the tree is irregular"
            ),
            "payload_keys": sorted(manifest_payload),
            "n_frames": discovery.count,
        },
        "slurm": None
        if sbatch_path is None
        else {
            "source": FROM_CLI,
            "account": (slurm or SlurmOptions()).account,
            "partition": (slurm or SlurmOptions()).partition,
            "nodes": (slurm or SlurmOptions()).nodes,
            "tasks": (slurm or SlurmOptions()).tasks,
            "cpus": (slurm or SlurmOptions()).cpus,
            "memory": (slurm or SlurmOptions()).memory,
            "time": (slurm or SlurmOptions()).time,
            "detail": "a starting proposal; nothing here was measured",
        },
        "generated_files": [str(p) for p in written],
        "warnings": warnings,
        "unresolved_questions": unresolved,
        "ok": not unresolved,
        "refusals": [
            "subsystem names and atom boundaries are never inferred from a structure",
            "a population column block is accepted only when exactly one reading fits",
            "frame mode is never inferred from filenames or directory layout",
            "a cycle length is never chosen when directory coverage and NSW-1 disagree",
        ],
    }
    return PreparedCampaign(
        report=report,
        written=written,
        state_map_path=state_map_path,
        atom_groups_path=atom_groups_path,
        manifest_path=manifest_path,
        sbatch_path=sbatch_path,
        unresolved=unresolved,
    )
