"""Resolve provenance for archived Hefei-NAMD SHPROP files.

Real production SHPROP files are often plain numeric tables.  In particular,
BMIN/BMAX are properties of the NAMD input/basis, not guaranteed SHPROP
header fields, and NAMDTINI may only survive in the historical SHPROP filename
(``SHPROP.<start-frame>``).  This module keeps those independent provenance
sources explicit and adapts the character workflow without inventing metadata.

Precedence used here:

* basis band numbers: explicit ``state_map.json:band_numbers`` > registered
  campaign basis > optional agreeing SHPROP BMIN/BMAX metadata;
* NAMDTINI: optional SHPROP metadata > validated ``SHPROP.<integer>`` suffix;
* cyclic period: explicit projection-manifest ``cycle_length`` > optional SHPROP
  ``NSW - 1`` metadata.

Disagreement is an error or a reported mismatch; values are never silently
reconciled.
"""

from __future__ import annotations

import json
import re
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import character, prepare
from .io.hefei import ShpropStructure, shprop_structure
from .populations import StateMap


# Basis provenance established from the original campaign setup.  These are
# VASP band numbers, ordered exactly like population_columns.
CAMPAIGN_BANDS: Dict[Tuple[str, str], List[int]] = {
    ("bcf_pcbm", "A"): [976, 977, 978, 979, 980, 981],
}
CAMPAIGN_NAME_TO_BANDS: Dict[str, Tuple[List[int], str]] = {
    "FAPI_001_BCF_PCBM_A": (
        CAMPAIGN_BANDS[("bcf_pcbm", "A")],
        "preset_bcf_pcbm_A",
    )
}

_SHPROP_SUFFIX = re.compile(r"^SHPROP\.(\d+)$", re.IGNORECASE)
_PREPARE_PRESET: ContextVar[Optional[Tuple[str, Optional[str]]]] = ContextVar(
    "namd_analysis_prepare_preset", default=None
)
_INSTALLED = False

# Original callables captured once by install().
_ORIGINAL_STATE_MAP_FROM_DICT = None
_ORIGINAL_STATE_MAP_AS_DICT = None
_ORIGINAL_PREFLIGHT = None
_ORIGINAL_PREPARE_CAMPAIGN = None


def _integer(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _validate_band_numbers(values: Sequence[Any], nstates: int, source: str) -> List[int]:
    bands: List[int] = []
    for value in values:
        number = _integer(value)
        if number is None or number <= 0:
            raise character.CharacterError(
                f"{source}: band_numbers must contain positive integer VASP band numbers"
            )
        bands.append(number)
    if len(bands) != nstates:
        raise character.CharacterError(
            f"{source}: band_numbers has {len(bands)} entries but the state map has "
            f"{nstates} population columns"
        )
    if len(set(bands)) != len(bands):
        raise character.CharacterError(f"{source}: band_numbers contains duplicates")
    return bands


def resolve_band_numbers(
    state_map: StateMap, records: Sequence[ShpropStructure]
) -> Tuple[List[int], str]:
    """Resolve exact VASP band numbers without requiring SHPROP BMIN/BMAX."""
    explicit = getattr(state_map, "band_numbers", None)
    if explicit is not None:
        return (
            _validate_band_numbers(
                explicit, len(state_map.population_columns), "state_map.json"
            ),
            "state_map.band_numbers",
        )

    registered = CAMPAIGN_NAME_TO_BANDS.get(state_map.name)
    if registered is not None:
        bands, source = registered
        return (
            _validate_band_numbers(bands, len(state_map.population_columns), source),
            source,
        )

    windows: Dict[Tuple[int, int], List[str]] = {}
    missing: List[str] = []
    for record in records:
        lo = _integer(record.metadata.get("BMIN"))
        hi = _integer(record.metadata.get("BMAX"))
        if lo is None or hi is None:
            missing.append(record.path.name)
            continue
        if hi < lo:
            raise character.CharacterError(
                f"{record.path}: optional SHPROP metadata has BMAX={hi} < BMIN={lo}"
            )
        windows.setdefault((lo, hi), []).append(record.path.name)
    if not missing and len(windows) == 1:
        lo, hi = next(iter(windows))
        bands = list(range(lo, hi + 1))
        return (
            _validate_band_numbers(
                bands, len(state_map.population_columns), "SHPROP BMIN/BMAX metadata"
            ),
            "SHPROP_BMIN_BMAX_metadata",
        )
    if len(windows) > 1:
        raise character.CharacterError(
            "SHPROP files contain conflicting optional BMIN/BMAX metadata; "
            "declare band_numbers explicitly in the state map"
        )
    raise character.CharacterError(
        "VASP band numbers are not present in these SHPROP tables. Add explicit "
        "'band_numbers' to state_map.json (one per population column), use a "
        "registered campaign preset, or supply SHPROP files whose optional metadata "
        "contains agreeing BMIN/BMAX. BMIN/BMAX are not assumed to be SHPROP headers."
    )


def resolve_namdtini(record: ShpropStructure) -> Tuple[int, str]:
    """Resolve one history's sampling origin, accepting the legacy filename."""
    header = _integer(record.metadata.get("NAMDTINI"))
    match = _SHPROP_SUFFIX.match(record.path.name)
    filename = int(match.group(1)) if match else None

    if header is not None:
        if header < 1:
            raise character.CharacterError(
                f"{record.path}: NAMDTINI={header} is not a positive one-based frame"
            )
        if filename is not None and filename != header:
            raise character.CharacterError(
                f"{record.path}: SHPROP metadata says NAMDTINI={header}, but the "
                f"validated filename suffix says {filename}. The two provenance "
                "sources disagree, so frame alignment is refused."
            )
        return header, "SHPROP_NAMDTINI_metadata"

    if filename is not None and filename >= 1:
        return filename, "SHPROP_filename_suffix"

    raise character.CharacterError(
        f"{record.path}: NAMDTINI is absent from SHPROP metadata and the filename is "
        "not of the validated form SHPROP.<positive integer>. Supply an original "
        "history with its start-frame filename or add explicit provenance upstream."
    )


def _resolve_period(
    record: ShpropStructure,
    manifest: character.ProjectionManifest,
    frame_mode: str,
) -> Tuple[Optional[int], str, Optional[int]]:
    raw_nsw = _integer(record.metadata.get("NSW"))
    header_period = raw_nsw - 1 if raw_nsw is not None else None
    if frame_mode != "dish-cyclic":
        return None, "not_applicable", header_period
    if manifest.cycle_length is not None:
        return int(manifest.cycle_length), "projection_manifest", header_period
    if header_period is not None and header_period > 0:
        return header_period, "SHPROP_NSW_minus_1_metadata", header_period
    raise character.CharacterError(
        f"{record.path}: cyclic alignment needs a period, but the projection manifest "
        "has no cycle_length and this SHPROP has no usable optional NSW metadata. "
        "Set cycle_length explicitly in projection_manifest.json."
    )


def _frames(start: int, ntime: int, mode: str, period: Optional[int]) -> np.ndarray:
    tion = np.arange(1, ntime + 1, dtype=int)
    if mode == "linear":
        return start + tion - 1
    if mode == "dish-cyclic":
        if period is None or period <= 0:
            raise character.CharacterError("cyclic frame period must be positive")
        values = np.mod(tion + start - 1, period)
        values[values == 0] = period
        return values
    raise character.CharacterError(f"unknown frame alignment mode {mode!r}")


def plan_analysis(
    shprop_paths: Sequence,
    state_map: StateMap,
    projection_manifest,
    frame_mode: str,
) -> character.AnalysisPlan:
    """Real-archive planner with explicit provenance fallbacks."""
    paths = [Path(path) for path in shprop_paths]
    if not paths:
        raise character.CharacterError("no SHPROP files were supplied")
    try:
        state_map.validate()
    except ValueError as exc:
        raise character.CharacterError(str(exc)) from exc
    if len({path.resolve() for path in paths}) != len(paths):
        raise character.CharacterError("duplicate population files")

    records = [shprop_structure(path) for path in paths]
    widths = {record.n_columns for record in records}
    heights = {record.n_rows for record in records}
    if len(widths) != 1:
        raise character.CharacterError(
            "SHPROP files have different column counts; no interpolation or truncation is performed"
        )
    if len(heights) != 1:
        raise character.CharacterError(
            "SHPROP files have different row counts; no interpolation or truncation is performed"
        )
    ncolumns = next(iter(widths))
    ntime = next(iter(heights))
    if ntime < 2:
        raise character.CharacterError(
            "population time must strictly increase with at least two samples"
        )
    needed = max([state_map.time_column] + list(state_map.population_columns))
    if needed >= ncolumns:
        raise character.CharacterError(
            f"configuration references column {needed} but the SHPROP files have {ncolumns} columns"
        )

    bands, band_source = resolve_band_numbers(state_map, records)
    manifest = character._load_projection_manifest(projection_manifest)
    manifest_frames = sorted(frame for frame, _ in manifest.frames)

    frames_by_file: List[np.ndarray] = []
    alignments: List[Dict[str, Any]] = []
    n_time: List[int] = []
    periods_used: List[Optional[int]] = []

    for record in records:
        start, start_source = resolve_namdtini(record)
        period, period_source, header_period = _resolve_period(record, manifest, frame_mode)
        n_time.append(record.n_rows)
        values = _frames(start, record.n_rows, frame_mode, period)
        frames_by_file.append(values)
        periods_used.append(period)

        wraps = int(np.count_nonzero(values[1:] != values[:-1] + 1))
        alignments.append(
            {
                "path": str(record.path.resolve()),
                "file": record.path.name,
                "NAMDTINI": start,
                "NAMDTINI_source": start_source,
                "NSW": record.metadata.get("NSW"),
                "n_time_points": record.n_rows,
                "header_cycle_length": header_period,
                "cycle_length_used": period if frame_mode == "dish-cyclic" else None,
                "cycle_length_source": period_source,
                "header_cycle_mismatch": bool(
                    frame_mode == "dish-cyclic"
                    and header_period is not None
                    and period is not None
                    and header_period != period
                ),
                "BMIN": bands[0],
                "BMAX": bands[-1],
                "band_numbers": list(bands),
                "band_numbers_source": band_source,
                "frame_mode": frame_mode,
                "first_projection_frame": int(values[0]),
                "last_projection_frame": int(values[-1]),
                "first_five_frames": [int(v) for v in values[:5]],
                "last_five_frames": [int(v) for v in values[-5:]],
                "unique_projection_frames_used": int(len(set(int(v) for v in values))),
                "wrap_count": wraps,
            }
        )

    if frame_mode == "dish-cyclic" and len(set(periods_used)) > 1:
        raise character.CharacterError(
            "the SHPROP histories resolve to different cyclic periods; split the "
            "campaign or declare one explicit projection-manifest cycle_length"
        )

    required = sorted({int(frame) for values in frames_by_file for frame in values})
    return character.AnalysisPlan(
        shprop_paths=paths,
        manifest=manifest,
        bmin=bands[0],
        bmax=bands[-1],
        bands=bands,
        frame_mode=frame_mode,
        frames_by_file=frames_by_file,
        required_frames=required,
        alignments=alignments,
        n_time=n_time,
        manifest_frames=manifest_frames,
        shprop_structures=records,
    )


def preflight_report(*args, **kwargs):
    """Run the ordinary preflight, then expose the provenance it resolved."""
    report = _ORIGINAL_PREFLIGHT(*args, **kwargs)
    if not report.get("ok") and report.get("stage") == "planning":
        return report
    state_map = args[1] if len(args) > 1 else kwargs.get("state_map")
    if state_map is not None and report.get("band_window"):
        registered = CAMPAIGN_NAME_TO_BANDS.get(state_map.name)
        explicit = getattr(state_map, "band_numbers", None)
        if explicit is not None:
            report["band_window"]["band_numbers"] = list(explicit)
            report["band_window"]["source"] = "state_map.band_numbers"
        elif registered is not None:
            report["band_window"]["band_numbers"] = list(registered[0])
            report["band_window"]["source"] = registered[1]
    for row in report.get("shprop", []):
        if row.get("NAMDTINI") is not None:
            match = _SHPROP_SUFFIX.match(str(row.get("file", "")))
            row["NAMDTINI_source"] = (
                "SHPROP_filename_suffix" if match else "SHPROP_NAMDTINI_metadata"
            )
    return report


def _survey_shprop(paths: Sequence) -> prepare.ShpropSurvey:
    """Preparation survey that treats basis provenance as independent of SHPROP."""
    paths = [Path(path) for path in paths]
    if not paths:
        raise prepare.PrepareError("no SHPROP files were supplied")
    structures = [shprop_structure(path) for path in paths]

    widths = {record.n_columns for record in structures}
    heights = {record.n_rows for record in structures}
    if len(widths) != 1:
        raise prepare.PrepareError("SHPROP files have different column counts")
    if len(heights) != 1:
        raise prepare.PrepareError("SHPROP files have different row counts")

    context = _PREPARE_PRESET.get()
    bands: Optional[List[int]] = None
    if context is not None:
        bands = CAMPAIGN_BANDS.get((context[0], context[1] or ""))
    if bands is None:
        windows = []
        for record in structures:
            lo = _integer(record.metadata.get("BMIN"))
            hi = _integer(record.metadata.get("BMAX"))
            if lo is None or hi is None:
                windows = []
                break
            windows.append((lo, hi))
        if windows and len(set(windows)) == 1:
            lo, hi = windows[0]
            bands = list(range(lo, hi + 1))
    if not bands:
        raise prepare.PrepareError(
            "SHPROP does not contain BMIN/BMAX, which is allowed. Preparation now "
            "needs a registered campaign preset with known band_numbers (or SHPROP "
            "files carrying optional agreeing BMIN/BMAX metadata)."
        )

    starts: List[int] = []
    for record in structures:
        try:
            start, _source = resolve_namdtini(record)
        except character.CharacterError as exc:
            raise prepare.PrepareError(str(exc)) from exc
        starts.append(start)

    nsw: List[Optional[int]] = []
    for record in structures:
        value = _integer(record.metadata.get("NSW"))
        nsw.append(value)
    periods = sorted({value - 1 for value in nsw if value is not None})

    return prepare.ShpropSurvey(
        structures=structures,
        bmin=bands[0],
        bmax=bands[-1],
        n_states=len(bands),
        n_columns=next(iter(widths)),
        n_rows=next(iter(heights)),
        namdtini=starts,
        nsw=nsw,
        header_periods=periods,
    )


def _prepare_campaign(*args, **kwargs):
    preset = kwargs.get("preset")
    campaign = kwargs.get("campaign")
    token = _PREPARE_PRESET.set((preset, campaign) if preset is not None else None)
    try:
        prepared = _ORIGINAL_PREPARE_CAMPAIGN(*args, **kwargs)
    finally:
        _PREPARE_PRESET.reset(token)

    bands = CAMPAIGN_BANDS.get((preset, campaign)) if preset is not None else None
    if bands:
        payload = json.loads(prepared.state_map_path.read_text(encoding="utf-8"))
        payload["band_numbers"] = list(bands)
        prepared.state_map_path.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        prepared.report["band_numbers"] = {
            "value": list(bands),
            "source": f"preset_{preset}_{campaign}",
            "detail": (
                "exact VASP band numbers from the registered campaign provenance; "
                "they are not inferred from SHPROP"
            ),
        }
        shprop_report = prepared.report.get("shprop")
        if isinstance(shprop_report, dict):
            shprop_report["basis_source"] = f"preset_{preset}_{campaign}"
            shprop_report["NAMDTINI_source"] = (
                "optional SHPROP metadata, otherwise validated SHPROP.<integer> filename"
            )
    return prepared


def _state_map_from_dict(cls, payload: Mapping[str, Any]):
    state_map = _ORIGINAL_STATE_MAP_FROM_DICT(cls, dict(payload))
    raw = payload.get("band_numbers")
    if raw is not None:
        state_map.band_numbers = _validate_band_numbers(
            raw, len(state_map.population_columns), "state_map.json"
        )
    return state_map


def _state_map_as_dict(self):
    payload = _ORIGINAL_STATE_MAP_AS_DICT(self)
    bands = getattr(self, "band_numbers", None)
    if bands is not None:
        payload["band_numbers"] = list(bands)
    return payload


def install() -> None:
    """Install provenance-aware adapters before CLI modules bind callables."""
    global _INSTALLED
    global _ORIGINAL_STATE_MAP_FROM_DICT, _ORIGINAL_STATE_MAP_AS_DICT
    global _ORIGINAL_PREFLIGHT, _ORIGINAL_PREPARE_CAMPAIGN
    if _INSTALLED:
        return

    _ORIGINAL_STATE_MAP_FROM_DICT = StateMap.from_dict.__func__
    _ORIGINAL_STATE_MAP_AS_DICT = StateMap.as_dict
    _ORIGINAL_PREFLIGHT = character.preflight_report
    _ORIGINAL_PREPARE_CAMPAIGN = prepare.prepare_campaign

    StateMap.from_dict = classmethod(_state_map_from_dict)
    StateMap.as_dict = _state_map_as_dict

    character.plan_analysis = plan_analysis
    character.preflight_report = preflight_report
    prepare.survey_shprop = _survey_shprop
    prepare.prepare_campaign = _prepare_campaign
    _INSTALLED = True
