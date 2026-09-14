"""Frame-dependent physical-state character from PROCAR projections.

An adiabatic band index can exchange BCF/PCBM/perovskite character during an
MD trajectory.  Projection-weighted populations therefore combine every
*original* SHPROP history with the PROCAR frame that actually generated that
NAMD step, and only then average across SHPROP files.

The implementation is deliberately strict: no nearest-energy band tracking,
no inferred frame alignment, and no implicit k-point/spin averaging.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .io.hefei import read_shprop_with_metadata
from .io.procar import ProcarFormatError, read_procar_ion_totals
from .populations import PopulationSet, StateMap, load_population_set


class CharacterError(ValueError):
    """Raised when state-character analysis would require guessing."""


@dataclass
class AtomGroupMap:
    """One-based PROCAR ion indices assigned to physical subsystems."""

    groups: Dict[str, List[int]]
    complete_atoms: bool = True
    min_projection_weight: float = 0.5

    @classmethod
    def from_json(cls, path) -> "AtomGroupMap":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        raw = payload.get("groups")
        if not isinstance(raw, dict) or not raw:
            raise CharacterError("atom-group JSON needs a nonempty 'groups' object")
        complete = payload.get("complete_atoms", True)
        if type(complete) is not bool:
            raise CharacterError("complete_atoms must be a JSON boolean")
        groups = {str(name): _expand_atom_spec(spec) for name, spec in raw.items()}
        obj = cls(
            groups=groups,
            complete_atoms=complete,
            min_projection_weight=float(payload.get("min_projection_weight", 0.5)),
        )
        obj.validate()
        return obj

    def validate(self) -> None:
        if not (0.0 <= self.min_projection_weight <= 2.0):
            raise CharacterError("min_projection_weight must lie between 0 and 2")
        seen: Dict[int, str] = {}
        for name, atoms in self.groups.items():
            if not atoms:
                raise CharacterError(f"atom group {name!r} is empty")
            for atom in atoms:
                if atom <= 0:
                    raise CharacterError("PROCAR ion indices are one-based positive integers")
                if atom in seen:
                    raise CharacterError(
                        f"PROCAR ion {atom} appears in both {seen[atom]!r} and {name!r}"
                    )
                seen[atom] = name

    @property
    def names(self) -> List[str]:
        return list(self.groups)


@dataclass
class ProjectionManifest:
    """Resolved mapping between engine frame numbers and PROCAR files."""

    frames: List[Tuple[int, Path]]
    cycle_length: Optional[int] = None


@dataclass
class ProjectionSeries:
    """Normalized subsystem character for selected bands over MD frames."""

    frames: np.ndarray
    bands: np.ndarray
    group_names: List[str]
    weights: np.ndarray  # (nframe, nband, ngroup), normalized over declared groups
    captured_projection: np.ndarray  # raw sum over declared groups
    total_projection: np.ndarray  # raw sum over every PROCAR ion
    source_paths: List[Path]
    quality_threshold: float
    cycle_length: Optional[int] = None

    def frame_index(self) -> Dict[int, int]:
        return {int(frame): i for i, frame in enumerate(self.frames)}

    def band_index(self) -> Dict[int, int]:
        return {int(band): i for i, band in enumerate(self.bands)}

    def quality_summary(self) -> Dict[str, Any]:
        captured = self.captured_projection
        return {
            "minimum_captured_projection": float(np.min(captured)),
            "median_captured_projection": float(np.median(captured)),
            "maximum_captured_projection": float(np.max(captured)),
            "quality_threshold": self.quality_threshold,
            "samples_below_threshold": int(np.count_nonzero(captured < self.quality_threshold)),
            "fraction_below_threshold": float(np.mean(captured < self.quality_threshold)),
        }


@dataclass
class CharacterPopulationResult:
    """Projection-weighted population ensemble and alignment diagnostics."""

    time_ns: np.ndarray
    group_names: List[str]
    per_file: np.ndarray  # (nfiles, ntime, ngroups)
    mean: np.ndarray
    sem: Optional[np.ndarray]
    fixed_mean: Dict[str, np.ndarray]
    projection: ProjectionSeries
    file_alignment: List[Dict[str, Any]]
    conservation: Dict[str, float]


def _expand_atom_spec(spec: Any) -> List[int]:
    if not isinstance(spec, list):
        raise CharacterError("each atom group must be a JSON list of integers/ranges")
    atoms: List[int] = []
    for item in spec:
        if type(item) is int:
            atoms.append(item)
            continue
        if isinstance(item, str):
            text = item.strip()
            if "-" in text:
                left, right = text.split("-", 1)
                try:
                    start, stop = int(left), int(right)
                except ValueError as exc:
                    raise CharacterError(f"invalid atom range {item!r}") from exc
                if stop < start:
                    raise CharacterError(f"atom range decreases: {item!r}")
                atoms.extend(range(start, stop + 1))
                continue
            try:
                atoms.append(int(text))
                continue
            except ValueError as exc:
                raise CharacterError(f"invalid atom selector {item!r}") from exc
        raise CharacterError(f"invalid atom selector {item!r}")
    if len(set(atoms)) != len(atoms):
        raise CharacterError("an atom group contains duplicate ion indices")
    return atoms


def _load_projection_manifest(path) -> ProjectionManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    entries: List[Tuple[int, Path]] = []

    raw_cycle = payload.get("cycle_length")
    cycle_length = None
    if raw_cycle is not None:
        if type(raw_cycle) is not int or raw_cycle <= 0:
            raise CharacterError("projection manifest cycle_length must be a positive integer")
        cycle_length = raw_cycle

    if "frames" in payload:
        if not isinstance(payload["frames"], list) or not payload["frames"]:
            raise CharacterError("projection manifest 'frames' must be a nonempty list")
        for record in payload["frames"]:
            try:
                frame = int(record["frame"])
                raw_path = str(record["procar"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CharacterError("each projection frame needs integer frame and procar") from exc
            procar = Path(raw_path)
            if not procar.is_absolute():
                procar = base / procar
            entries.append((frame, procar))
    elif "procar_pattern" in payload:
        try:
            first = int(payload["first_frame"])
            last = int(payload["last_frame"])
            step = int(payload.get("frame_step", 1))
            pattern = str(payload["procar_pattern"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CharacterError(
                "pattern manifests need procar_pattern, first_frame and last_frame"
            ) from exc
        if step <= 0 or last < first:
            raise CharacterError("projection manifest frame range is invalid")
        for frame in range(first, last + 1, step):
            procar = Path(pattern.format(frame=frame))
            if not procar.is_absolute():
                procar = base / procar
            entries.append((frame, procar))
    else:
        raise CharacterError("projection manifest needs either 'frames' or 'procar_pattern'")

    frames = [frame for frame, _ in entries]
    if any(frame <= 0 for frame in frames):
        raise CharacterError("projection frame numbers must be positive")
    if len(set(frames)) != len(frames):
        raise CharacterError("projection manifest contains duplicate frame numbers")
    entries.sort(key=lambda item: item[0])
    missing = [str(p) for _, p in entries if not p.is_file()]
    if missing:
        preview = ", ".join(missing[:3])
        more = "" if len(missing) <= 3 else f" (+{len(missing) - 3} more)"
        raise CharacterError(f"projection manifest references missing PROCAR files: {preview}{more}")

    if cycle_length is not None:
        available = set(frames)
        required_cycle = set(range(1, cycle_length + 1))
        missing_cycle = sorted(required_cycle - available)
        if missing_cycle:
            preview = missing_cycle[:10]
            raise CharacterError(
                f"cycle_length={cycle_length} but the manifest lacks cycle frames "
                f"{preview}{'...' if len(missing_cycle) > 10 else ''}"
            )
    return ProjectionManifest(frames=entries, cycle_length=cycle_length)


def load_projection_series(
    manifest_path,
    atom_groups: AtomGroupMap,
    required_bands: Sequence[int],
) -> ProjectionSeries:
    """Load subsystem weights for exactly the bands needed by the NAMD basis."""
    required = [int(band) for band in required_bands]
    if not required or len(set(required)) != len(required):
        raise CharacterError("required band list must be nonempty and unique")
    manifest = _load_projection_manifest(manifest_path)
    entries = manifest.frames
    group_names = atom_groups.names
    frame_weights: List[np.ndarray] = []
    captured_rows: List[np.ndarray] = []
    total_rows: List[np.ndarray] = []
    nions_expected: Optional[int] = None

    for _frame, path in entries:
        try:
            projection = read_procar_ion_totals(path)
        except ProcarFormatError as exc:
            raise CharacterError(str(exc)) from exc
        if nions_expected is None:
            nions_expected = projection.nions
            flat_atoms = sorted(atom for atoms in atom_groups.groups.values() for atom in atoms)
            if flat_atoms and flat_atoms[-1] > nions_expected:
                raise CharacterError(
                    f"atom map references ion {flat_atoms[-1]} but PROCAR has {nions_expected} ions"
                )
            if atom_groups.complete_atoms and flat_atoms != list(range(1, nions_expected + 1)):
                raise CharacterError(
                    "complete_atoms: true requires every PROCAR ion 1..N exactly once across groups"
                )
        elif projection.nions != nions_expected:
            raise CharacterError(
                f"{path}: {projection.nions} ions but earlier PROCARs have {nions_expected}"
            )

        band_lookup = projection.band_index()
        missing_bands = [band for band in required if band not in band_lookup]
        if missing_bands:
            raise CharacterError(f"{path}: missing required VASP bands {missing_bands}")

        raw = np.empty((len(required), len(group_names)), dtype=float)
        total = np.empty(len(required), dtype=float)
        for bi, band in enumerate(required):
            ion_values = projection.ion_totals[band_lookup[band]]
            total[bi] = float(np.sum(ion_values))
            for gi, group in enumerate(group_names):
                indices = np.asarray(atom_groups.groups[group], dtype=int) - 1
                raw[bi, gi] = float(np.sum(ion_values[indices]))
        captured = raw.sum(axis=1)
        if np.any(captured <= 1.0e-14):
            bad = [required[i] for i in np.flatnonzero(captured <= 1.0e-14)]
            raise CharacterError(
                f"{path}: declared subsystems capture zero projection for bands {bad}; "
                "normalizing their character would be meaningless"
            )
        frame_weights.append(raw / captured[:, None])
        captured_rows.append(captured)
        total_rows.append(total)

    return ProjectionSeries(
        frames=np.asarray([frame for frame, _ in entries], dtype=int),
        bands=np.asarray(required, dtype=int),
        group_names=group_names,
        weights=np.stack(frame_weights, axis=0),
        captured_projection=np.stack(captured_rows, axis=0),
        total_projection=np.stack(total_rows, axis=0),
        source_paths=[path for _, path in entries],
        quality_threshold=atom_groups.min_projection_weight,
        cycle_length=manifest.cycle_length,
    )


def _int_metadata(metadata: Mapping[str, Any], key: str, path: Path) -> int:
    value = metadata.get(key)
    if type(value) is int:
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise CharacterError(
        f"{path}: SHPROP header lacks integer {key}; frame-dependent character "
        "cannot be aligned by inference"
    )


def aligned_frames(
    metadata: Mapping[str, Any],
    ntime: int,
    mode: str,
    path: Path,
    cycle_length: Optional[int] = None,
) -> np.ndarray:
    """Return the electronic-structure frame used at each NAMD time point.

    For ``dish-cyclic`` an explicit projection-manifest ``cycle_length`` wins
    over ``NSW - 1`` from the SHPROP header.  This is necessary for archived
    campaigns whose engine/input bookkeeping does not exactly match the number
    of saved electronic frames.  The chosen period is recorded in provenance.
    """
    start = _int_metadata(metadata, "NAMDTINI", path)
    tion = np.arange(1, ntime + 1, dtype=int)
    if mode == "linear":
        return start + tion - 1
    if mode == "dish-cyclic":
        if cycle_length is None:
            nsw = _int_metadata(metadata, "NSW", path)
            period = nsw - 1
        else:
            period = cycle_length
        if period <= 0:
            raise CharacterError(f"{path}: cyclic frame period must be positive")
        frames = np.mod(tion + start - 1, period)
        frames[frames == 0] = period
        return frames
    raise CharacterError(f"unknown frame alignment mode {mode!r}")


def character_populations(
    shprop_paths: Sequence,
    state_map: StateMap,
    projection_manifest,
    atom_groups: AtomGroupMap,
    frame_mode: str,
) -> CharacterPopulationResult:
    """Project each original SHPROP into physical character before averaging.

    This ordering is essential.  Different SHPROP histories may have different
    NAMDTINI values, so averaging their adiabatic populations first destroys
    the phase needed to select the correct PROCAR frame.
    """
    paths = [Path(path) for path in shprop_paths]
    population: PopulationSet = load_population_set(paths, state_map)
    records = [read_shprop_with_metadata(path) for path in paths]

    band_windows: List[Tuple[int, int]] = []
    for record in records:
        bmin = _int_metadata(record.metadata, "BMIN", record.path)
        bmax = _int_metadata(record.metadata, "BMAX", record.path)
        if bmax < bmin:
            raise CharacterError(f"{record.path}: BMAX < BMIN")
        band_windows.append((bmin, bmax))
    if len(set(band_windows)) != 1:
        raise CharacterError(f"SHPROP files use different BMIN/BMAX windows: {band_windows}")
    bmin, bmax = band_windows[0]
    bands = list(range(bmin, bmax + 1))
    if len(bands) != len(state_map.population_columns):
        raise CharacterError(
            f"SHPROP basis {bmin}:{bmax} has {len(bands)} states but state map "
            f"declares {len(state_map.population_columns)} population columns. "
            "Character analysis requires one declared population column per basis "
            "state, ordered from BMIN through BMAX."
        )

    projection = load_projection_series(projection_manifest, atom_groups, bands)
    frame_lookup = projection.frame_index()
    band_lookup = projection.band_index()
    projection_band_indices = np.asarray([band_lookup[band] for band in bands], dtype=int)

    physical_files: List[np.ndarray] = []
    alignments: List[Dict[str, Any]] = []
    for record in records:
        frames = aligned_frames(
            record.metadata,
            record.table.shape[0],
            frame_mode,
            record.path,
            cycle_length=projection.cycle_length,
        )
        missing = sorted({int(frame) for frame in frames if int(frame) not in frame_lookup})
        if missing:
            preview = missing[:10]
            raise CharacterError(
                f"{record.path}: projection manifest lacks {len(missing)} required frames "
                f"{preview}{'...' if len(missing) > 10 else ''}"
            )
        frame_indices = np.asarray([frame_lookup[int(frame)] for frame in frames], dtype=int)
        weights = projection.weights[frame_indices][:, projection_band_indices, :]
        pops = record.table[:, state_map.population_columns]
        physical = np.einsum("ts,tsg->tg", pops, weights)
        physical_files.append(physical)

        header_nsw = record.metadata.get("NSW")
        header_period = None
        if isinstance(header_nsw, (int, float)):
            header_period = int(header_nsw) - 1
        cycle_used = projection.cycle_length if projection.cycle_length is not None else header_period
        alignments.append(
            {
                "path": str(record.path.resolve()),
                "NAMDTINI": _int_metadata(record.metadata, "NAMDTINI", record.path),
                "NSW": header_nsw,
                "header_cycle_length": header_period,
                "cycle_length_used": cycle_used if frame_mode == "dish-cyclic" else None,
                "cycle_length_source": (
                    "projection_manifest"
                    if frame_mode == "dish-cyclic" and projection.cycle_length is not None
                    else "SHPROP_NSW_minus_1"
                    if frame_mode == "dish-cyclic"
                    else "not_applicable"
                ),
                "header_cycle_mismatch": bool(
                    frame_mode == "dish-cyclic"
                    and projection.cycle_length is not None
                    and header_period is not None
                    and projection.cycle_length != header_period
                ),
                "BMIN": bmin,
                "BMAX": bmax,
                "frame_mode": frame_mode,
                "first_projection_frame": int(frames[0]),
                "last_projection_frame": int(frames[-1]),
                "unique_projection_frames_used": int(np.unique(frames).size),
            }
        )

    per_file = np.stack(physical_files, axis=0)
    mean = per_file.mean(axis=0)
    sem = None
    if len(paths) > 1:
        sem = per_file.std(axis=0, ddof=1) / np.sqrt(len(paths))

    fixed_mean: Dict[str, np.ndarray] = {}
    for name, columns in state_map.groups.items():
        fixed_mean[name] = population.mean[:, columns].sum(axis=1)

    total = mean.sum(axis=1)
    conservation = {
        "min": float(np.min(total)),
        "max": float(np.max(total)),
        "mean": float(np.mean(total)),
        "max_abs_deviation_from_one": float(np.max(np.abs(total - 1.0))),
    }
    if state_map.complete_population and conservation["max_abs_deviation_from_one"] > 1.0e-5:
        raise CharacterError(
            "projection-weighted physical populations do not conserve the complete "
            "SHPROP population; check projection normalization and state alignment"
        )

    return CharacterPopulationResult(
        time_ns=population.time_ns,
        group_names=projection.group_names,
        per_file=per_file,
        mean=mean,
        sem=sem,
        fixed_mean=fixed_mean,
        projection=projection,
        file_alignment=alignments,
        conservation=conservation,
    )


def character_swap_rows(
    projection: ProjectionSeries, dominance_threshold: float = 0.6
) -> Tuple[List[List[Any]], Dict[str, Any]]:
    """Report dominant-subsystem changes along the underlying MD trajectory."""
    if not (0.0 < dominance_threshold <= 1.0):
        raise CharacterError("dominance_threshold must lie in (0, 1]")
    rows: List[List[Any]] = []
    mixed = 0
    total = projection.weights.shape[0] * projection.weights.shape[1]
    swap_count = 0
    for bi, band in enumerate(projection.bands):
        weights = projection.weights[:, bi, :]
        dominant = np.argmax(weights, axis=1)
        confidence = np.max(weights, axis=1)
        mixed += int(np.count_nonzero(confidence < dominance_threshold))
        for fi in range(1, len(projection.frames)):
            if dominant[fi] == dominant[fi - 1]:
                continue
            swap_count += 1
            before = int(dominant[fi - 1])
            after = int(dominant[fi])
            rows.append(
                [
                    int(band),
                    int(projection.frames[fi - 1]),
                    int(projection.frames[fi]),
                    projection.group_names[before],
                    projection.group_names[after],
                    float(confidence[fi - 1]),
                    float(confidence[fi]),
                ]
            )
    summary = {
        "dominance_threshold": dominance_threshold,
        "dominant_character_swaps": swap_count,
        "mixed_frame_band_samples": mixed,
        "total_frame_band_samples": total,
        "mixed_fraction": float(mixed / total) if total else 0.0,
    }
    return rows, summary


def projection_table_rows(projection: ProjectionSeries) -> Iterable[List[Any]]:
    """Long-form normalized and raw character table for auditing."""
    for fi, frame in enumerate(projection.frames):
        for bi, band in enumerate(projection.bands):
            dominant = int(np.argmax(projection.weights[fi, bi]))
            for gi, group in enumerate(projection.group_names):
                yield [
                    int(frame),
                    int(band),
                    group,
                    float(projection.weights[fi, bi, gi]),
                    float(projection.captured_projection[fi, bi]),
                    float(projection.total_projection[fi, bi]),
                    projection.group_names[dominant],
                    float(np.max(projection.weights[fi, bi])),
                ]
