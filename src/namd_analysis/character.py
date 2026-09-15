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
from .io.procar import (
    ProcarFormatError,
    procar_band_numbers,
    procar_structure,
    read_procar_ion_totals,
)
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
        payload = _load_json(Path(path), "atom-group map")
        raw = payload.get("groups")
        if not isinstance(raw, dict) or not raw:
            raise CharacterError(
                f"{path}: atom-group JSON needs a nonempty 'groups' object"
            )
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

    def quality_summary(self, worst_n: int = 10) -> Dict[str, Any]:
        """How much of each band actually lands inside the declared subsystems.

        The normalized weights divide by this, so a band whose captured
        projection is small has character that is mostly an artefact of the
        division.  These are diagnostics: nothing here discards, repairs or
        reweights a low-projection sample, because doing so silently would
        change the science.
        """
        captured = self.captured_projection
        threshold = self.quality_threshold
        below = captured < threshold
        nframes, nbands = captured.shape

        per_band: List[Dict[str, Any]] = []
        for bi in range(nbands):
            column = captured[:, bi]
            per_band.append(
                {
                    "band": int(self.bands[bi]),
                    "median_captured_projection": float(np.median(column)),
                    "minimum_captured_projection": float(np.min(column)),
                    "frame_of_minimum": int(self.frames[int(np.argmin(column))]),
                    "samples_below_threshold": int(np.count_nonzero(column < threshold)),
                    "fraction_below_threshold": float(np.mean(column < threshold)),
                }
            )

        per_frame_minimum = captured.min(axis=1)
        worst_frames = np.argsort(per_frame_minimum)[: max(0, worst_n)]

        flat = captured.reshape(-1)
        order = np.argsort(flat)[: max(0, worst_n)]
        worst_samples = []
        for position in order:
            fi, bi = divmod(int(position), nbands)
            worst_samples.append(
                {
                    "frame": int(self.frames[fi]),
                    "band": int(self.bands[bi]),
                    "captured_projection": float(captured[fi, bi]),
                    "total_projection": float(self.total_projection[fi, bi]),
                    "source": str(self.source_paths[fi]),
                }
            )

        return {
            "quality_threshold": threshold,
            "n_frames": int(nframes),
            "n_bands": int(nbands),
            "minimum_captured_projection": float(np.min(captured)),
            "p1_captured_projection": float(np.percentile(captured, 1)),
            "p5_captured_projection": float(np.percentile(captured, 5)),
            "median_captured_projection": float(np.median(captured)),
            "p95_captured_projection": float(np.percentile(captured, 95)),
            "maximum_captured_projection": float(np.max(captured)),
            "samples_below_threshold": int(np.count_nonzero(below)),
            "fraction_below_threshold": float(np.mean(below)),
            "per_band": per_band,
            "worst_frame_band_samples": worst_samples,
            "worst_frames_by_minimum_capture": [
                {
                    "frame": int(self.frames[int(fi)]),
                    "minimum_captured_projection": float(per_frame_minimum[int(fi)]),
                }
                for fi in worst_frames
            ],
            "note": (
                "captured_projection is the raw PROCAR weight falling inside the "
                "declared atom groups, before normalization. Values well below one "
                "mean the band lies largely outside every declared PAW sphere, so its "
                "normalized character is poorly determined. Nothing is discarded or "
                "repaired on this basis"
            ),
        }

    def dominance_summary(self, dominance_threshold: float = 0.6) -> Dict[str, Any]:
        """Per-band dominant-character occupancy and swap counts."""
        bands_out: List[Dict[str, Any]] = []
        for bi in range(self.weights.shape[1]):
            weights = self.weights[:, bi, :]
            dominant = np.argmax(weights, axis=1)
            confidence = np.max(weights, axis=1)
            occupancy = {
                name: float(np.mean(dominant == gi))
                for gi, name in enumerate(self.group_names)
            }
            swaps = int(np.count_nonzero(dominant[1:] != dominant[:-1]))
            bands_out.append(
                {
                    "band": int(self.bands[bi]),
                    "dominant_occupancy_fraction": occupancy,
                    "most_common_character": max(occupancy, key=occupancy.get),
                    "dominant_character_swaps": swaps,
                    "median_dominant_weight": float(np.median(confidence)),
                    "mixed_samples": int(np.count_nonzero(confidence < dominance_threshold)),
                    "mixed_fraction": float(np.mean(confidence < dominance_threshold)),
                }
            )
        return {"dominance_threshold": dominance_threshold, "per_band": bands_out}


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


def _load_json(path: Path, what: str) -> Any:
    """Read JSON, naming the file when it is malformed or unreadable."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise CharacterError(f"{path}: {what} could not be read ({exc})") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise CharacterError(
            f"{path}: {what} is not valid JSON - {exc.msg} at line {exc.lineno}, "
            f"column {exc.colno}"
        ) from exc


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
    payload = _load_json(manifest_path, "projection manifest")
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
                f"{manifest_path}: a pattern manifest needs procar_pattern, "
                f"first_frame and last_frame ({exc})"
            ) from exc
        if step <= 0 or last < first:
            raise CharacterError(
                f"{manifest_path}: frame range first_frame={first}, "
                f"last_frame={last}, frame_step={step} is empty or invalid; "
                "frame_step must be >= 1 and last_frame >= first_frame"
            )
        # Without a {frame} field every frame would resolve to the same file,
        # and the analysis would report constant character as though it were
        # frame-dependent.  That is silent and unrecoverable, so refuse it.
        if "{frame" not in pattern:
            raise CharacterError(
                f"{manifest_path}: procar_pattern {pattern!r} contains no "
                "'{frame}' field, so every frame would resolve to the same "
                "PROCAR and the character would be constant while appearing "
                "frame-dependent. Add {frame} (for example 'frames/{frame:04d}/PROCAR')."
            )
        for frame in range(first, last + 1, step):
            try:
                rendered = pattern.format(frame=frame)
            except (KeyError, IndexError, ValueError) as exc:
                raise CharacterError(
                    f"{manifest_path}: procar_pattern {pattern!r} could not be "
                    f"expanded for frame {frame}: {type(exc).__name__} {exc}. "
                    "The only field this package substitutes is {frame}."
                ) from exc
            procar = Path(rendered)
            if not procar.is_absolute():
                procar = base / procar
            entries.append((frame, procar))
    else:
        raise CharacterError(
            f"{manifest_path}: a projection manifest needs either a 'frames' list "
            "or a 'procar_pattern' with first_frame/last_frame"
        )

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
    required_frames: Optional[Sequence[int]] = None,
    manifest: Optional[ProjectionManifest] = None,
) -> ProjectionSeries:
    """Load subsystem weights for exactly the bands needed by the NAMD basis.

    ``required_frames`` restricts parsing to the electronic frames the supplied
    SHPROP histories actually visit.  A cyclic campaign can declare two thousand
    PROCAR files and touch only a fraction of them, and each one is large, so
    parsing the whole manifest to use part of it is the dominant cost.  The
    manifest is still validated in full -- every declared file must exist, frame
    numbers must be unique and positive, and a declared ``cycle_length`` must be
    completely covered -- because those checks are cheap and dropping them to
    save time would let a gap go unnoticed until it silently changed a number.

    ``manifest`` lets a caller that already parsed the manifest pass it in
    rather than re-reading it.
    """
    required = [int(band) for band in required_bands]
    if not required or len(set(required)) != len(required):
        raise CharacterError("required band list must be nonempty and unique")
    manifest = manifest if manifest is not None else _load_projection_manifest(manifest_path)
    entries = manifest.frames
    if required_frames is not None:
        wanted = {int(frame) for frame in required_frames}
        if not wanted:
            raise CharacterError("required frame list must be nonempty")
        available = {frame for frame, _ in entries}
        missing = sorted(wanted - available)
        if missing:
            preview = missing[:10]
            raise CharacterError(
                f"projection manifest lacks {len(missing)} required electronic frames "
                f"{preview}{'...' if len(missing) > 10 else ''}; the manifest covers "
                f"{min(available)}..{max(available)} ({len(available)} frames). "
                "Add the missing PROCAR files, or correct NAMDTINI/frame-mode/cycle_length."
            )
        entries = [(frame, path) for frame, path in entries if frame in wanted]
    group_names = atom_groups.names
    frame_weights: List[np.ndarray] = []
    captured_rows: List[np.ndarray] = []
    total_rows: List[np.ndarray] = []
    nions_expected: Optional[int] = None

    first_path: Optional[Path] = None
    for _frame, path in entries:
        try:
            projection = read_procar_ion_totals(path, bands=required)
        except ProcarFormatError as exc:
            raise CharacterError(str(exc)) from exc
        if nions_expected is None:
            nions_expected = projection.nions
            first_path = path
            flat_atoms = sorted(atom for atoms in atom_groups.groups.values() for atom in atoms)
            if flat_atoms and flat_atoms[-1] > nions_expected:
                raise CharacterError(
                    f"atom-group map references PROCAR ion {flat_atoms[-1]} but {path} "
                    f"has only {nions_expected} ions. Fix the atom-group JSON, or check "
                    "that it was written for this structure."
                )
            if atom_groups.complete_atoms:
                expected = list(range(1, nions_expected + 1))
                if flat_atoms != expected:
                    assigned = set(flat_atoms)
                    unassigned = sorted(set(expected) - assigned)
                    raise CharacterError(
                        "complete_atoms: true requires every PROCAR ion 1.."
                        f"{nions_expected} to appear exactly once across the atom "
                        f"groups. {path} has {nions_expected} ions; "
                        f"{len(assigned)} are assigned and {len(unassigned)} are not"
                        + (
                            f" (first missing: {unassigned[:10]}"
                            f"{'...' if len(unassigned) > 10 else ''})"
                            if unassigned
                            else ""
                        )
                        + ". Either complete the groups or set complete_atoms: false."
                    )
        elif projection.nions != nions_expected:
            raise CharacterError(
                f"{path}: {projection.nions} ions, but {first_path} has "
                f"{nions_expected}. Every PROCAR in one projection manifest must "
                "describe the same structure."
            )

        band_lookup = projection.band_index()
        missing_bands = [band for band in required if band not in band_lookup]
        if missing_bands:
            raise CharacterError(
                f"{path}: missing required VASP bands {missing_bands}. The NAMD basis "
                f"needs bands {required[0]}..{required[-1]}; widen the PROCAR band "
                "range or correct BMIN/BMAX."
            )

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
            bad_index = np.flatnonzero(captured <= 1.0e-14)
            bad = [required[i] for i in bad_index]
            worst = float(np.min(captured))
            raise CharacterError(
                f"{path}: the declared subsystems capture essentially zero projection "
                f"(smallest {worst:.3g}, expected order 1) for VASP band(s) {bad[:10]}"
                f"{'...' if len(bad) > 10 else ''}; normalizing their character would "
                "divide by zero and produce meaningless weights. Either those bands lie "
                "outside the PAW spheres of every declared group, or the atom-group map "
                "does not match this structure."
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
    if start < 1:
        raise CharacterError(
            f"{path}: NAMDTINI is {start}, but it is a one-based MD frame index "
            "and must be at least 1. A non-positive value shifts every frame "
            "assignment without changing anything that looks wrong downstream."
        )
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


@dataclass
class AnalysisPlan:
    """Everything decided before a single PROCAR is parsed.

    Building this is cheap: it reads the SHPROP headers and the projection
    manifest, and does all the arithmetic that decides which electronic frames
    are needed.  Both the preflight and the real run go through it, so what
    preflight checks is exactly what the run will do.
    """

    shprop_paths: List[Path]
    manifest: ProjectionManifest
    bmin: int
    bmax: int
    bands: List[int]
    frame_mode: str
    frames_by_file: List[np.ndarray]
    required_frames: List[int]
    alignments: List[Dict[str, Any]]
    n_time: List[int]
    manifest_frames: List[int]

    @property
    def manifest_frame_count(self) -> int:
        return len(self.manifest_frames)

    @property
    def required_frame_count(self) -> int:
        return len(self.required_frames)

    def missing_frames(self) -> List[int]:
        available = {frame for frame, _ in self.manifest.frames}
        return sorted(set(self.required_frames) - available)

    def consumption(self) -> Dict[str, Any]:
        """Declared manifest range against the subset actually consumed."""
        available = sorted(frame for frame, _ in self.manifest.frames)
        return {
            "manifest_declared_frames": len(available),
            "manifest_frame_min": available[0] if available else None,
            "manifest_frame_max": available[-1] if available else None,
            "frames_required_by_shprop": len(self.required_frames),
            "frames_required_min": self.required_frames[0] if self.required_frames else None,
            "frames_required_max": self.required_frames[-1] if self.required_frames else None,
            "procars_parsed": len(self.required_frames),
            "procars_skipped": max(0, len(available) - len(self.required_frames)),
            "note": (
                "only the electronic frames the supplied SHPROP histories actually "
                "visit are parsed; the whole manifest is still validated for "
                "existence, uniqueness and cycle coverage"
            ),
        }


def plan_analysis(
    shprop_paths: Sequence,
    state_map: StateMap,
    projection_manifest,
    frame_mode: str,
) -> AnalysisPlan:
    """Resolve band window and required electronic frames without parsing PROCARs."""
    paths = [Path(path) for path in shprop_paths]
    if not paths:
        raise CharacterError("no SHPROP files were supplied")
    records = [read_shprop_with_metadata(path) for path in paths]

    band_windows: List[Tuple[int, int]] = []
    for record in records:
        bmin = _int_metadata(record.metadata, "BMIN", record.path)
        bmax = _int_metadata(record.metadata, "BMAX", record.path)
        if bmax < bmin:
            raise CharacterError(
                f"{record.path}: SHPROP header has BMAX={bmax} < BMIN={bmin}; the "
                "basis window must increase"
            )
        band_windows.append((bmin, bmax))
    if len(set(band_windows)) != 1:
        grouped: Dict[Tuple[int, int], List[str]] = {}
        for record, window in zip(records, band_windows):
            grouped.setdefault(window, []).append(record.path.name)
        # Name the minority groups: with a thousand histories, listing them all
        # buries the one file that actually differs.
        parts = []
        for window, names in sorted(grouped.items(), key=lambda item: -len(item[1])):
            sample = ", ".join(names[:5]) + ("..." if len(names) > 5 else "")
            parts.append(f"{window[0]}:{window[1]} in {len(names)} file(s) ({sample})")
        raise CharacterError(
            f"SHPROP files declare {len(grouped)} different BMIN/BMAX windows: "
            + "; ".join(parts)
            + ". One character analysis covers one basis; split the campaign by window."
        )
    bmin, bmax = band_windows[0]
    bands = list(range(bmin, bmax + 1))
    if len(bands) != len(state_map.population_columns):
        raise CharacterError(
            f"SHPROP basis {bmin}:{bmax} has {len(bands)} states but the state map "
            f"declares {len(state_map.population_columns)} population columns "
            f"({state_map.population_columns}). Character analysis needs exactly one "
            "declared population column per basis state, ordered BMIN through BMAX. "
            "This is fixable in the state-map JSON."
        )

    manifest = _load_projection_manifest(projection_manifest)
    manifest_frames = sorted(frame for frame, _ in manifest.frames)

    frames_by_file: List[np.ndarray] = []
    alignments: List[Dict[str, Any]] = []
    n_time: List[int] = []
    for record in records:
        ntime = int(record.table.shape[0])
        n_time.append(ntime)
        frames = aligned_frames(
            record.metadata,
            ntime,
            frame_mode,
            record.path,
            cycle_length=manifest.cycle_length,
        )
        frames_by_file.append(frames)
        alignments.append(
            _alignment_record(record, frames, frame_mode, manifest, bmin, bmax, ntime)
        )

    required = sorted({int(frame) for frames in frames_by_file for frame in frames})
    if frame_mode == "dish-cyclic" and manifest.cycle_length is None:
        periods = {record["header_cycle_length"] for record in alignments}
        if len(periods) > 1:
            raise CharacterError(
                "the SHPROP histories imply different cyclic periods from their "
                f"own NSW headers ({sorted(p for p in periods if p is not None)}), "
                "so each file would wrap on a different cycle and the ensemble "
                "average would mix inconsistent frame mappings. Split the campaign "
                "by period, or declare one explicit cycle_length in the projection "
                "manifest if you know which is correct."
            )
    return AnalysisPlan(
        shprop_paths=paths,
        manifest=manifest,
        bmin=bmin,
        bmax=bmax,
        bands=bands,
        frame_mode=frame_mode,
        frames_by_file=frames_by_file,
        required_frames=required,
        alignments=alignments,
        n_time=n_time,
        manifest_frames=manifest_frames,
    )


def _alignment_record(
    record,
    frames: np.ndarray,
    frame_mode: str,
    manifest: ProjectionManifest,
    bmin: int,
    bmax: int,
    ntime: int,
) -> Dict[str, Any]:
    """Human-readable audit of how one SHPROP maps onto electronic frames."""
    header_nsw = record.metadata.get("NSW")
    header_period = None
    if isinstance(header_nsw, (int, float)):
        header_period = int(header_nsw) - 1
    cyclic = frame_mode == "dish-cyclic"
    period_used = manifest.cycle_length if manifest.cycle_length is not None else header_period
    values = [int(frame) for frame in frames]
    # A wrap is a step that does not simply increase by one.
    wraps = sum(1 for a, b in zip(values, values[1:]) if b != a + 1)
    return {
        "path": str(record.path.resolve()),
        "file": record.path.name,
        "NAMDTINI": _int_metadata(record.metadata, "NAMDTINI", record.path),
        "NSW": header_nsw,
        "n_time_points": ntime,
        "header_cycle_length": header_period,
        "cycle_length_used": period_used if cyclic else None,
        "cycle_length_source": (
            "projection_manifest"
            if cyclic and manifest.cycle_length is not None
            else "SHPROP_NSW_minus_1"
            if cyclic
            else "not_applicable"
        ),
        "header_cycle_mismatch": bool(
            cyclic
            and manifest.cycle_length is not None
            and header_period is not None
            and manifest.cycle_length != header_period
        ),
        "BMIN": bmin,
        "BMAX": bmax,
        "frame_mode": frame_mode,
        "first_projection_frame": values[0],
        "last_projection_frame": values[-1],
        "first_five_frames": values[:5],
        "last_five_frames": values[-5:],
        "unique_projection_frames_used": int(len(set(values))),
        "wrap_count": wraps,
    }


def preflight_report(
    shprop_paths: Sequence,
    state_map: StateMap,
    projection_manifest,
    atom_groups: AtomGroupMap,
    frame_mode: str,
) -> Dict[str, Any]:
    """Answer the cheap questions before the expensive analysis runs.

    Reads SHPROP headers, the projection manifest, and the *header* of one
    representative PROCAR.  It parses no projection data and computes no
    populations, so it stays fast enough to run every time.

    Problems are collected rather than raised, so a single run tells you
    everything that is wrong instead of only the first thing.
    """
    problems: List[str] = []
    warnings: List[str] = []
    payload: Dict[str, Any] = {
        "frame_mode": frame_mode,
        "n_shprop_files": len(list(shprop_paths)),
    }

    try:
        plan = plan_analysis(shprop_paths, state_map, projection_manifest, frame_mode)
    except CharacterError as exc:
        payload["problems"] = [str(exc)]
        payload["warnings"] = warnings
        payload["ok"] = False
        payload["stage"] = "planning"
        payload["note"] = (
            "preflight stopped while resolving the band window and frame alignment; "
            "nothing downstream could be checked"
        )
        return payload

    payload["stage"] = "planned"
    payload["shprop"] = [
        {
            "file": record["file"],
            "path": record["path"],
            "NAMDTINI": record["NAMDTINI"],
            "NSW": record["NSW"],
            "n_time_points": record["n_time_points"],
            "header_cycle_length": record["header_cycle_length"],
            "cycle_length_used": record["cycle_length_used"],
            "cycle_length_source": record["cycle_length_source"],
            "header_cycle_mismatch": record["header_cycle_mismatch"],
            "first_five_frames": record["first_five_frames"],
            "last_five_frames": record["last_five_frames"],
            "unique_projection_frames_used": record["unique_projection_frames_used"],
            "wrap_count": record["wrap_count"],
        }
        for record in plan.alignments
    ]
    payload["band_window"] = {
        "BMIN": plan.bmin,
        "BMAX": plan.bmax,
        "basis_size": len(plan.bands),
        "state_map_population_columns": list(state_map.population_columns),
    }
    payload["distinct_namdtini"] = sorted({r["NAMDTINI"] for r in plan.alignments})
    payload["row_counts"] = sorted(set(plan.n_time))
    payload["frames"] = plan.consumption()

    explicit = plan.manifest.cycle_length
    header_periods = sorted(
        {r["header_cycle_length"] for r in plan.alignments if r["header_cycle_length"] is not None}
    )
    payload["cycle"] = {
        "explicit_manifest_cycle_length": explicit,
        "header_derived_periods_nsw_minus_1": header_periods,
        "disagree": bool(
            explicit is not None and header_periods and [explicit] != header_periods
        ),
        "applies_to_this_frame_mode": frame_mode == "dish-cyclic",
    }
    if payload["cycle"]["disagree"] and frame_mode == "dish-cyclic":
        warnings.append(
            f"projection manifest declares cycle_length={explicit} but the SHPROP "
            f"headers imply NSW-1={header_periods}. The explicit manifest value wins "
            "and the mismatch is recorded in the alignment report; confirm it is "
            "deliberate before trusting the frame mapping."
        )

    missing = plan.missing_frames()
    if missing:
        available = sorted(frame for frame, _ in plan.manifest.frames)
        problems.append(
            f"the projection manifest lacks {len(missing)} electronic frames that the "
            f"SHPROP histories require, first {missing[:10]}"
            f"{'...' if len(missing) > 10 else ''}. The manifest covers "
            f"{available[0]}..{available[-1]}. Fixable by adding those PROCAR files, "
            "or by correcting NAMDTINI, --frame-mode or cycle_length."
        )
    payload["missing_required_frames"] = missing[:50]
    payload["n_missing_required_frames"] = len(missing)

    # One representative PROCAR: header only.
    representative = None
    lookup = dict(plan.manifest.frames)
    for frame in plan.required_frames:
        if frame in lookup:
            representative = (frame, lookup[frame])
            break
    if representative is None:
        problems.append("no required frame is present in the manifest at all")
    else:
        frame, path = representative
        try:
            structure = procar_structure(path)
        except ProcarFormatError as exc:
            problems.append(f"representative PROCAR could not be read: {exc}")
        else:
            structure["frame"] = frame
            payload["representative_procar"] = structure
            if not structure["single_kpoint"]:
                problems.append(
                    f"{path}: {structure['n_kpoints']} k-points. Character analysis "
                    "requires one and will not select or average implicitly. This is "
                    "a property of the source data, not of the configuration."
                )
            if not structure["single_spin_block"]:
                problems.append(
                    f"{path}: multiple spin components "
                    f"{structure['spin_components_seen']}. Splitting or combining them "
                    "must be explicit upstream; the source data is incompatible as is."
                )
            nions = structure["n_ions"]
            declared = sorted(
                atom for atoms in atom_groups.groups.values() for atom in atoms
            )
            payload["atom_coverage"] = {
                "procar_ions": nions,
                "assigned_ions": len(declared),
                "complete_atoms": atom_groups.complete_atoms,
                "max_declared_ion": declared[-1] if declared else None,
            }
            if declared and nions is not None and declared[-1] > nions:
                problems.append(
                    f"the atom-group map references PROCAR ion {declared[-1]} but "
                    f"{path} has only {nions}. Fixable in the atom-group JSON."
                )
            elif atom_groups.complete_atoms and nions is not None:
                unassigned = sorted(set(range(1, nions + 1)) - set(declared))
                payload["atom_coverage"]["unassigned_ions"] = unassigned[:20]
                payload["atom_coverage"]["n_unassigned_ions"] = len(unassigned)
                if unassigned:
                    problems.append(
                        f"complete_atoms is true but {len(unassigned)} of the {nions} "
                        f"PROCAR ions are in no group, first {unassigned[:10]}"
                        f"{'...' if len(unassigned) > 10 else ''}. Either assign them "
                        "or set complete_atoms: false. Fixable in the atom-group JSON."
                    )
            bands_declared = structure["n_bands"]
            # Read the band numbers rather than assuming they run 1..NBANDS:
            # comparing a required band *number* against a band *count* would
            # be wrong for any file that labels its blocks differently.
            present = set(procar_band_numbers(path, limit=bands_declared))
            absent = [band for band in plan.bands if band not in present]
            payload["band_coverage"] = {
                "procar_declared_band_count": bands_declared,
                "procar_band_numbers_seen": sorted(present)[:20],
                "required_bands": [plan.bands[0], plan.bands[-1]],
                "required_bands_absent": absent,
            }
            if absent:
                problems.append(
                    f"the NAMD basis needs VASP band(s) {absent[:10]}"
                    f"{'...' if len(absent) > 10 else ''} but {path} labels bands "
                    f"{sorted(present)[:10]}{'...' if len(present) > 10 else ''}. "
                    "Either BMIN/BMAX is wrong, or the PROCAR was written over too "
                    "narrow a band range."
                )

    payload["problems"] = problems
    payload["warnings"] = warnings
    payload["ok"] = not problems
    payload["note"] = (
        "preflight reads SHPROP headers, the projection manifest and one PROCAR "
        "header. It parses no projection data and produces no physical populations."
    )
    return payload


def character_populations(
    shprop_paths: Sequence,
    state_map: StateMap,
    projection_manifest,
    atom_groups: AtomGroupMap,
    frame_mode: str,
    plan: Optional[AnalysisPlan] = None,
) -> CharacterPopulationResult:
    """Project each original SHPROP into physical character before averaging.

    This ordering is essential.  Different SHPROP histories may have different
    NAMDTINI values, so averaging their adiabatic populations first destroys
    the phase needed to select the correct PROCAR frame.
    """
    paths = [Path(path) for path in shprop_paths]
    if plan is None:
        plan = plan_analysis(paths, state_map, projection_manifest, frame_mode)
    else:
        # The plan carries one frame series per file, zipped against the files
        # below. A plan built from different paths, or a different mode, would
        # pair the wrong frames with the wrong history silently.
        planned = [Path(item).resolve() for item in plan.shprop_paths]
        given = [item.resolve() for item in paths]
        if planned != given:
            raise CharacterError(
                "the supplied AnalysisPlan was built from different SHPROP files "
                f"than were passed: plan has {[p.name for p in plan.shprop_paths]}, "
                f"call has {[p.name for p in paths]}. Frames would be paired with "
                "the wrong history."
            )
        if plan.frame_mode != frame_mode:
            raise CharacterError(
                f"the supplied AnalysisPlan was built for frame mode "
                f"{plan.frame_mode!r} but {frame_mode!r} was requested"
            )
    population: PopulationSet = load_population_set(paths, state_map)
    records = [read_shprop_with_metadata(path) for path in paths]
    bands = plan.bands

    projection = load_projection_series(
        projection_manifest,
        atom_groups,
        bands,
        required_frames=plan.required_frames,
        manifest=plan.manifest,
    )
    frame_lookup = projection.frame_index()
    band_lookup = projection.band_index()
    projection_band_indices = np.asarray([band_lookup[band] for band in bands], dtype=int)

    physical_files: List[np.ndarray] = []
    alignments: List[Dict[str, Any]] = list(plan.alignments)
    for record, frames in zip(records, plan.frames_by_file):
        frame_indices = np.asarray([frame_lookup[int(frame)] for frame in frames], dtype=int)
        weights = projection.weights[frame_indices][:, projection_band_indices, :]
        pops = record.table[:, state_map.population_columns]
        physical = np.einsum("ts,tsg->tg", pops, weights)
        physical_files.append(physical)

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
            f"SHPROP population: total ranges [{conservation['min']:.8g}, "
            f"{conservation['max']:.8g}] and departs from 1 by up to "
            f"{conservation['max_abs_deviation_from_one']:.3g} (tolerance 1e-5). "
            "Check that the state map covers the whole basis and that the atom "
            "groups capture the projection; this is not repaired automatically"
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
    """Report dominant-subsystem changes along the underlying MD trajectory.

    Only the electronic frames the SHPROP histories actually visit are loaded,
    so consecutive entries in ``projection.frames`` need not be adjacent MD
    frames.  A change observed across a gap did happen, but it happened
    *somewhere inside* the gap and this analysis did not look there.  Such a
    change is reported with its frame gap and counted separately from an
    adjacent, fully resolved swap, rather than being presented as a single
    step between two distant frames.
    """
    if not (0.0 < dominance_threshold <= 1.0):
        raise CharacterError("dominance_threshold must lie in (0, 1]")
    rows: List[List[Any]] = []
    mixed = 0
    total = projection.weights.shape[0] * projection.weights.shape[1]
    swap_count = 0
    adjacent_swaps = 0
    gap_changes = 0
    frames_list = [int(frame) for frame in projection.frames]
    unobserved = sum(
        max(0, b - a - 1) for a, b in zip(frames_list, frames_list[1:])
    )
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
            gap = frames_list[fi] - frames_list[fi - 1]
            if gap == 1:
                adjacent_swaps += 1
                resolution = "adjacent"
            else:
                gap_changes += 1
                resolution = "across_gap"
            rows.append(
                [
                    int(band),
                    int(projection.frames[fi - 1]),
                    int(projection.frames[fi]),
                    projection.group_names[before],
                    projection.group_names[after],
                    float(confidence[fi - 1]),
                    float(confidence[fi]),
                    int(gap),
                    resolution,
                ]
            )
    summary = {
        "dominance_threshold": dominance_threshold,
        "dominant_character_swaps": swap_count,
        "adjacent_swaps": adjacent_swaps,
        "changes_across_a_frame_gap": gap_changes,
        "mixed_frame_band_samples": mixed,
        "total_frame_band_samples": total,
        "mixed_fraction": float(mixed / total) if total else 0.0,
        "frames_examined": len(frames_list),
        "md_frames_skipped_between_examined_frames": unobserved,
        "resolution_note": (
            "only the electronic frames the SHPROP histories visit are loaded, so "
            f"{unobserved} MD frame(s) lie between examined frames and were not "
            "inspected. A change marked 'across_gap' happened somewhere inside "
            "that gap, not in one step between the two frames named. Swap counts "
            "are a lower bound on the number of character changes along the full "
            "MD trajectory"
            if unobserved
            else "every examined frame is adjacent to the next, so the swap count "
            "resolves every change over the range examined"
        ),
    }
    return rows, summary


DISCREPANCY_HEADER = [
    "group",
    "max_abs_difference",
    "time_of_max_ns",
    "rms_difference",
    "integrated_abs_difference_ns",
    "mean_signed_difference",
    "fixed_at_max",
    "projected_at_max",
]


def fixed_vs_projected_summary(result: "CharacterPopulationResult") -> Dict[str, Any]:
    """How far the dynamic character moves the answer, per shared group.

    Compares the projection-weighted population against the fixed-column
    population on the same time grid, for groups named in both.  A large
    discrepancy says the fixed map was mislabelling occupation; a small one
    says the fixed map was adequate for this campaign.  Neither is a defect.
    """
    time_ns = result.time_ns
    lookup = {name: i for i, name in enumerate(result.group_names)}
    shared = [name for name in result.fixed_mean if name in lookup]
    rows: List[List[Any]] = []
    records: List[Dict[str, Any]] = []
    integrate = getattr(np, "trapezoid", None) or np.trapz
    for name in shared:
        fixed = np.asarray(result.fixed_mean[name], dtype=float)
        projected = result.mean[:, lookup[name]]
        difference = projected - fixed
        peak = int(np.argmax(np.abs(difference)))
        record = {
            "group": name,
            "max_abs_difference": float(np.abs(difference)[peak]),
            "time_of_max_ns": float(time_ns[peak]),
            "rms_difference": float(np.sqrt(np.mean(difference**2))),
            "integrated_abs_difference_ns": float(integrate(np.abs(difference), x=time_ns)),
            "mean_signed_difference": float(np.mean(difference)),
            "fixed_at_max": float(fixed[peak]),
            "projected_at_max": float(projected[peak]),
        }
        records.append(record)
        rows.append([record[key] for key in DISCREPANCY_HEADER])

    largest = max(records, key=lambda r: r["max_abs_difference"]) if records else None
    return {
        "shared_groups": shared,
        "groups_only_in_fixed_map": sorted(set(result.fixed_mean) - set(lookup)),
        "groups_only_in_projection": sorted(set(lookup) - set(result.fixed_mean)),
        "per_group": records,
        "rows": rows,
        "largest_disagreement": largest,
        "note": (
            "the fixed comparison assigns each SHPROP column to one group for the "
            "whole trajectory; the projected one reassigns character frame by frame. "
            "A large difference means the fixed labelling was wrong somewhere, not "
            "that either number is a flux or a transfer"
        ),
    }


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
