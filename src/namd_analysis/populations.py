"""SHPROP population averaging and group analysis.

All column numbers in a configuration are zero-based *table* columns, not
VASP band numbers.  Group membership is a declaration by the user about the
spatial character of each state; nothing here infers it, and a state whose
character changes during the trajectory cannot be tracked by a fixed map.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .io.hefei import read_shprop
from .units import FS_PER_NS

#: Divisors, not multipliers: ``t_fs * (1/1e6)`` is not exactly ``t_fs / 1e6``,
#: and the difference is enough to drop the frame sitting exactly on a
#: requested window boundary.
TIME_DIVISORS = {"fs": FS_PER_NS, "ps": 1.0e3, "ns": 1.0}
TIME_UNITS = TIME_DIVISORS

#: Absolute tolerance on the population sum of a complete state map, and on
#: the range of any single population column.  One value, used everywhere a
#: population is validated, so averaging cannot be held to a looser standard
#: than the files it averages.
CONSERVATION_ATOL = 1.0e-5


class ConfigError(ValueError):
    """Raised for an inconsistent or ambiguous state map."""


def validate_band_numbers(values, nstates: int, source: str) -> List[int]:
    """Check a declared VASP band list, or say exactly what is wrong with it.

    Band numbers are provenance, not data: nothing downstream can detect a
    wrong one, so every way of being wrong is rejected here rather than
    normalized.
    """
    bands: List[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{source}: band numbers must be positive integers")
        if isinstance(value, float) and not float(value).is_integer():
            raise ConfigError(f"{source}: band numbers must be positive integers")
        number = int(value)
        if number <= 0:
            raise ConfigError(f"{source}: band numbers must be positive integers")
        bands.append(number)
    if len(bands) != nstates:
        raise ConfigError(
            f"{source}: {len(bands)} band number(s) for {nstates} population "
            "column(s); there must be exactly one band per column, in the same order"
        )
    if len(set(bands)) != len(bands):
        raise ConfigError(f"{source}: band numbers contain duplicates")
    return bands


class InputMismatchError(ValueError):
    """Raised when SHPROP files cannot be averaged together as given."""


@dataclass
class StateMap:
    """A declared assignment of table columns to physical groups."""

    name: str
    time_column: int
    time_unit: str
    population_columns: List[int]
    groups: Dict[str, List[int]]
    complete_population: bool = False
    recombined_group: Optional[str] = None
    notes: str = ""
    #: Exact VASP band numbers, one per population column and in the same
    #: order.  Production SHPROP files are plain numeric tables that need not
    #: carry BMIN/BMAX, so this is the highest-precedence statement of which
    #: bands the basis is, and the only one a user can make directly.
    band_numbers: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StateMap":
        if not isinstance(payload.get("complete_population", False), bool):
            raise ConfigError("complete_population must be a JSON boolean")
        raw_columns = [payload.get("time_column", 0)] + payload.get("population_columns", [])
        for columns in payload.get("groups", {}).values():
            raw_columns += columns
        if any(type(col) is not int or col < 0 for col in raw_columns):
            raise ConfigError("column indices must be nonnegative integers")
        try:
            groups = {
                str(key): [int(col) for col in cols]
                for key, cols in payload["groups"].items()
            }
            state_map = cls(
                name=str(payload.get("name", "unnamed")),
                time_column=int(payload["time_column"]),
                time_unit=str(payload.get("time_unit", "fs")),
                population_columns=[int(c) for c in payload["population_columns"]],
                groups=groups,
                complete_population=bool(payload.get("complete_population", False)),
                recombined_group=payload.get("recombined_group"),
                notes=str(payload.get("notes", "")),
                band_numbers=(
                    list(payload["band_numbers"])
                    if payload.get("band_numbers") is not None
                    else None
                ),
            )
        except KeyError as exc:
            raise ConfigError(f"configuration is missing required key {exc}") from exc
        state_map.validate()
        return state_map

    @classmethod
    def from_json(cls, path) -> "StateMap":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> None:
        if self.time_unit not in TIME_DIVISORS:
            raise ConfigError(
                f"time_unit {self.time_unit!r} is not one of {sorted(TIME_DIVISORS)}"
            )
        if not self.groups:
            raise ConfigError("at least one group must be declared")
        declared = self.population_columns
        if len(set(declared)) != len(declared):
            raise ConfigError("population_columns contains duplicates")
        if self.time_column in declared:
            raise ConfigError("time_column is also listed as a population column")
        seen: Dict[int, str] = {}
        for name, columns in self.groups.items():
            if not columns:
                raise ConfigError(f"group {name!r} is empty")
            for column in columns:
                if column not in declared:
                    raise ConfigError(
                        f"group {name!r} uses column {column}, which is not in "
                        "population_columns"
                    )
                if column in seen:
                    raise ConfigError(
                        f"column {column} appears in both {seen[column]!r} and "
                        f"{name!r}; groups must be disjoint"
                    )
                seen[column] = name
        if not declared or self.time_column < 0 or any(c < 0 for c in declared):
            raise ConfigError("population columns must be nonempty and nonnegative")
        if self.complete_population and set(seen) != set(declared):
            raise ConfigError("complete_population requires exhaustive groups")
        if self.band_numbers is not None:
            validate_band_numbers(
                self.band_numbers, len(declared), "state_map band_numbers"
            )
        if self.recombined_group is not None:
            if self.recombined_group not in self.groups:
                raise ConfigError(
                    f"recombined_group {self.recombined_group!r} is not a declared group"
                )
            if not self.complete_population:
                raise ConfigError(
                    "recombined_group requires complete_population: true, because "
                    "survival = 1 - P(recombined) is only defined for a complete map"
                )

    @property
    def to_ns(self) -> float:
        """Divisor taking the file's time unit to nanoseconds."""
        return TIME_DIVISORS[self.time_unit]

    def as_dict(self) -> Dict[str, Any]:
        """The declaration itself, in a canonical order."""
        return {
            "name": self.name,
            "time_column": self.time_column,
            "time_unit": self.time_unit,
            "population_columns": list(self.population_columns),
            "band_numbers": None if self.band_numbers is None else list(self.band_numbers),
            "groups": {name: list(cols) for name, cols in self.groups.items()},
            "complete_population": self.complete_population,
            "recombined_group": self.recombined_group,
            "notes": self.notes,
        }

    def fingerprint(self) -> str:
        """SHA-256 of the canonical declaration.

        Two runs that report the same fingerprint used the same column-to-group
        assignment, whatever the file it was read from was called.
        """
        import hashlib

        canonical = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def ungrouped_columns(self) -> List[int]:
        assigned = {c for cols in self.groups.values() for c in cols}
        return [c for c in self.population_columns if c not in assigned]


@dataclass
class PopulationSet:
    """Per-file SHPROP tables averaged on a single shared time grid."""

    paths: List[Path]
    time_ns: np.ndarray
    mean: np.ndarray
    sem: Optional[np.ndarray]
    n_files: int
    total_population: np.ndarray
    conservation: Dict[str, float] = field(default_factory=dict)
    #: Per-file tables, shape ``(nfiles, nrows, ncols)``.  Kept so that a
    #: resampling estimate can draw whole files; averaging discards this.
    stack: Optional[np.ndarray] = None


def load_population_set(paths: Sequence, state_map: StateMap) -> PopulationSet:
    """Average SHPROP files with equal weight, without interpolation.

    Every file must share an identical time grid; differing grids are an
    error rather than something to resample.  Each file gets equal weight, so
    pre-averaged and individual files must not be mixed.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        raise InputMismatchError("no SHPROP files were given")

    state_map.validate()
    if len({p.resolve() for p in paths}) != len(paths):
        raise InputMismatchError("duplicate population files")
    tables = [read_shprop(path) for path in paths]

    shapes = {t.shape for t in tables}
    if len(shapes) != 1:
        grouped: Dict[Any, List[str]] = {}
        for path, table in zip(paths, tables):
            grouped.setdefault(table.shape, []).append(path.name)
        # Name the minority rather than listing every file: a campaign can pass
        # a thousand histories, and the one that differs must be findable.
        parts = []
        for shape, names in sorted(grouped.items(), key=lambda item: -len(item[1])):
            sample = ", ".join(names[:5]) + ("..." if len(names) > 5 else "")
            parts.append(f"{shape} in {len(names)} file(s) ({sample})")
        raise InputMismatchError(
            "SHPROP files have different shapes: "
            + "; ".join(parts)
            + ". No interpolation or truncation is performed."
        )

    ncols = tables[0].shape[1]
    needed = max([state_map.time_column] + state_map.population_columns)
    if needed >= ncols:
        raise InputMismatchError(
            f"configuration references column {needed} but the files have {ncols} columns"
        )

    stack = np.stack(tables, axis=0)
    time_raw = stack[:, :, state_map.time_column]
    if not np.array_equal(time_raw, np.broadcast_to(time_raw[0], time_raw.shape)):
        raise InputMismatchError(
            "SHPROP files do not share an identical time grid; no interpolation "
            "or truncation is performed"
        )

    if time_raw.shape[1] < 2 or np.any(np.diff(time_raw[0]) <= 0):
        raise InputMismatchError("population time must strictly increase with at least two samples")
    values = stack[:, :, state_map.population_columns]
    if values.min() < -CONSERVATION_ATOL or values.max() > 1 + CONSERVATION_ATOL:
        raise InputMismatchError("population columns outside [0,1]; verify state map")
    totals = values.sum(axis=2)
    if state_map.complete_population and not np.allclose(
        totals, 1, atol=CONSERVATION_ATOL, rtol=0
    ):
        raise InputMismatchError("complete populations must sum to one in every file")
    mean = stack.mean(axis=0)
    sem = None
    if len(paths) > 1:
        sem = stack.std(axis=0, ddof=1) / np.sqrt(len(paths))

    time_ns = time_raw[0] / state_map.to_ns
    total = mean[:, state_map.population_columns].sum(axis=1)
    conservation = {
        "min": float(np.min(total)),
        "max": float(np.max(total)),
        "mean": float(np.mean(total)),
        "max_abs_deviation_from_one": float(np.max(np.abs(total - 1.0))),
    }
    return PopulationSet(
        paths=paths,
        time_ns=time_ns,
        mean=mean,
        sem=sem,
        n_files=len(paths),
        total_population=total,
        conservation=conservation,
        stack=stack,
    )


def per_file_group_populations(
    population: PopulationSet, state_map: StateMap
) -> Optional[np.ndarray]:
    """Group populations for each input file, shape ``(nfiles, ntime, ngroups)``.

    Returns ``None`` when the per-file tables were not retained.
    """
    if population.stack is None:
        return None
    columns = [state_map.groups[name] for name in state_map.groups]
    return np.stack(
        [population.stack[:, :, group].sum(axis=2) for group in columns], axis=2
    )


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(y, x=x))


@dataclass
class GroupSeries:
    name: str
    columns: List[int]
    values: np.ndarray
    sem: Optional[np.ndarray]

    def summary(self, time_ns: np.ndarray) -> Dict[str, Any]:
        return {
            "group": self.name,
            "columns": list(self.columns),
            "initial": float(self.values[0]),
            "final": float(self.values[-1]),
            "net_change": float(self.values[-1] - self.values[0]),
            "max": float(np.max(self.values)),
            "time_of_max_ns": float(time_ns[int(np.argmax(self.values))]),
            "min": float(np.min(self.values)),
            "window_integral_ns": trapezoid(self.values, time_ns),
        }


def group_series(population: PopulationSet, state_map: StateMap) -> List[GroupSeries]:
    """Sum per file before computing SEM, preserving within-group covariance."""
    series = []
    for name, columns in state_map.groups.items():
        values = population.mean[:, columns].sum(axis=1)
        sem = None
        if population.sem is not None:
            if population.stack is not None:
                grouped_files = population.stack[:, :, columns].sum(axis=2)
                sem = grouped_files.std(axis=0, ddof=1) / np.sqrt(population.n_files)
            else:
                # Covariance cannot be reconstructed from marginal SEM alone.
                sem = None
        series.append(
            GroupSeries(name=name, columns=list(columns), values=values, sem=sem)
        )
    return series


def survival(population: PopulationSet, state_map: StateMap) -> Optional[np.ndarray]:
    """S(t) = 1 - P(recombined group), defined only for a complete map."""
    if state_map.recombined_group is None or not state_map.complete_population:
        return None
    columns = state_map.groups[state_map.recombined_group]
    return 1.0 - population.mean[:, columns].sum(axis=1)
