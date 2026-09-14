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


class ConfigError(ValueError):
    """Raised for an inconsistent or ambiguous state map."""


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

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StateMap":
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

    tables = [read_shprop(path) for path in paths]

    shapes = {t.shape for t in tables}
    if len(shapes) != 1:
        detail = ", ".join(f"{p.name}:{t.shape}" for p, t in zip(paths, tables))
        raise InputMismatchError(f"SHPROP files have different shapes ({detail})")

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
    """Sum the mapped columns of each group; combine SEM in quadrature."""
    series = []
    for name, columns in state_map.groups.items():
        values = population.mean[:, columns].sum(axis=1)
        sem = None
        if population.sem is not None:
            # Between-file errors of columns within one group are not
            # independent; this is a first-order estimate, reported as such.
            sem = np.sqrt(np.sum(population.sem[:, columns] ** 2, axis=1))
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
