"""Canonical generation of a master SHPROP from the original SHPROP files.

The historical shell workflow (``SHPROP_avg.sh``) averaged **one** selected
population column and copied every other column from whichever file happened to
be first.  A file produced that way is not an averaged multistate population
history: the column that was averaged and the columns that were not are
describing different things, and nothing in the file says which is which.

This module produces the master a scientific analysis actually needs.  Every
population column declared by the state map is averaged with equal weight per
file,

.. math::  \\bar{P}_i(t) = \\frac{1}{N} \\sum_r P_{i,r}(t)

and nothing else is touched:

* the **time column** is verified identical across every input and copied
  through exactly.  No interpolation, no truncation to a common length, no
  shifting;
* **other columns** -- the running energy Hefei-NAMD writes in column 1, for
  example -- are copied only when every input agrees on them exactly.  When
  they disagree, a mean is not obviously the right answer (the mean of a state
  energy over trajectories that were in different states is not any state's
  energy), so master generation refuses unless an explicit policy says to
  average them;
* when the state map declares a complete population basis, conservation is
  checked in **each input file independently** as well as in the result, so
  averaging cannot bury a bad file inside a good-looking mean.

The between-file spread is written to a separate table rather than added as
extra columns, so ``SHPROP.master`` stays readable by anything that reads a
SHPROP file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .io.hefei import read_shprop
from .populations import CONSERVATION_ATOL, InputMismatchError, StateMap
from .provenance import fingerprint, sha256_file

#: How a column that is neither time nor population was handled.
COPIED = "copied_identical_across_files"
AVERAGED = "averaged_under_explicit_policy"

#: ``%.17g`` is the shortest format that round-trips an IEEE double exactly, so
#: reading ``SHPROP.master`` back gives the numbers that were computed rather
#: than a rounding of them.  The conservation check on the written file would
#: otherwise be checking a different table from the one validated here.
MASTER_FORMAT = "%.17g"

INPUT_HEADER = [
    "path",
    "bytes",
    "sha256",
    "modified_utc",
    "n_rows",
    "n_columns",
    "conservation_max_abs_deviation_from_one",
]


class MasterError(InputMismatchError):
    """Raised when a canonical master SHPROP cannot be produced as asked."""


@dataclass
class MasterShprop:
    """A canonical mean SHPROP plus everything needed to defend it."""

    paths: List[Path]
    table: np.ndarray
    state_map: StateMap
    time_column: int
    population_columns: List[int]
    extra_columns: List[int]
    extra_column_policy: Dict[int, str]
    population_sem: Optional[np.ndarray]
    group_values: Dict[str, np.ndarray]
    group_sem: Dict[str, np.ndarray]
    per_file_conservation: List[Dict[str, float]]
    master_conservation: Dict[str, float]
    rows: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def n_files(self) -> int:
        return len(self.paths)

    @property
    def time_values(self) -> np.ndarray:
        return self.table[:, self.time_column]

    def averaging_rule(self) -> Dict[str, Any]:
        return {
            "population_columns": (
                "Pbar_i(t) = (1/N) sum_r P_i,r(t) over all N source files, equal "
                "weight per file, computed independently for EVERY declared "
                "population column"
            ),
            "time_column": (
                "verified identical in every source file and copied through "
                "exactly; no interpolation, truncation or time shifting"
            ),
            "other_columns": {
                str(column): self.extra_column_policy[column]
                for column in self.extra_columns
            },
            "n_source_files": self.n_files,
            "weighting": "equal weight per file",
            "not_done": [
                "no column is normalized, reordered or repaired",
                "files with different time grids or shapes are rejected rather "
                "than resampled onto a common grid",
                "the historical single-column average (SHPROP_avg.sh) is not "
                "reproduced: every population column is averaged",
            ],
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_source_files": self.n_files,
            "n_rows": int(self.table.shape[0]),
            "n_columns": int(self.table.shape[1]),
            "time_column": self.time_column,
            "population_columns": list(self.population_columns),
            "extra_columns": list(self.extra_columns),
            "extra_column_policy": {
                str(k): v for k, v in self.extra_column_policy.items()
            },
            "state_map": self.state_map.as_dict(),
            "state_map_fingerprint": self.state_map.fingerprint(),
            "averaging_rule": self.averaging_rule(),
            "conservation": {
                "tolerance": CONSERVATION_ATOL,
                "complete_population_declared": self.state_map.complete_population,
                "per_source_file": self.per_file_conservation,
                "master": self.master_conservation,
                "checked": (
                    "every source file independently, and the master"
                    if self.state_map.complete_population
                    else "range only; a partial map makes no conservation claim"
                ),
            },
            "between_file_sem": (
                "per population column and per declared group, written to "
                "population_sem.csv. Groups are summed within each file before "
                "the spread is taken, so within-group covariance is preserved; a "
                "grouped SEM is never built from marginal state SEMs. It is NOT "
                "inserted into SHPROP.master, which stays SHPROP-compatible"
                if self.population_sem is not None
                else "omitted: one file gives no between-file spread"
            ),
            "interpretation_limits": [
                "the between-file SEM measures the spread between the supplied "
                "files, which commonly share one MD trajectory and correlated "
                "initial conditions; it is not an independent ensemble error bar",
                "averaging populations does not produce hopping events; "
                "directional transfer counts are not recoverable from this file",
            ],
        }


def _conservation(totals: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(totals)),
        "max": float(np.max(totals)),
        "mean": float(np.mean(totals)),
        "max_abs_deviation_from_one": float(np.max(np.abs(totals - 1.0))),
    }


def build_master(
    paths: Sequence,
    state_map: StateMap,
    average_extra_columns: bool = False,
) -> MasterShprop:
    """Average the original SHPROP files into one canonical master table.

    ``average_extra_columns`` is the explicit policy that permits averaging a
    column which is neither time nor population and which the inputs disagree
    on.  Without it such a disagreement is an error, because silently keeping
    the first file's copy would attach one trajectory's metadata to everyone
    else's populations.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        raise MasterError("no SHPROP files were given")
    state_map.validate()

    resolved = [p.resolve() for p in paths]
    if len(set(resolved)) != len(resolved):
        raise MasterError(
            "the same SHPROP file was supplied more than once; every file gets "
            "equal weight, so a repeat silently reweights the average"
        )

    digests: Dict[str, Path] = {}
    for path in paths:
        digest = sha256_file(path)
        if digest in digests:
            raise MasterError(
                f"{path} and {digests[digest]} are byte-identical (sha256 "
                f"{digest[:16]}...); averaging them would weight that history "
                "twice. Supply each original SHPROP once."
            )
        digests[digest] = path

    tables = [read_shprop(path) for path in paths]
    shapes = {t.shape for t in tables}
    if len(shapes) != 1:
        detail = ", ".join(f"{p.name}:{t.shape}" for p, t in zip(paths, tables))
        raise MasterError(
            f"SHPROP files have different shapes ({detail}); they are not "
            "truncated to a common length"
        )

    stack = np.stack(tables, axis=0)
    n_files, n_rows, n_cols = stack.shape
    needed = max([state_map.time_column] + state_map.population_columns)
    if needed >= n_cols:
        raise MasterError(
            f"the state map references column {needed} but the files have "
            f"{n_cols} columns"
        )

    time_raw = stack[:, :, state_map.time_column]
    if not np.array_equal(time_raw, np.broadcast_to(time_raw[0], time_raw.shape)):
        offenders = [
            str(path)
            for path, column in zip(paths, time_raw)
            if not np.array_equal(column, time_raw[0])
        ]
        raise MasterError(
            "SHPROP files do not share an identical time grid, so there is no "
            "common time axis to write: "
            + ", ".join(offenders[:4])
            + (" and others" if len(offenders) > 4 else "")
            + ". No interpolation, truncation or time shifting is performed."
        )
    if n_rows < 2 or np.any(np.diff(time_raw[0]) <= 0):
        raise MasterError(
            "the time column must strictly increase over at least two samples"
        )

    values = stack[:, :, state_map.population_columns]
    if values.min() < -CONSERVATION_ATOL or values.max() > 1 + CONSERVATION_ATOL:
        raise MasterError(
            "population columns fall outside [0, 1]; verify the state map before "
            "averaging"
        )

    per_file_conservation: List[Dict[str, float]] = []
    for path, table in zip(paths, values):
        totals = table.sum(axis=1)
        record = _conservation(totals)
        per_file_conservation.append({"path": str(path), **record})
        if (
            state_map.complete_population
            and record["max_abs_deviation_from_one"] > CONSERVATION_ATOL
        ):
            raise MasterError(
                f"{path}: the declared complete population sums to between "
                f"{record['min']:.8g} and {record['max']:.8g}, outside the "
                f"tolerance {CONSERVATION_ATOL:g}. Averaging would hide this "
                "file inside the mean, so the master is not generated."
            )

    master = np.empty((n_rows, n_cols), dtype=float)
    master[:, state_map.time_column] = time_raw[0]
    master[:, state_map.population_columns] = values.mean(axis=0)

    extra_columns = [
        column
        for column in range(n_cols)
        if column != state_map.time_column
        and column not in state_map.population_columns
    ]
    policy: Dict[int, str] = {}
    for column in extra_columns:
        series = stack[:, :, column]
        identical = np.array_equal(series, np.broadcast_to(series[0], series.shape))
        if identical:
            master[:, column] = series[0]
            policy[column] = COPIED
        elif average_extra_columns:
            master[:, column] = series.mean(axis=0)
            policy[column] = AVERAGED
        else:
            spread = float(np.max(np.abs(series - series[0])))
            raise MasterError(
                f"column {column} is neither the time column nor a declared "
                f"population column, and the source files disagree on it by up "
                f"to {spread:.6g}. The first file's copy is not the average of "
                "anything, and the mean of a per-trajectory quantity such as a "
                "running state energy need not be meaningful. Either declare it "
                "in the state map, or pass --average-extra-columns to state "
                "that averaging it is what you want."
            )

    master_totals = master[:, state_map.population_columns].sum(axis=1)
    master_conservation = _conservation(master_totals)
    if (
        state_map.complete_population
        and master_conservation["max_abs_deviation_from_one"] > CONSERVATION_ATOL
    ):  # pragma: no cover - unreachable while every input already conserves
        raise MasterError(
            "the averaged populations do not sum to one within "
            f"{CONSERVATION_ATOL:g} despite every input file doing so"
        )

    population_sem = None
    if n_files > 1:
        population_sem = values.std(axis=0, ddof=1) / np.sqrt(n_files)

    group_values: Dict[str, np.ndarray] = {}
    group_sem: Dict[str, np.ndarray] = {}
    for name, columns in state_map.groups.items():
        # Sum into the physical group inside each file first: a group's spread
        # is not reconstructable from the marginal spreads of its members.
        per_file = stack[:, :, columns].sum(axis=2)
        group_values[name] = per_file.mean(axis=0)
        if n_files > 1:
            group_sem[name] = per_file.std(axis=0, ddof=1) / np.sqrt(n_files)

    rows = []
    for record, table, conservation in zip(
        fingerprint(paths), tables, per_file_conservation
    ):
        rows.append(
            {
                **record,
                "n_rows": int(table.shape[0]),
                "n_columns": int(table.shape[1]),
                "conservation": {
                    key: value for key, value in conservation.items() if key != "path"
                },
            }
        )

    return MasterShprop(
        paths=paths,
        table=master,
        state_map=state_map,
        time_column=state_map.time_column,
        population_columns=list(state_map.population_columns),
        extra_columns=extra_columns,
        extra_column_policy=policy,
        population_sem=population_sem,
        group_values=group_values,
        group_sem=group_sem,
        per_file_conservation=per_file_conservation,
        master_conservation=master_conservation,
        rows=rows,
    )


def write_master(master: MasterShprop, path) -> Path:
    """Write ``SHPROP.master`` as a plain numeric table, no header."""
    path = Path(path)
    np.savetxt(path, master.table, fmt=MASTER_FORMAT, delimiter=" ")
    return path


def sem_table(master: MasterShprop) -> Dict[str, Any]:
    """The between-file spread, as a header and rows for a CSV."""
    header = ["time"]
    columns: List[np.ndarray] = [master.time_values]
    for index, column in enumerate(master.population_columns):
        header.append(f"column_{column}_mean")
        columns.append(master.table[:, column])
        if master.population_sem is not None:
            header.append(f"column_{column}_sem")
            columns.append(master.population_sem[:, index])
    for name in master.state_map.groups:
        header.append(f"group_{name}_mean")
        columns.append(master.group_values[name])
        if name in master.group_sem:
            header.append(f"group_{name}_sem")
            columns.append(master.group_sem[name])
    return {"header": header, "rows": np.column_stack(columns).tolist()}


def input_rows(master: MasterShprop) -> List[List[Any]]:
    return [
        [
            record["path"],
            record["bytes"],
            record["sha256"],
            record["modified_utc"],
            record["n_rows"],
            record["n_columns"],
            record["conservation"]["max_abs_deviation_from_one"],
        ]
        for record in master.rows
    ]
