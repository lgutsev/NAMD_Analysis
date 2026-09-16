"""Bounded-memory ensemble statistics over SHPROP histories.

An archived campaign can hold several histories of a few hundred megabytes
each.  Reading them all into one ``(nfiles, nrows, ncols)`` stack costs several
gigabytes before any analysis starts -- and the ordinary text reader costs
several times that again, because it builds one Python list per row before the
array exists.  That is what kills a preflight on a normal node.

Nothing here changes what is computed.  The ensemble mean and the between-file
standard error are the same quantities as before, formed by an online update
instead of from a materialized stack, and the validation applied to each chunk
is the validation the whole-file reader applies to the whole table.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

#: Above this many bytes an accumulator is backed by a memory map rather than
#: held in RAM.  Two arrays of this size (mean and M2) per accumulator is the
#: real footprint, so the effective ceiling is twice this before spilling.
DEFAULT_MEMMAP_THRESHOLD_BYTES = 512 * 1024 * 1024


class StreamingError(ValueError):
    """Raised when streamed input cannot be combined without guessing."""


def allocate(
    shape,
    memmap_dir: Optional[Path] = None,
    threshold_bytes: int = DEFAULT_MEMMAP_THRESHOLD_BYTES,
    name: str = "acc",
) -> np.ndarray:
    """Zeroed float64 array, spilled to a memory map when it is large.

    The caller decides where a spill file lives so that a scheduler's local
    scratch can be used; with no directory the array stays in RAM whatever its
    size, because silently writing gigabytes into a home directory would be
    worse than the allocation it avoids.
    """
    nbytes = int(np.prod(shape)) * 8
    if memmap_dir is None or nbytes <= threshold_bytes:
        return np.zeros(shape, dtype=float)
    memmap_dir = Path(memmap_dir)
    memmap_dir.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f"{name}_", suffix=".dat", dir=str(memmap_dir), delete=False
    )
    handle.close()
    mapped = np.memmap(handle.name, dtype=float, mode="w+", shape=tuple(shape))
    mapped[...] = 0.0
    return mapped


@dataclass
class OnlineEnsemble:
    """Welford mean/variance across whole files, updated in row slices.

    Each file contributes exactly one sample to every time row, so the sample
    count is a scalar: :meth:`begin_file` raises it once, and the file's row
    slices then update disjoint parts of the same accumulator.  Updating a
    slice more than once per file, or leaving one out, would corrupt the
    variance silently, so :meth:`begin_file` tracks coverage and
    :meth:`finish_file` checks it.
    """

    ntime: int
    nvalues: int
    memmap_dir: Optional[Path] = None
    threshold_bytes: int = DEFAULT_MEMMAP_THRESHOLD_BYTES
    name: str = "ensemble"
    count: int = 0
    mean: np.ndarray = field(init=False)
    m2: np.ndarray = field(init=False)
    #: Spill files this accumulator created, if any.  Reported so a caller can
    #: name them, and removed by :meth:`release`; a campaign large enough to
    #: need them is large enough that leaving them behind matters.
    spill_files: List[Path] = field(default_factory=list, init=False)
    _rows_this_file: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        shape = (self.ntime, self.nvalues)
        self.mean = allocate(shape, self.memmap_dir, self.threshold_bytes, f"{self.name}_mean")
        self.m2 = allocate(shape, self.memmap_dir, self.threshold_bytes, f"{self.name}_m2")
        for array in (self.mean, self.m2):
            if isinstance(array, np.memmap):
                self.spill_files.append(Path(array.filename))

    def release(self) -> None:
        """Close any memory maps and delete their backing files.

        Safe to call on an in-RAM accumulator, where it does nothing.  After
        this the arrays must not be read again, so a caller that still needs
        the mean should copy it out first.
        """
        for attribute in ("mean", "m2"):
            array = getattr(self, attribute, None)
            if isinstance(array, np.memmap):
                array.flush()
                array._mmap.close()
            setattr(self, attribute, None)
        for path in self.spill_files:
            try:
                Path(path).unlink()
            except OSError:
                pass
        self.spill_files = []

    def begin_file(self) -> None:
        self.count += 1
        self._rows_this_file = 0

    def update(self, start: int, values: np.ndarray) -> None:
        """Fold one file's rows ``[start, start + len(values))`` into the mean."""
        rows = values.shape[0]
        if rows == 0:
            return
        stop = start + rows
        if stop > self.ntime:
            raise StreamingError(
                f"{self.name}: rows {start}..{stop - 1} exceed the {self.ntime} "
                "time points the first history established"
            )
        block_mean = self.mean[start:stop]
        delta = values - block_mean
        block_mean += delta / self.count
        self.m2[start:stop] += delta * (values - block_mean)
        self._rows_this_file += rows

    def finish_file(self) -> None:
        if self._rows_this_file != self.ntime:
            raise StreamingError(
                f"{self.name}: a history contributed {self._rows_this_file} rows "
                f"but the ensemble is {self.ntime} rows wide; streaming must cover "
                "every row of every history exactly once"
            )

    def mean_array(self) -> np.ndarray:
        return np.asarray(self.mean)

    def sem(self) -> Optional[np.ndarray]:
        """Between-file standard error, or ``None`` for a single history.

        Identical to ``stack.std(axis=0, ddof=1) / sqrt(n)`` on the stack this
        accumulator replaces; ``M2 / (n - 1)`` is that same sample variance.
        """
        if self.count < 2:
            return None
        variance = np.asarray(self.m2) / (self.count - 1)
        return np.sqrt(variance) / np.sqrt(self.count)


@dataclass
class TimeGridCheck:
    """Incremental verification that every history shares one time grid.

    The whole-file path compares complete time columns with
    ``np.array_equal``.  Streaming compares the same values slice by slice
    against the grid the first history established, and additionally carries
    the last value across a chunk boundary so that strict increase is checked
    at the joins as well as inside them.
    """

    reference: Optional[np.ndarray] = None
    _previous: Optional[float] = None
    _rows: int = 0

    def begin_file(self) -> None:
        self._previous = None
        self._rows = 0

    def observe(self, path, start: int, times: np.ndarray) -> None:
        if times.size == 0:
            return
        if self.reference is None:
            raise StreamingError("the reference time grid must be set before comparing")
        stop = start + times.size
        if stop > self.reference.size:
            raise StreamingError(
                f"{path}: has more rows than the first history; no interpolation "
                "or truncation is performed"
            )
        if not np.array_equal(times, self.reference[start:stop]):
            first_bad = int(np.flatnonzero(times != self.reference[start:stop])[0])
            row = start + first_bad
            raise StreamingError(
                f"{path}: time grid differs from the first history at row {row} "
                f"({times[first_bad]!r} vs {self.reference[row]!r}); no "
                "interpolation or truncation is performed"
            )
        if self._previous is not None and times[0] <= self._previous:
            raise StreamingError(
                f"{path}: population time must strictly increase; row {start} "
                f"does not exceed row {start - 1}"
            )
        if times.size > 1 and np.any(np.diff(times) <= 0):
            bad = int(np.flatnonzero(np.diff(times) <= 0)[0])
            raise StreamingError(
                f"{path}: population time must strictly increase; row "
                f"{start + bad + 1} does not exceed row {start + bad}"
            )
        self._previous = float(times[-1])
        self._rows += int(times.size)

    def finish_file(self, path) -> None:
        if self.reference is not None and self._rows != self.reference.size:
            raise StreamingError(
                f"{path}: {self._rows} rows against {self.reference.size} in the "
                "first history; no interpolation or truncation is performed"
            )


@dataclass
class ConservationTally:
    """Running population-range and sum diagnostics over streamed chunks."""

    atol: float
    minimum: float = float("inf")
    maximum: float = float("-inf")
    total_min: float = float("inf")
    total_max: float = float("-inf")
    checked_rows: int = 0

    def observe(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        totals = values.sum(axis=1)
        self.total_min = min(self.total_min, float(totals.min()))
        self.total_max = max(self.total_max, float(totals.max()))
        self.checked_rows += int(values.shape[0])

    def in_unit_range(self) -> bool:
        return self.minimum >= -self.atol and self.maximum <= 1.0 + self.atol

    def conserved(self) -> bool:
        return (
            abs(self.total_min - 1.0) <= self.atol and abs(self.total_max - 1.0) <= self.atol
        )


def summarize_totals(totals_min: float, totals_max: float, mean_total: np.ndarray) -> Dict[str, Any]:
    """The conservation block reported alongside any ensemble population."""
    return {
        "min": float(np.min(mean_total)),
        "max": float(np.max(mean_total)),
        "mean": float(np.mean(mean_total)),
        "max_abs_deviation_from_one": float(np.max(np.abs(mean_total - 1.0))),
        "per_file_min": float(totals_min),
        "per_file_max": float(totals_max),
    }


def chunk_plan(n_rows: int, chunk_rows: int) -> List[Any]:
    """Row slices a streaming pass will use, for reporting and for tests."""
    if chunk_rows < 1:
        raise StreamingError("chunk_rows must be at least 1")
    return [(start, min(start + chunk_rows, n_rows)) for start in range(0, n_rows, chunk_rows)]
