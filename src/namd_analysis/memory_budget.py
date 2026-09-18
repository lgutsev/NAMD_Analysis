"""Estimate what a character run will allocate, and decide how to fit a budget.

The point is to decide *before* reading anything, so that a campaign too large
for the node fails in seconds rather than after an hour of PROCAR parsing, and
so that the choices made -- retain or drop per-history results, spill the
accumulators or keep them in RAM -- are stated rather than implicit.

Nothing here touches a numerical tolerance.  Memory management chooses where
arrays live; it never changes what is computed or what is validated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

BYTES_PER_FLOAT = 8

#: Assume this much headroom for the interpreter, numpy scratch, matplotlib and
#: the operating system before the estimate is compared with a budget.  It is a
#: floor, not a measurement, and the report says so.
BASE_OVERHEAD_BYTES = 512 * 1024 * 1024

#: Suffix multipliers for a human budget string.
_UNITS = {"": 1, "B": 1, "K": 1024, "KB": 1024, "M": 1024**2, "MB": 1024**2,
          "G": 1024**3, "GB": 1024**3, "T": 1024**4, "TB": 1024**4}
_SIZE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]*)\s*$")

RETAIN_CHOICES = ("auto", "yes", "no")


class BudgetError(ValueError):
    """Raised when a memory budget cannot be satisfied or cannot be parsed."""


def parse_size(text: str) -> int:
    """``"24G"`` -> bytes.  Accepts B/K/M/G/T, decimal values, no suffix = bytes."""
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        value = int(text)
        if value <= 0:
            raise BudgetError("memory budget must be positive")
        return value
    match = _SIZE.match(str(text))
    if not match:
        raise BudgetError(
            f"cannot read {text!r} as a size; use forms like 24G, 512M, 1073741824"
        )
    number, unit = match.group(1), match.group(2).upper()
    if unit not in _UNITS:
        raise BudgetError(
            f"unknown size unit {match.group(2)!r}; use one of "
            f"{sorted(u for u in _UNITS if u)}"
        )
    value = int(float(number) * _UNITS[unit])
    if value <= 0:
        raise BudgetError("memory budget must be positive")
    return value


def human(nbytes: float) -> str:
    """Bytes as a short human string, for report lines and error messages."""
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TiB"


@dataclass
class Allocation:
    """One array the run will hold, and what its size scales with."""

    name: str
    nbytes: int
    scales_with: str
    detail: str
    resident: bool = True
    spillable: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "bytes": int(self.nbytes),
            "human": human(self.nbytes),
            "scales_with": self.scales_with,
            "detail": self.detail,
            "resident": self.resident,
            "spillable": self.spillable,
        }


@dataclass
class MemoryPlan:
    """What the run will allocate, what it will do about it, and why."""

    allocations: List[Allocation]
    budget_bytes: Optional[int]
    retain_per_file: bool
    retain_choice: str
    spill_accumulators: bool
    memmap_dir: Optional[str]
    chunk_rows: int
    decisions: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    @property
    def resident_bytes(self) -> int:
        return sum(a.nbytes for a in self.allocations if a.resident)

    @property
    def total_with_overhead(self) -> int:
        return self.resident_bytes + BASE_OVERHEAD_BYTES

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "budget_bytes": self.budget_bytes,
            "budget_human": None if self.budget_bytes is None else human(self.budget_bytes),
            "estimated_resident_bytes": self.resident_bytes,
            "estimated_resident_human": human(self.resident_bytes),
            "assumed_overhead_bytes": BASE_OVERHEAD_BYTES,
            "estimated_total_human": human(self.total_with_overhead),
            "fits_budget": None if self.budget_bytes is None else self.ok,
            "retain_per_file": self.retain_per_file,
            "retain_per_file_choice": self.retain_choice,
            "spill_accumulators_to_memmap": self.spill_accumulators,
            "accumulator_memmap_dir": self.memmap_dir,
            "shprop_chunk_rows": self.chunk_rows,
            "allocations": [a.as_dict() for a in self.allocations],
            "decisions": list(self.decisions),
            "problems": list(self.problems),
            "note": (
                "an estimate of the arrays this run will hold, made before any file "
                "is opened. The overhead term is an assumed floor for the "
                "interpreter, numpy scratch and plotting, not a measurement. "
                "Memory choices never change a computed value or a validation "
                "tolerance"
            ),
        }


def estimate_memory(
    n_files: int,
    n_time: int,
    n_states: int,
    n_groups: int,
    n_fixed_groups: int,
    n_frames: int,
    n_bands: int,
    n_columns: int,
    chunk_rows: int,
    budget: Optional[int] = None,
    retain_per_file: str = "auto",
    memmap_dir: Optional[str] = None,
) -> MemoryPlan:
    """Decide retention and spilling from the sizes the plan already knows.

    The target shape is ``O(ntime x ngroups) + O(chunk_rows x nstates)``.  The
    one term that unavoidably carries ``n_files`` is the per-history result
    array, which is a diagnostic: the ensemble mean and standard error come
    from a running accumulator and never depend on it.  That is why it is the
    first thing dropped.
    """
    if retain_per_file not in RETAIN_CHOICES:
        raise BudgetError(
            f"--retain-per-file must be one of {list(RETAIN_CHOICES)}, not {retain_per_file!r}"
        )
    if chunk_rows < 1:
        raise BudgetError("chunk rows must be at least 1")

    f = BYTES_PER_FLOAT
    fixed = [
        Allocation(
            "projected_accumulator",
            2 * n_time * n_groups * f,
            "ntime x ngroups",
            "running mean and M2 for the projection-weighted populations",
            spillable=True,
        ),
        Allocation(
            "fixed_accumulator",
            2 * n_time * n_fixed_groups * f,
            "ntime x nfixed_groups",
            "running mean and M2 for the fixed-column comparison",
            spillable=True,
        ),
        Allocation(
            "time_grid",
            n_time * f,
            "ntime",
            "the one shared time grid every history is checked against",
        ),
        Allocation(
            "frame_alignment",
            n_files * n_time * 8,
            "nfiles x ntime",
            "one resolved frame index per row per history; what makes the "
            "alignment audit and the wrap count exact",
        ),
        Allocation(
            "projection_weights",
            n_frames * n_bands * n_groups * f,
            "nframes x nbands x ngroups",
            "normalized subsystem weights for the frames actually visited",
        ),
        Allocation(
            "chunk_buffer",
            chunk_rows * n_columns * f * 4,
            "chunk_rows x ncolumns",
            "one SHPROP chunk plus the Python float buffer it is built from",
        ),
        Allocation(
            "chunk_weights",
            chunk_rows * n_states * n_groups * f,
            "chunk_rows x nstates x ngroups",
            "per-chunk gather of the projection weights",
        ),
    ]
    per_file_bytes = n_files * n_time * n_groups * f
    decisions: List[str] = []

    keep = retain_per_file == "yes"
    if retain_per_file == "auto":
        overhead = sum(a.nbytes for a in fixed) + BASE_OVERHEAD_BYTES
        if budget is None:
            keep = per_file_bytes <= 256 * 1024 * 1024
            decisions.append(
                f"per-history results {'kept' if keep else 'dropped'}: "
                f"{human(per_file_bytes)} against a 256.0 MiB default ceiling "
                "(no --memory-budget given)"
            )
        else:
            keep = overhead + per_file_bytes <= budget
            decisions.append(
                f"per-history results {'kept' if keep else 'dropped'}: they would add "
                f"{human(per_file_bytes)} to {human(overhead)}, against a budget of "
                f"{human(budget)}"
            )
    elif retain_per_file == "yes":
        decisions.append(
            f"per-history results kept because --retain-per-file yes was given "
            f"({human(per_file_bytes)})"
        )
    else:
        decisions.append("per-history results dropped because --retain-per-file no was given")

    allocations = list(fixed)
    allocations.append(
        Allocation(
            "per_file_results",
            per_file_bytes,
            "nfiles x ntime x ngroups",
            "per-history projected populations; a diagnostic, never an input to "
            "the mean or the standard error",
            resident=keep,
        )
    )

    plan = MemoryPlan(
        allocations=allocations,
        budget_bytes=budget,
        retain_per_file=keep,
        retain_choice=retain_per_file,
        spill_accumulators=False,
        memmap_dir=memmap_dir,
        chunk_rows=chunk_rows,
        decisions=decisions,
    )

    if budget is not None and plan.total_with_overhead > budget:
        spillable = sum(a.nbytes for a in allocations if a.spillable and a.resident)
        if memmap_dir and plan.total_with_overhead - spillable <= budget:
            plan.spill_accumulators = True
            for allocation in allocations:
                if allocation.spillable:
                    allocation.resident = False
            plan.decisions.append(
                f"accumulators spilled to memory maps under {memmap_dir}: that moves "
                f"{human(spillable)} out of RAM and brings the estimate within budget"
            )
        else:
            over = plan.total_with_overhead - budget
            hint = (
                "give --accumulator-memmap-dir so the running mean and variance can "
                "spill to disk"
                if not memmap_dir
                else "the remainder does not fit even with the accumulators spilled"
            )
            plan.problems.append(
                f"estimated {human(plan.total_with_overhead)} against a budget of "
                f"{human(budget)}, over by {human(over)}. Largest terms: "
                + ", ".join(
                    f"{a.name} {human(a.nbytes)}"
                    for a in sorted(allocations, key=lambda x: -x.nbytes)[:3]
                    if a.resident
                )
                + f". {hint}."
            )
    elif plan.spill_accumulators is False and memmap_dir:
        plan.decisions.append(
            f"accumulators kept in RAM: {human(plan.resident_bytes)} is within budget, "
            f"so {memmap_dir} was not needed"
        )
    return plan
