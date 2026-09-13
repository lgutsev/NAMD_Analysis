"""Campaign inventory: what exists on disk, and which legacy fits failed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .io.hefei import KNOWN_FILES, load_run_directory, looks_like_run_directory

#: Legacy single-exponential fits at or below this R^2 are reported as failed.
FAILED_FIT_R2 = 0.9


@dataclass
class InventoryEntry:
    path: Path
    name: str
    present: List[str]
    missing: List[str]
    params: Dict[str, Any]
    nstates_from_inp: Optional[int]
    dt_fs_from_inp: Optional[float]
    legacy_tau_ns: Optional[float]
    legacy_r_squared: Optional[float]
    shprop_files: int
    bytes_total: int

    @property
    def legacy_fit_failed(self) -> Optional[bool]:
        if self.legacy_r_squared is None:
            return None
        return self.legacy_r_squared < FAILED_FIT_R2

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "present": self.present,
            "missing": self.missing,
            "nstates_from_inp": self.nstates_from_inp,
            "dt_fs_from_inp": self.dt_fs_from_inp,
            "algo": self.params.get("ALGO"),
            "nsample": self.params.get("NSAMPLE"),
            "ntraj": self.params.get("NTRAJ"),
            "namdtime": self.params.get("NAMDTIME"),
            "lhole": self.params.get("LHOLE"),
            "shprop_files": self.shprop_files,
            "bytes_total": self.bytes_total,
            "legacy_tau_ns": self.legacy_tau_ns,
            "legacy_r_squared": self.legacy_r_squared,
            "legacy_fit_failed": self.legacy_fit_failed,
        }


INVENTORY_HEADER = [
    "name",
    "path",
    "nstates_from_inp",
    "dt_fs_from_inp",
    "algo",
    "nsample",
    "ntraj",
    "namdtime",
    "lhole",
    "shprop_files",
    "present",
    "missing",
    "legacy_tau_ns",
    "legacy_r_squared",
    "legacy_fit_failed",
]


def _entry_row(entry: InventoryEntry) -> List[Any]:
    payload = entry.as_dict()
    payload["present"] = ";".join(entry.present)
    payload["missing"] = ";".join(entry.missing)
    return [payload[key] for key in INVENTORY_HEADER]


def scan(root, max_depth: int = 4) -> List[InventoryEntry]:
    """Find run directories under ``root`` without reading the large tables."""
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    entries: List[InventoryEntry] = []
    for candidate in _walk(root, max_depth):
        if not looks_like_run_directory(candidate):
            continue
        run = load_run_directory(candidate)
        shprop = sorted(candidate.glob("SHPROP.*"))
        total = 0
        for item in candidate.iterdir():
            if item.is_file():
                total += item.stat().st_size
        entries.append(
            InventoryEntry(
                path=candidate,
                name=candidate.name,
                present=run.present,
                missing=[f for f in KNOWN_FILES if f not in run.present],
                params=run.params,
                nstates_from_inp=run.declared_nstates,
                dt_fs_from_inp=run.dt_fs_from_inp,
                legacy_tau_ns=(run.legacy_fit or {}).get("tau_ns"),
                legacy_r_squared=(run.legacy_fit or {}).get("r_squared"),
                shprop_files=len(shprop),
                bytes_total=total,
            )
        )
    entries.sort(key=lambda e: str(e.path))
    return entries


def _walk(root: Path, max_depth: int) -> Iterable[Path]:
    stack = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        yield current
        if depth >= max_depth:
            continue
        try:
            children = sorted(p for p in current.iterdir() if p.is_dir())
        except (PermissionError, OSError):
            continue
        for child in children:
            if child.name.startswith("__MACOSX"):
                continue
            stack.append((child, depth + 1))


def summarize(entries: List[InventoryEntry]) -> Dict[str, Any]:
    failed = [e for e in entries if e.legacy_fit_failed]
    return {
        "n_run_directories": len(entries),
        "n_with_legacy_fit": sum(1 for e in entries if e.legacy_r_squared is not None),
        "n_failed_legacy_fits": len(failed),
        "failed_legacy_fit_threshold_r2": FAILED_FIT_R2,
        "failed_legacy_fits": [
            {
                "name": e.name,
                "path": str(e.path),
                "tau_ns": e.legacy_tau_ns,
                "r_squared": e.legacy_r_squared,
            }
            for e in failed
        ],
        "n_with_shprop": sum(1 for e in entries if e.shprop_files > 0),
    }


def rows(entries: List[InventoryEntry]) -> List[List[Any]]:
    return [_entry_row(entry) for entry in entries]
