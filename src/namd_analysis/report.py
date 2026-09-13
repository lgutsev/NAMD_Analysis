"""Output-directory handling, JSON reports and CSV tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


class OutputExistsError(RuntimeError):
    """Raised when an output directory already holds results."""


def prepare_output(path, overwrite: bool = False) -> Path:
    """Create the output directory.

    A non-empty existing directory is an error unless ``overwrite`` is set,
    so a report is never silently mixed with an earlier one.
    """
    out = Path(path)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise OutputExistsError(
            f"{out} already exists and is not empty; choose a new folder or pass --overwrite"
        )
    out.mkdir(parents=True, exist_ok=True)
    return out


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, payload: Dict[str, Any]) -> Path:
    path = Path(path)
    path.write_text(json.dumps(_plain(payload), indent=2) + "\n", encoding="utf-8")
    return path


def write_csv(path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> Path:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if item is None else _plain(item) for item in row])
    return path
