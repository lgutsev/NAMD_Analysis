"""Input fingerprints and run metadata written into every report."""

from __future__ import annotations

import datetime as _dt
import hashlib
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(paths: Iterable) -> List[Dict[str, Any]]:
    """SHA-256, size and mtime for each input file, in the given order."""
    records = []
    for item in paths:
        path = Path(item)
        stat = path.stat()
        records.append(
            {
                "path": str(path),
                "bytes": stat.st_size,
                "modified_utc": _dt.datetime.fromtimestamp(
                    stat.st_mtime, _dt.timezone.utc
                ).isoformat(),
                "sha256": sha256_file(path),
            }
        )
    return records


def environment(argv: Iterable[str] | None = None) -> Dict[str, Any]:
    from . import __version__

    return {
        "tool": "namd-analysis",
        "version": __version__,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": list(argv) if argv is not None else list(sys.argv),
    }
