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


def launcher_manifests(paths=(), explicit=()):
    """Import adjacent launcher manifests verbatim; never trust/execute file paths in them.

    Only same-directory *_manifest.json and .namdforge/state.json are discovered.
    Campaign-level manifests can be passed explicitly to avoid unbounded ancestor scans.
    """
    import json
    found = {Path(p).resolve() for p in explicit}
    for item in paths:
        parent = Path(item).resolve().parent
        found.update(p.resolve() for p in parent.glob("*_manifest.json"))
        state = parent / ".namdforge" / "state.json"
        if state.is_file():
            found.add(state.resolve())
    records = []
    for path in sorted(found):
        record = {"input": fingerprint([path])[0],
                  "verification": "manifest fingerprint only; upstream artifacts and claims not independently verified"}
        if path.stat().st_size > 10 * 1024 * 1024:
            record.update(status="not_imported", error="manifest exceeds 10 MiB")
        else:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("manifest root must be an object")
                record.update(status="imported", content=value)
            except (ValueError, UnicodeError) as exc:
                record.update(status="invalid", error=str(exc))
        records.append(record)
    return records
