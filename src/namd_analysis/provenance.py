"""Input fingerprints and run metadata written into every report."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import sys
from functools import lru_cache
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
                "hashed": True,
            }
        )
    return records


def describe(paths: Iterable) -> List[Dict[str, Any]]:
    """Size and mtime for each input, without reading its contents.

    Hashing identifies a file beyond doubt, but it costs a full read. A
    campaign of two thousand PROCARs is a hundred gigabytes, so hashing them
    to write a report would double the cost of the analysis that just read
    them once. This records what can be known from the directory entry, and
    says plainly that it is not a hash.
    """
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
                "sha256": None,
                "hashed": False,
            }
        )
    return records


def _porcelain_paths(status: str | None) -> List[str]:
    """Paths out of ``git status --porcelain`` output.

    Each line is two status columns, a space, then the path; a rename adds
    ``ORIG -> NEW`` and the new name is the one that matters.
    """
    paths = []
    for line in (status or "").splitlines():
        if len(line) <= 3:
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path:
            paths.append(path.strip('"'))
    return paths


@lru_cache(maxsize=1)
def _code_provenance_uncached() -> str:
    """Which implementation produced this output, as exactly as it can be known.

    A version string alone is not enough for a production run. ``0.7.2`` names
    a release; it does not distinguish the commit that release was cut from,
    still less an editable checkout that has moved on since. A/B/C results are
    only comparable if they came from the *same* implementation, and "same
    version" is a weaker claim than "same commit".

    So when the package is imported from a git working tree -- which is what
    ``pip install -e .`` gives -- the commit SHA, branch and dirty flag are
    recorded beside the version. ``dirty`` is the one that matters most: it
    says uncommitted changes were present, so the SHA alone does not identify
    what ran.

    Never raises. Git may be absent, the checkout may be a tarball, the
    repository may be owned by another user; provenance that could abort an
    analysis would be worse than provenance that says it does not know.

    Cached, and serialized to JSON so the cache cannot hand a caller a dict it
    might mutate. The commit cannot change while a process runs, and every
    report asks for this, so a production run must not pay for several git
    subprocesses per output file on a shared filesystem.
    """
    from . import __version__

    record: Dict[str, Any] = {
        "version": __version__,
        "package_path": None,
        "editable_checkout": False,
        "git": {"available": False, "reason": "not attempted"},
    }
    try:
        package_dir = Path(__file__).resolve().parent
        record["package_path"] = str(package_dir)
    except OSError as exc:  # pragma: no cover - defensive
        record["git"] = {"available": False, "reason": str(exc)}
        return json.dumps(record)

    import subprocess

    def _git(*args: str, keep_leading: bool = False) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(package_dir), *args],
                capture_output=True, text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            record["git"] = {"available": False, "reason": str(exc)}
            return None
        if done.returncode != 0:
            record["git"] = {
                "available": False,
                "reason": (done.stderr or "git returned a non-zero status").strip(),
            }
            return None
        # `--porcelain` puts the status flags in the first two columns, and an
        # unstaged change leaves column one blank. Stripping that blank shifts
        # every path by a character, so the leading space is kept where it is
        # load-bearing.
        if keep_leading:
            return done.stdout.rstrip("\r\n")
        return done.stdout.strip()

    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return json.dumps(record)

    status = _git("status", "--porcelain", keep_leading=True)
    record["editable_checkout"] = True
    record["git"] = {
        "available": True,
        "commit": commit,
        "commit_short": commit[:12],
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _git("describe", "--tags", "--always", "--dirty"),
        "dirty": bool(status),
        "uncommitted_paths": sorted(_porcelain_paths(status))[:50],
        "note": (
            "the package was imported from a git working tree, so the commit "
            "below identifies the implementation exactly. dirty=true means "
            "uncommitted changes were present and the commit alone does NOT "
            "identify what ran"
        ),
    }
    return json.dumps(record)


def code_provenance() -> Dict[str, Any]:
    """See :func:`_code_provenance_uncached`; this hands back a fresh copy."""
    return json.loads(_code_provenance_uncached())


def environment(argv: Iterable[str] | None = None) -> Dict[str, Any]:
    from . import __version__

    return {
        "tool": "namd-analysis",
        "version": __version__,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": list(argv) if argv is not None else list(sys.argv),
        # A version names a release; a commit names the implementation. A/B/C
        # are only comparable if they came from the same one.
        "code": code_provenance(),
    }


def launcher_manifests(paths=(), explicit=()):
    """Import adjacent launcher manifests verbatim; never trust/execute file paths in them.

    Only same-directory *_manifest.json and .namdforge/state.json are discovered.
    Campaign-level manifests can be passed explicitly to avoid unbounded ancestor scans.
    """
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
