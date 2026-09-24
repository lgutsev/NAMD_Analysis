"""Read the few VASP settings that decide what a PROCAR weight means.

Only what the projection audit needs: ``LORBIT`` and ``RWIGS`` from an
INCAR (what was requested), an OUTCAR or a ``vasprun.xml`` (what VASP ran
with), plus a general INCAR reader the validation campaign uses to derive its
inputs from the production INCAR.  Nothing is inferred from defaults: a tag
that is absent is reported absent.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class VaspSettingsError(ValueError):
    """Raised when a VASP input or output cannot be read without guessing."""


_TAG = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def _strip_comment(text: str) -> str:
    for marker in ("!", "#"):
        position = text.find(marker)
        if position >= 0:
            text = text[:position]
    return text


def parse_incar_text(text: str) -> List[Tuple[str, str]]:
    """``(TAG, value)`` pairs in file order, upper-cased tags, comments removed.

    ``;`` separates several assignments on one line, as VASP allows.  A line
    that is not an assignment is ignored only when it is blank or a comment;
    anything else is an error, because silently skipping a malformed line
    would drop a setting from a derived INCAR.
    """
    pairs: List[Tuple[str, str]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw)
        for piece in line.split(";"):
            if not piece.strip():
                continue
            match = _TAG.match(piece)
            if match is None:
                raise VaspSettingsError(
                    f"INCAR line {lineno}: cannot read {raw.strip()!r} as TAG = value"
                )
            pairs.append((match.group(1).upper(), match.group(2).strip()))
    return pairs


def read_incar(path) -> Dict[str, str]:
    """Tag -> raw value.  A tag set twice is an error, not last-wins."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise VaspSettingsError(f"cannot read INCAR {path}: {exc}") from exc
    tags: Dict[str, str] = {}
    for tag, value in parse_incar_text(text):
        if tag in tags and tags[tag] != value:
            raise VaspSettingsError(
                f"{path}: {tag} is set twice ({tags[tag]!r} and {value!r}); VASP "
                "would use one of them and a derived INCAR must not guess which"
            )
        tags[tag] = value
    return tags


def _int_value(text: str, what: str) -> int:
    token = text.split()[0] if text.split() else ""
    try:
        return int(float(token))
    except ValueError:
        raise VaspSettingsError(f"{what}: {text!r} is not an integer") from None


def incar_lorbit(tags: Dict[str, str]) -> Optional[int]:
    return _int_value(tags["LORBIT"], "INCAR LORBIT") if "LORBIT" in tags else None


def incar_rwigs(tags: Dict[str, str]) -> Optional[List[float]]:
    if "RWIGS" not in tags:
        return None
    try:
        return [float(item) for item in tags["RWIGS"].split()]
    except ValueError:
        raise VaspSettingsError(f"INCAR RWIGS {tags['RWIGS']!r} is not a list of numbers") from None


_OUTCAR_LORBIT = re.compile(r"^\s*LORBIT\s*=\s*(-?\d+)")
_OUTCAR_RWIGS = re.compile(r"^\s*RWIGS\s*=\s*(.*?)\s*wigner-seitz radius for each atom type", re.I)

#: OUTCAR lines scanned before giving up.  The parameter block that holds
#: LORBIT is written before the first ionic step, a few thousand lines in.
OUTCAR_SCAN_LINES = 200_000


def outcar_settings(path) -> Dict[str, object]:
    """LORBIT and the per-type RWIGS line from an OUTCAR's parameter block.

    The POTCAR headers in an OUTCAR also print an ``RWIGS`` for each species;
    that is the pseudopotential's own default, not the run's setting, and is
    deliberately not matched.
    """
    path = Path(path)
    lorbit: Optional[int] = None
    rwigs: Optional[List[float]] = None
    rwigs_text: Optional[str] = None
    try:
        with path.open("r", errors="replace") as handle:
            for lineno, line in enumerate(handle, start=1):
                if lorbit is None:
                    match = _OUTCAR_LORBIT.match(line)
                    if match:
                        lorbit = int(match.group(1))
                if rwigs_text is None:
                    match = _OUTCAR_RWIGS.match(line)
                    if match:
                        rwigs_text = match.group(1)
                        try:
                            rwigs = [float(item) for item in rwigs_text.split()]
                        except ValueError:
                            rwigs = None
                if (lorbit is not None and rwigs_text is not None) or lineno > OUTCAR_SCAN_LINES:
                    break
    except OSError as exc:
        raise VaspSettingsError(f"cannot read OUTCAR {path}: {exc}") from exc
    return {"LORBIT": lorbit, "RWIGS": rwigs, "RWIGS_text": rwigs_text}


_VASPRUN_LORBIT = re.compile(r'<i\s+type="int"\s+name="LORBIT">\s*(-?\d+)\s*</i>')


def vasprun_lorbit(path) -> Optional[int]:
    """The LORBIT VASP recorded in ``vasprun.xml``; streamed, stops when found."""
    path = Path(path)
    try:
        with path.open("r", errors="replace") as handle:
            for lineno, line in enumerate(handle, start=1):
                match = _VASPRUN_LORBIT.search(line)
                if match:
                    return int(match.group(1))
                if lineno > OUTCAR_SCAN_LINES:
                    break
    except OSError as exc:
        raise VaspSettingsError(f"cannot read vasprun.xml {path}: {exc}") from exc
    return None


def render_incar(tags: Dict[str, str], header: str = "") -> str:
    """Write tags back one per line, in the given order."""
    lines = [f"# {line}" for line in header.splitlines()] if header else []
    width = max((len(tag) for tag in tags), default=0)
    lines += [f"{tag.ljust(width)} = {value}" for tag, value in tags.items()]
    return "\n".join(lines) + "\n"
