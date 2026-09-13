"""Minimal Fortran namelist reader for the Hefei-NAMD ``inp`` file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

_TRUE = {".true.", ".t.", "t", "true"}
_FALSE = {".false.", ".f.", "f", "false"}


def _coerce(token: str) -> Any:
    text = token.strip().rstrip(",").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    lowered = text.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text.replace("d", "e").replace("D", "E"))
    except ValueError:
        return text


def read_namelist(path) -> Dict[str, Any]:
    """Parse a single-group namelist into ``{UPPERCASE_KEY: value}``.

    Unknown keys are kept as-is; nothing is defaulted or invented.
    """
    path = Path(path)
    values: Dict[str, Any] = {}
    with path.open("r", errors="replace") as handle:
        for raw in handle:
            line = raw.split("!")[0].strip()
            if not line or line.startswith("&") or line == "/":
                continue
            if "=" not in line:
                continue
            key, _, rhs = line.partition("=")
            key = key.strip().upper()
            if key:
                values[key] = _coerce(rhs)
    return values


def basis_size(params: Dict[str, Any]):
    """Number of states implied by BMIN/BMAX, or ``None`` if absent."""
    bmin, bmax = params.get("BMIN"), params.get("BMAX")
    if isinstance(bmin, int) and isinstance(bmax, int):
        return bmax - bmin + 1
    return None
