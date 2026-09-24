"""A small synthetic production campaign for the robustness and full-space tests.

Six ions in three fragments, a handful of frames and bands, PROCARs written in
VASP's own dialect, and a ``projection_character.csv`` produced by the real
character code path (``load_projection_series`` + ``projection_table_rows``),
so the audit is always tested against what ``character-populations`` writes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from character_fixtures import write_procar

from namd_analysis.character import AtomGroupMap, load_projection_series, projection_table_rows
from namd_analysis.report import write_csv

GROUPS = {"perovskite": [1, 2], "BCF": [3, 4], "PCBM": [5, 6]}
GROUP_NAMES = list(GROUPS)
NIONS = 6

PROJECTION_CHARACTER_HEADER = [
    "frame",
    "band",
    "group",
    "normalized_weight",
    "captured_projection",
    "total_projection",
    "dominant_group",
    "dominant_weight",
]

POSCAR = """synthetic six-ion cell
1.0
  10.0 0.0 0.0
  0.0 10.0 0.0
  0.0 0.0 10.0
Pb C H
2 2 2
Direct
0.10 0.10 0.10
0.30 0.10 0.10
0.10 0.50 0.50
0.30 0.50 0.50
0.60 0.80 0.80
0.80 0.80 0.80
"""

INCAR = """SYSTEM = synthetic
ENCUT = 400
PREC = Normal ! comment
ISMEAR = 0; SIGMA = 0.01
NBANDS = 1200
LORBIT = 11
NSW = 1
IBRION = 0
LWAVE = .FALSE.
"""


def ion_values(raw: Sequence[float]) -> list:
    """Spread each fragment's raw weight evenly over its two ions."""
    values = [0.0] * NIONS
    for weight, atoms in zip(raw, GROUPS.values()):
        for atom in atoms:
            values[atom - 1] = float(weight) / len(atoms)
    return values


def write_campaign(
    root: Path,
    raw: Mapping[int, Mapping[int, Tuple[float, float, float]]],
    decimals: int = 6,
    incar: str = INCAR,
) -> Dict[str, Path]:
    """Frame directories, manifest, atom groups and projection_character.csv.

    ``raw[frame][band]`` is ``(W_perovskite, W_BCF, W_PCBM)``.
    """
    root = Path(root)
    frames = sorted(raw)
    records = []
    for frame in frames:
        directory = root / "production" / f"{frame:04d}"
        directory.mkdir(parents=True)
        write_procar(
            directory / "PROCAR",
            {band: ion_values(weights) for band, weights in raw[frame].items()},
            orbitals="spd",
            decimals=decimals,
        )
        (directory / "INCAR").write_text(incar)
        (directory / "POSCAR").write_text(POSCAR)
        (directory / "POTCAR").write_text("synthetic POTCAR\n")
        records.append({"frame": frame, "procar": str(directory / "PROCAR")})
    manifest = root / "projection_manifest.json"
    manifest.write_text(json.dumps({"frames": records}))
    groups_path = root / "atom_groups.json"
    groups_path.write_text(
        json.dumps({"groups": GROUPS, "complete_atoms": True, "min_projection_weight": 0.5})
    )
    bands = sorted(next(iter(raw.values())))
    series = load_projection_series(manifest, AtomGroupMap.from_json(groups_path), bands)
    character = root / "projection_character.csv"
    write_csv(character, PROJECTION_CHARACTER_HEADER, projection_table_rows(series))
    return {
        "root": root,
        "manifest": manifest,
        "atom_groups": groups_path,
        "projection_character": character,
        "series": series,
    }


#: The four behaviours the audit must tell apart, as (W_perovskite, W_BCF, W_PCBM).
SCENARIOS = {
    # Band 10: substantial raw weight on both BCF and PCBM, high capture.
    "genuinely_mixed": (0.20, 0.30, 0.30),
    # Band 11: tiny absolute weights, which normalization inflates to ~40% each.
    "tiny_but_normalized": (0.010, 0.021, 0.019),
    # Band 12: almost the whole band on the perovskite.
    "high_capture_pure": (0.95, 0.0, 0.0),
    # Band 13: low capture, but all of it on one fragment.
    "low_capture_unmixed": (0.30, 0.0, 0.0),
}
SCENARIO_BAND = {name: 10 + i for i, name in enumerate(SCENARIOS)}


def scenario_raw(n_frames: int = 6, jitter: float = 0.002, seed: int = 7):
    """Every frame carries all four scenarios, each on its own band."""
    rng = np.random.default_rng(seed)
    raw = {}
    for frame in range(1, n_frames + 1):
        raw[frame] = {}
        for name, weights in SCENARIOS.items():
            noisy = np.clip(np.asarray(weights) + rng.uniform(-jitter, jitter, 3) * (np.asarray(weights) > 0), 0.0, None)
            raw[frame][SCENARIO_BAND[name]] = tuple(float(v) for v in noisy)
    return raw
