"""Fixtures that imitate a real BCF/PCBM campaign layout on disk.

Small enough to run in a unit test, shaped like the archive the preparation
command is meant to read: numbered zero-padded frame directories each holding
a PROCAR, and SHPROP histories whose headers carry the basis window and the
sampling origin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


def write_campaign_procar(
    path: Path,
    n_ions: int,
    bands: Sequence[int],
    weights_by_band: Optional[Dict[int, np.ndarray]] = None,
    n_kpoints: int = 1,
) -> Path:
    """A minimal single-k-point PROCAR with per-ion ``tot`` columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "PROCAR lm decomposed",
        f"# of k-points:    {n_kpoints}         # of bands:   {len(bands)}         "
        f"# of ions:    {n_ions}",
        "",
        " k-point     1 :    0.00000000 0.00000000 0.00000000     weight = 1.00000000",
        "",
    ]
    for band in bands:
        values = (
            weights_by_band[band]
            if weights_by_band and band in weights_by_band
            else np.full(n_ions, 1.0 / n_ions)
        )
        lines.append(f"band     {band} # energy   -1.00000000 # occ.  1.00000000")
        lines.append("")
        lines.append("ion      s      p      d    tot")
        for ion in range(1, n_ions + 1):
            total = float(values[ion - 1])
            lines.append(
                f"{ion:4d}  {total:.4f} 0.0000 0.0000 {total:.4f}"
            )
        lines.append(f"tot    {values.sum():.4f} 0.0000 0.0000 {values.sum():.4f}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_campaign_shprop(
    path: Path,
    namdtini: int,
    populations: np.ndarray,
    bmin: int = 10,
    bmax: int = 15,
    nsw: int = 21,
    energy: Optional[np.ndarray] = None,
    time_step: float = 1.0,
    extra_columns: int = 0,
) -> Path:
    """A SHPROP history: time, an energy-like column, then populations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = populations.shape[0]
    times = np.arange(1, rows + 1, dtype=float) * time_step
    if energy is None:
        # Outside [0,1] on purpose: the energy column must be rejectable by
        # the same value test that accepts a population block.
        energy = np.linspace(-3.5, -2.5, rows)
    columns = [times, energy] + [populations[:, i] for i in range(populations.shape[1])]
    for index in range(extra_columns):
        columns.append(np.full(rows, 7.0 + index))
    table = np.column_stack(columns)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# NAMDTINI = {namdtini}\n")
        handle.write(f"# NSW = {nsw}\n")
        handle.write(f"# BMIN = {bmin}\n")
        handle.write(f"# BMAX = {bmax}\n")
        for row in table:
            handle.write(" ".join(f"{value:.10E}" for value in row) + "\n")
    return path


def six_state_populations(rows: int, seed: int = 0) -> np.ndarray:
    """Six columns that sum to one in every row."""
    rng = np.random.default_rng(seed)
    values = rng.dirichlet(np.ones(6), size=rows)
    return values


#: The confirmed FAPI_001_BCF_PCBM_A partition, as ranges.
A_PARTITION = {
    "perovskite": ["2-28", "134-268", "283-363", "364-417", "420-446"],
    "BCF": [1, "29-46", "119-133"],
    "PCBM": ["47-118", "269-282", "418-419"],
}
A_IONS = 446


def build_campaign(
    root: Path,
    frames: Iterable[int] = range(1, 21),
    padding: int = 4,
    n_ions: int = A_IONS,
    bands: Sequence[int] = tuple(range(10, 16)),
    namdtini: Sequence[int] = (1, 5),
    rows: int = 12,
    nsw: int = 21,
    extra_columns: int = 0,
    procar_name: str = "PROCAR",
    kpoints_first: int = 1,
) -> Dict[str, object]:
    """A campaign directory tree plus a SHPROP directory beside it."""
    root = Path(root)
    projection = root / "FAPI_001_BCF_PCBM_A"
    shprop_dir = root / "NuTest"
    frame_paths: Dict[int, Path] = {}
    for frame in frames:
        name = str(frame).zfill(padding)
        procar = projection / name / procar_name
        write_campaign_procar(
            procar,
            n_ions,
            bands,
            n_kpoints=kpoints_first if frame == min(frames) else 1,
        )
        frame_paths[frame] = procar

    shprop_paths: List[Path] = []
    for index, start in enumerate(namdtini):
        populations = six_state_populations(rows, seed=index)
        shprop_paths.append(
            write_campaign_shprop(
                shprop_dir / f"SHPROP.{start}",
                start,
                populations,
                bmin=bands[0],
                bmax=bands[-1],
                nsw=nsw,
                extra_columns=extra_columns,
            )
        )
    return {
        "root": root,
        "projection_dir": projection,
        "shprop_dir": shprop_dir,
        "shprop": shprop_paths,
        "frames": frame_paths,
        "n_ions": n_ions,
        "bands": list(bands),
    }
