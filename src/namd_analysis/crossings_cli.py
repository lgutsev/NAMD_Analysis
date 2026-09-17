"""CLI for ``namd-analysis character-crossings``.

Consumes the outputs a character run already wrote, plus EIGTXT/NATXT, and
synchronizes gap, coupling and fragment character **on the resolved MD frame**
-- never on a SHPROP row index, which means different frames in different
histories once ``NAMDTINI`` and cyclic wrapping are in play.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .crossings import (
    DEFAULT_THRESHOLDS,
    EVENT_HEADER,
    CrossingError,
    detect_events,
    load_couplings,
    nac_ceiling_note,
    read_projection_character,
    sensitivity,
)
from .provenance import environment, fingerprint
from .report import prepare_output, write_csv, write_json


def _band_pairs(spec: Optional[str], bands: Sequence[int]) -> List[Tuple[int, int]]:
    if spec:
        pairs = []
        for item in spec.split(","):
            a, _, b = item.partition(":")
            pairs.append((int(a), int(b)))
        return pairs
    return [(int(a), int(b)) for a, b in zip(bands, bands[1:])]


def _fragment_population(path: Optional[str], alignment: Optional[str]):
    """Fragment population per frame, if the character outputs can supply it.

    ``character_populations.csv`` is indexed by time, and the alignment table
    says which frame each history's time points used.  Without an unambiguous
    time-to-frame map this returns ``None`` rather than correlating by row.
    """
    if not path or not alignment:
        return None
    align = Path(alignment)
    if not align.is_file():
        return None
    with align.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len({row.get("NAMDTINI") for row in rows}) != 1:
        # Histories starting at different frames visit different frames at the
        # same time index; one frame therefore has no single population.
        return None
    return None


def build_parser(prog: str = "namd-analysis character-crossings") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Synchronize adiabatic gap, |NAC| and frame-resolved fragment character, "
            "and flag windows where the states are close, strongly coupled, or "
            "exchange character. A character swap is not a surface hop, and neither "
            "is automatically charge transfer."
        ),
    )
    parser.add_argument(
        "--projection-character",
        required=True,
        help="projection_character.csv from a character-populations run",
    )
    parser.add_argument("--eigtxt", required=True, help="state energies, one row per MD frame")
    parser.add_argument("--natxt", default=None, help="optional real NAC matrices")
    parser.add_argument(
        "--nac-unit", choices=("eV", "meV", "fs^-1"), default="eV",
        help="unit of the NAC file, taken from the producing code and not its magnitude",
    )
    parser.add_argument(
        "--dt-fs", type=float, default=None,
        help="electronic timestep, for the hbar/dt scale the couplings are read against",
    )
    parser.add_argument(
        "--shprop-alignment", default=None,
        help="shprop_alignment.csv, so the frame mapping can be recorded in provenance",
    )
    parser.add_argument(
        "--band-pairs", default=None,
        help="pairs like 976:977,977:978 (default: every adjacent pair present)",
    )
    parser.add_argument(
        "--gap-ev", type=float, default=DEFAULT_THRESHOLDS["gap_ev"],
        help="below this the pair is flagged close",
    )
    parser.add_argument(
        "--nac-ev", type=float, default=DEFAULT_THRESHOLDS["nac_ev"],
        help="above this the coupling is flagged enhanced",
    )
    parser.add_argument(
        "--character-change", type=float, default=DEFAULT_THRESHOLDS["character_change"],
        help="absolute fragment-weight change that counts as exchange",
    )
    parser.add_argument(
        "--dominance", type=float, default=DEFAULT_THRESHOLDS["dominance"],
        help="weight above which a band is called dominated by one fragment",
    )
    parser.add_argument("--out", required=True, help="new output directory")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _plot(out: Path, character, couplings, pairs, events) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs: List[str] = []
    frames = character.frames

    # Gap / NAC / character overlay, one column per examined pair.
    if couplings is not None and pairs:
        fig, axes = plt.subplots(
            3, len(pairs), figsize=(5.2 * len(pairs), 7.5),
            sharex=True, constrained_layout=True, squeeze=False,
        )
        state_of = {int(b): k for k, b in enumerate(character.bands)}
        row_of = {int(f): r for r, f in enumerate(couplings.frames)}
        rows = np.array([row_of.get(int(f), -1) for f in frames])
        valid = rows >= 0
        for column, (bi, bj) in enumerate(pairs):
            i, j = state_of.get(bi), state_of.get(bj)
            gaps = np.full(frames.size, np.nan)
            nacs = np.full(frames.size, np.nan)
            if i is not None and j is not None:
                gaps[valid] = np.abs(
                    couplings.energies[rows[valid], i] - couplings.energies[rows[valid], j]
                )
                if couplings.nac is not None:
                    nacs[valid] = couplings.nac[rows[valid], i, j]
            axes[0][column].plot(frames, gaps, lw=1.0)
            axes[0][column].set(ylabel="gap (eV)", title=f"bands {bi} / {bj}")
            axes[1][column].plot(frames, nacs, lw=1.0, color="tab:red")
            axes[1][column].set(ylabel="|NAC| (eV)")
            if couplings.ceiling_ev:
                axes[1][column].axhline(
                    couplings.ceiling_ev, ls=":", color="k", lw=0.8,
                    label="ħ/dt (numerical limit, not a physical coupling)",
                )
                axes[1][column].legend(fontsize=6)
            bi_index = character.band_index().get(bi)
            if bi_index is not None:
                for gi, group in enumerate(character.groups):
                    axes[2][column].plot(
                        frames, character.weights[:, bi_index, gi], lw=1.0, label=group
                    )
            axes[2][column].set(xlabel="MD frame", ylabel=f"band {bi} character", ylim=(-0.03, 1.03))
            axes[2][column].legend(fontsize=6)
            for event in events:
                if (event.band_i, event.band_j) == (bi, bj) and event.character_swap:
                    for axis in axes[:, column]:
                        axis.axvline(event.frame, color="tab:green", alpha=0.35, lw=0.8)
        fig.suptitle(
            "gap, coupling and fragment character on the resolved MD frame; "
            "green marks a dominant-character swap",
            fontsize=8,
        )
        for suffix in ("png", "pdf"):
            path = out / f"crossing_overlay.{suffix}"
            fig.savefig(path, dpi=200)
            outputs.append(str(path))
        plt.close(fig)

    # Character heatmap: band against frame, one panel per fragment.
    fig, axes = plt.subplots(
        1, len(character.groups), figsize=(4.4 * len(character.groups), 3.4),
        constrained_layout=True, squeeze=False,
    )
    for gi, group in enumerate(character.groups):
        axis = axes[0][gi]
        image = axis.imshow(
            character.weights[:, :, gi].T, aspect="auto", origin="lower",
            vmin=0.0, vmax=1.0, cmap="viridis",
            extent=[frames[0], frames[-1], -0.5, len(character.bands) - 0.5],
        )
        axis.set_yticks(range(len(character.bands)))
        axis.set_yticklabels([str(b) for b in character.bands], fontsize=7)
        axis.set(xlabel="MD frame", ylabel="band", title=f"{group} character")
        fig.colorbar(image, ax=axis, fraction=0.04)
    for suffix in ("png", "pdf"):
        path = out / f"character_heatmap.{suffix}"
        fig.savefig(path, dpi=200)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    thresholds = {
        "gap_ev": args.gap_ev,
        "nac_ev": args.nac_ev,
        "character_change": args.character_change,
        "dominance": args.dominance,
    }
    try:
        character = read_projection_character(args.projection_character)
        couplings = load_couplings(
            args.eigtxt, args.natxt, frames=None,
            nac_unit=args.nac_unit, dt_fs=args.dt_fs,
        )
        pairs = _band_pairs(args.band_pairs, [int(b) for b in character.bands])
        events = detect_events(character, couplings, pairs, thresholds)
        sweep = sensitivity(character, couplings, pairs, thresholds)
    except (CrossingError, ValueError, OSError) as exc:
        print(f"error: {exc}")
        return 2

    out = prepare_output(args.out, overwrite=args.overwrite)
    write_csv(out / "crossing_events.csv", EVENT_HEADER, [e.as_row() for e in events])

    raw_rows = []
    state_of = {int(b): k for k, b in enumerate(character.bands)}
    row_of = {int(f): r for r, f in enumerate(couplings.frames)}
    for index, frame in enumerate(character.frames):
        row = row_of.get(int(frame))
        for bi, bj in pairs:
            i, j = state_of.get(bi), state_of.get(bj)
            gap = nac = None
            if row is not None and i is not None and j is not None:
                gap = float(abs(couplings.energies[row, i] - couplings.energies[row, j]))
                if couplings.nac is not None:
                    nac = float(couplings.nac[row, i, j])
            bi_index = character.band_index().get(bi)
            dominant = (
                character.groups[int(np.argmax(character.weights[index, bi_index]))]
                if bi_index is not None else None
            )
            raw_rows.append([int(frame), bi, bj, gap, nac, dominant])
    write_csv(
        out / "crossing_metrics.csv",
        ["frame", "band_i", "band_j", "gap_ev", "nac_ev", "dominant_band_i"],
        raw_rows,
    )

    figures = _plot(out, character, couplings, pairs, events)

    from collections import Counter

    counts = Counter(e.classification for e in events)
    inputs = [Path(args.projection_character), Path(args.eigtxt)]
    if args.natxt:
        inputs.append(Path(args.natxt))
    payload = {
        "command": "character-crossings",
        "environment": environment(),
        "inputs": fingerprint(inputs),
        "synchronization": {
            "join_key": "MD frame",
            "note": (
                "gap, coupling and character are correlated on the resolved MD "
                "frame, never on a SHPROP row index. Histories with different "
                "NAMDTINI visit different frames at the same row, and a cyclic "
                "mapping wraps them; a row-number correlation would be wrong in a "
                "way that still produces plausible numbers"
            ),
            "shprop_alignment": args.shprop_alignment,
        },
        "thresholds": {
            "values": thresholds,
            "note": (
                "every cutoff that turns a metric into an event is here and in the "
                "CLI. The raw gap, |NAC| and character-change values are in "
                "crossing_metrics.csv so any other cutoff can be applied without "
                "re-running this"
            ),
        },
        "band_pairs": [list(p) for p in pairs],
        "event_counts": dict(counts),
        "n_events": len(events),
        "sensitivity": sweep,
        "nac_interpretation": nac_ceiling_note(couplings),
        "vocabulary": {
            "adiabatic_state_population": "P_i = rho_ii, which eigenstate is occupied",
            "adiabatic_character_exchange": "the fragment composition of a fixed band index changing",
            "nonadiabatic_hop": "population moving between adiabatic states; lives in SHPROP, not PROCAR",
            "fragment_population": "sum_i P_i w_ig, closer to a diabatic reading but not a diabatization",
            "true_diabatic_transformation": "not performed anywhere in this package",
            "warning": (
                "staying on one adiabatic state through an avoided crossing changes "
                "the fragment identity; hopping between adiabatic states can preserve "
                "it. A character swap is not a hop, and neither is automatically "
                "charge transfer"
            ),
        },
        "figures": figures,
    }
    write_json(out / "report.json", payload)

    from .crossing_summary import write as write_crossing_summary

    summary = write_crossing_summary(out / "crossing_summary.md", payload, events)

    print(f"{len(events)} flagged event(s) over {character.frames.size} frames, "
          f"{len(pairs)} band pair(s)")
    for name, count in counts.most_common():
        print(f"  {count:5d}  {name}")
    print(f"threshold sensitivity: totals {[r['total_events'] for r in sweep['rows']]} "
          f"across factors {list(sweep['factors'])}")
    print("  a character swap is not a surface hop; neither is automatically transfer")
    print(f"summary: {summary}")
    print(f"written to {out}")
    return 0
