#!/usr/bin/env python3
"""
check_band_switching.py

Diagnose apparent one-frame KS-energy "kinks" in a trajectory composed of
numbered daughter directories (e.g. 0001/, 0002/, ...), each containing PROCAR.

The script is intended to distinguish:
  1) genuine spectrum-wide jumps,
  2) fixed-band index reordering / state switching,
  3) local character exchange among neighboring KS states.

Run from the ROOT directory containing the daughter frame directories.

Examples
--------
Inspect specific suspicious frames and +/- 2 neighbors:
    python check_band_switching.py --frames 1498 1510 1812 1849

Use a wider neighborhood:
    python check_band_switching.py --frames 1498 --radius 3

Automatically find large one-frame jumps for bands 976-981:
    python check_band_switching.py --auto

Change automatic jump threshold (eV):
    python check_band_switching.py --auto --jump-threshold 0.12

Write CSV plus text report:
    python check_band_switching.py --auto --csv band_switching.csv --report band_switching.txt

Notes
-----
* Frame directory names are treated as the actual frame labels.
* Energies are read directly from PROCAR without smoothing.
* Fragment grouping follows the same FAPbI3/BCF/PCBM convention used in the
  trajectory plotting workflow:
      first 27 C -> PVSK
      next 18 C  -> BCF
      remaining C -> PCBM
      O -> PCBM
      B,F -> BCF
      everything else -> PVSK
* The script does NOT decide that a crossing is "real" merely from a small gap.
  It reports energies and chemical weights so the user can inspect whether
  neighboring band indices exchange character.
"""

import os
import re
import csv
import argparse
from glob import glob
from collections import defaultdict

import numpy as np


# ---------------------------
# PROCAR / POSCAR parsing
# ---------------------------

def read_poscar_symbols(poscar):
    with open(poscar, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    toks5 = lines[5].split()

    def is_int_list(toks):
        try:
            [int(x) for x in toks]
            return True
        except Exception:
            return False

    if is_int_list(toks5):
        symbols = lines[0].split()
        counts = [int(x) for x in toks5]
    else:
        symbols = toks5
        counts = [int(x) for x in lines[6].split()]

    if len(symbols) != len(counts):
        raise ValueError(
            f"POSCAR parse mismatch: symbols={symbols}, counts={counts}"
        )

    expanded = []
    for sym, n in zip(symbols, counts):
        expanded.extend([sym] * n)
    return np.asarray(expanded)


def build_groups_from_rules(symbols):
    n = len(symbols)
    idx_all = np.arange(n, dtype=int)
    c_idx = idx_all[symbols == "C"]

    if len(c_idx) < 45:
        raise ValueError(
            f"Expected at least 45 C atoms for PVSK/BCF/PCBM grouping; found {len(c_idx)}"
        )

    pvsk_idx = set(c_idx[:27].tolist())
    bcf_idx = set(c_idx[27:45].tolist())
    pcbm_idx = set(c_idx[45:].tolist())

    bcf_idx.update(idx_all[np.isin(symbols, ["B", "F"])].tolist())
    pcbm_idx.update(idx_all[symbols == "O"].tolist())

    assigned = pvsk_idx | bcf_idx | pcbm_idx
    pvsk_idx.update(i for i in idx_all.tolist() if i not in assigned)

    return (
        np.asarray(sorted(pvsk_idx), dtype=int),
        np.asarray(sorted(bcf_idx), dtype=int),
        np.asarray(sorted(pcbm_idx), dtype=int),
    )


def parse_procar(procar, spin=0, kpoint=0):
    """
    Parse energies and ion-resolved total projection weights from a standard
    VASP PROCAR file.

    Returns
    -------
    energies : (nbands,)
    ion_weights : (nbands, nions)
    """
    with open(procar, "r") as f:
        lines = [line for line in f if line.strip()]

    nkpts, nbands, nions = [
        int(xx) for xx in re.sub(r"[^0-9]", " ", lines[1]).split()
    ]

    # Energies from band header lines
    energy_vals = np.asarray(
        [line.split()[-4] for line in lines if "occ." in line],
        dtype=float,
    )
    nspin = len(energy_vals) // (nkpts * nbands)
    energy_vals.resize(nspin, nkpts, nbands)
    energies = energy_vals[spin, kpoint, :].copy()

    # Numerical ion rows; last column is total projection
    weight_vals = np.asarray(
        [line.split()[-1] for line in lines if not re.search(r"[a-zA-Z]", line)],
        dtype=float,
    )
    nspin_w = len(weight_vals) // (nkpts * nbands * nions)
    if nspin_w != nspin:
        raise ValueError(
            f"Inconsistent spin count in {procar}: energies={nspin}, weights={nspin_w}"
        )
    weight_vals.resize(nspin, nkpts, nbands, nions)
    ion_weights = weight_vals[spin, kpoint, :, :].copy()

    return energies, ion_weights


def find_poscar(root, frame_dirs):
    root_poscar = os.path.join(root, "POSCAR")
    if os.path.isfile(root_poscar):
        return root_poscar

    for d in frame_dirs:
        p = os.path.join(d, "POSCAR")
        if os.path.isfile(p):
            return p

    raise FileNotFoundError("Could not find POSCAR in root or any frame directory.")


# ---------------------------
# Frame handling
# ---------------------------

def detect_frame_dirs(root="."):
    dirs = sorted(
        d for d in glob(os.path.join(root, "[0-9][0-9][0-9][0-9]"))
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "PROCAR"))
    )
    if not dirs:
        raise FileNotFoundError(
            "No 4-digit daughter directories containing PROCAR were found."
        )
    return dirs


def frame_label(path):
    return int(os.path.basename(os.path.normpath(path)))


def load_selected_frames(frame_dirs, wanted_labels, groups, band_numbers, spin=0, kpoint=0):
    pvsk_idx, bcf_idx, pcbm_idx = groups
    by_label = {frame_label(d): d for d in frame_dirs}
    records = {}

    for label in sorted(wanted_labels):
        if label not in by_label:
            continue

        procar = os.path.join(by_label[label], "PROCAR")
        energies, ion_weights = parse_procar(procar, spin=spin, kpoint=kpoint)

        w_pvsk = ion_weights[:, pvsk_idx].sum(axis=1)
        w_bcf = ion_weights[:, bcf_idx].sum(axis=1)
        w_pcbm = ion_weights[:, pcbm_idx].sum(axis=1)
        wtot = w_pvsk + w_bcf + w_pcbm + 1e-15

        fp = w_pvsk / wtot
        fb = w_bcf / wtot
        fc = w_pcbm / wtot

        rows = []
        for b in band_numbers:
            ib = b - 1
            if ib < 0 or ib >= len(energies):
                continue
            dominant = max(
                [("PVSK", fp[ib]), ("BCF", fb[ib]), ("PCBM", fc[ib])],
                key=lambda x: x[1]
            )[0]
            rows.append({
                "frame": label,
                "band": b,
                "energy": float(energies[ib]),
                "pvsk": float(fp[ib]),
                "bcf": float(fb[ib]),
                "pcbm": float(fc[ib]),
                "dominant": dominant,
            })

        records[label] = rows

    return records


def load_all_energies(frame_dirs, band_numbers, spin=0, kpoint=0):
    labels = []
    matrix = []

    for d in frame_dirs:
        label = frame_label(d)
        energies, _ = parse_procar(os.path.join(d, "PROCAR"), spin=spin, kpoint=kpoint)
        row = []
        for b in band_numbers:
            ib = b - 1
            row.append(energies[ib] if 0 <= ib < len(energies) else np.nan)
        labels.append(label)
        matrix.append(row)

    return np.asarray(labels, dtype=int), np.asarray(matrix, dtype=float)


# ---------------------------
# Diagnostics
# ---------------------------

def detect_jumps(labels, energies, band_numbers, threshold):
    events = []
    for i in range(1, len(labels)):
        # Only compare truly consecutive frame labels
        if labels[i] != labels[i-1] + 1:
            continue

        de = np.abs(energies[i] - energies[i-1])
        for j, val in enumerate(de):
            if np.isfinite(val) and val >= threshold:
                events.append({
                    "frame_prev": int(labels[i-1]),
                    "frame": int(labels[i]),
                    "band": int(band_numbers[j]),
                    "abs_jump": float(val),
                    "e_prev": float(energies[i-1, j]),
                    "e_curr": float(energies[i, j]),
                })
    return events


def spectrum_shift_score(labels, energies):
    """
    Median absolute shift over selected bands at each consecutive frame pair.
    Large values affecting many bands may indicate a spectrum-wide shift rather
    than a single-band identity switch.
    """
    out = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1] + 1:
            continue
        de = np.abs(energies[i] - energies[i-1])
        out.append(
            (int(labels[i]), float(np.nanmedian(de)), float(np.nanmax(de)))
        )
    return out


def print_frame_block(records, center_frames, radius, band_numbers):
    lines = []
    for center in center_frames:
        lines.append("=" * 104)
        lines.append(f"Center frame {center} (showing +/- {radius} frames)")
        lines.append("=" * 104)

        for fr in range(center - radius, center + radius + 1):
            if fr not in records:
                lines.append(f"\nFrame {fr}: MISSING / no PROCAR")
                continue

            lines.append(f"\nFrame {fr}")
            lines.append(
                " band      energy/eV    PVSK       BCF        PCBM       dominant"
            )
            lines.append(
                " -----     ---------    -------    -------    -------    --------"
            )
            for r in records[fr]:
                lines.append(
                    f" {r['band']:>5d}    {r['energy']:>10.6f}    "
                    f"{r['pvsk']:>7.3f}    {r['bcf']:>7.3f}    "
                    f"{r['pcbm']:>7.3f}    {r['dominant']}"
                )

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Diagnose fixed-band energy kinks / band switching from root trajectory directory."
    )
    ap.add_argument(
        "--frames",
        nargs="*",
        type=int,
        default=[],
        help="Suspicious center frame labels to inspect.",
    )
    ap.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Neighboring frames shown on each side of selected frames.",
    )
    ap.add_argument(
        "--auto",
        action="store_true",
        help="Automatically detect large one-frame energy jumps.",
    )
    ap.add_argument(
        "--jump-threshold",
        type=float,
        default=0.12,
        help="Automatic absolute one-frame jump threshold in eV.",
    )
    ap.add_argument(
        "--band-min",
        type=int,
        default=976,
        help="Lowest VASP band number to inspect.",
    )
    ap.add_argument(
        "--band-max",
        type=int,
        default=981,
        help="Highest VASP band number to inspect.",
    )
    ap.add_argument("--spin", type=int, default=0)
    ap.add_argument("--kpoint", type=int, default=0)
    ap.add_argument(
        "--csv",
        default="band_switching_diagnostic.csv",
        help="CSV output for inspected frame/band records.",
    )
    ap.add_argument(
        "--report",
        default="band_switching_diagnostic.txt",
        help="Human-readable diagnostic report.",
    )
    args = ap.parse_args()

    frame_dirs = detect_frame_dirs(".")
    labels_available = [frame_label(d) for d in frame_dirs]

    poscar = find_poscar(".", frame_dirs)
    symbols = read_poscar_symbols(poscar)
    groups = build_groups_from_rules(symbols)
    band_numbers = list(range(args.band_min, args.band_max + 1))

    selected_centers = list(args.frames)

    if args.auto:
        labels, energies = load_all_energies(
            frame_dirs,
            band_numbers,
            spin=args.spin,
            kpoint=args.kpoint,
        )
        jumps = detect_jumps(
            labels, energies, band_numbers, args.jump_threshold
        )

        print(f"Detected {len(jumps)} band jumps >= {args.jump_threshold:.3f} eV.")

        # Rank by largest jump and use unique destination frames as centers.
        ranked = sorted(jumps, key=lambda x: x["abs_jump"], reverse=True)
        auto_centers = []
        for ev in ranked:
            if ev["frame"] not in auto_centers:
                auto_centers.append(ev["frame"])

        selected_centers.extend(auto_centers)

        # Write a quick auto-jump summary.
        with open("auto_jump_events.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame_prev", "frame", "band",
                    "abs_jump", "e_prev", "e_curr"
                ],
            )
            writer.writeheader()
            writer.writerows(ranked)

    selected_centers = sorted(set(selected_centers))

    if not selected_centers:
        raise SystemExit(
            "No frames selected. Use --frames FRAME ... and/or --auto."
        )

    wanted = set()
    for center in selected_centers:
        for fr in range(center - args.radius, center + args.radius + 1):
            wanted.add(fr)

    records = load_selected_frames(
        frame_dirs,
        wanted,
        groups,
        band_numbers,
        spin=args.spin,
        kpoint=args.kpoint,
    )

    report_text = print_frame_block(
        records,
        selected_centers,
        args.radius,
        band_numbers,
    )

    print(report_text)

    with open(args.report, "w") as f:
        f.write(report_text)
        f.write("\n")

    with open(args.csv, "w", newline="") as f:
        fieldnames = [
            "frame", "band", "energy",
            "pvsk", "bcf", "pcbm", "dominant"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fr in sorted(records):
            writer.writerows(records[fr])

    print("\nWrote:")
    print(" ", args.report)
    print(" ", args.csv)
    if args.auto:
        print("  auto_jump_events.csv")


if __name__ == "__main__":
    main()