#!/usr/bin/env python3
"""
check_fragment_level_shifts.py

Diagnose whether abrupt one-frame KS-energy excursions are:
  * common-mode shifts of a fragment-localized manifold,
  * changes in relative fragment alignment,
  * or broad spectrum-wide shifts.

Run from the ROOT directory containing numbered daughter folders such as:
    0001/ 0002/ ... 1999/

Each daughter folder should contain PROCAR. OUTCAR is optional but, when
present, E-fermi and final TOTEN are extracted for additional diagnostics.

The analysis uses the same FAPbI3/BCF/PCBM atom partition convention used in
the associated trajectory analysis:
    first 27 C  -> PVSK
    next 18 C   -> BCF
    remaining C -> PCBM
    O           -> PCBM
    B,F         -> BCF
    everything else -> PVSK

Main outputs
------------
fragment_level_shifts.csv
    Per-frame energies, fragment-weighted centroids, internal PCBM spacings,
    relative alignments, E-fermi, and simple common-mode-shift metrics.

fragment_level_shift_events.csv
    Frames flagged as candidate PCBM common-mode excursions.

fragment_level_shifts.png
    Four-panel overview:
      1) raw KS energies of selected bands
      2) fragment centroids relative to PVSK reference
      3) PCBM internal spacings
      4) common-mode / rigidity diagnostic

Usage
-----
Basic:
    python check_fragment_level_shifts.py

Focus plot on a frame interval:
    python check_fragment_level_shifts.py --frame-min 1450 --frame-max 1530

Change band window:
    python check_fragment_level_shifts.py --band-min 976 --band-max 981

Change candidate threshold:
    python check_fragment_level_shifts.py --shift-threshold 0.12

Scientific interpretation
-------------------------
A candidate fragment-level electrostatic/common-mode excursion is a frame where:
  * the PCBM centroid changes strongly relative to the preceding frame,
  * multiple PCBM-like states move in the same direction,
  * the change in their internal spacings is much smaller than the centroid shift.

This is a diagnostic, not a proof of electrostatic origin. Confirmation with a
real-space electrostatic potential reference (e.g. carefully aligned LOCPOT)
would be a separate follow-up.
"""

import os
import re
import csv
import argparse
from glob import glob

import numpy as np

import matplotlib as mpl
mpl.use("agg")
mpl.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# File / structure helpers
# ----------------------------------------------------------------------

def detect_frame_dirs(root="."):
    dirs = sorted(
        d for d in glob(os.path.join(root, "[0-9][0-9][0-9][0-9]"))
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "PROCAR"))
    )
    if not dirs:
        raise FileNotFoundError(
            "No four-digit daughter directories containing PROCAR were found."
        )
    return dirs


def frame_label(path):
    return int(os.path.basename(os.path.normpath(path)))


def find_poscar(root, frame_dirs):
    p = os.path.join(root, "POSCAR")
    if os.path.isfile(p):
        return p
    for d in frame_dirs:
        p = os.path.join(d, "POSCAR")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("Could not find POSCAR in root or frame directories.")


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
            f"Expected at least 45 carbon atoms; found {len(c_idx)}."
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


# ----------------------------------------------------------------------
# PROCAR / OUTCAR parsing
# ----------------------------------------------------------------------

def parse_procar(procar, spin=0, kpoint=0):
    with open(procar, "r") as f:
        lines = [line for line in f if line.strip()]

    nkpts, nbands, nions = [
        int(xx) for xx in re.sub(r"[^0-9]", " ", lines[1]).split()
    ]

    energy_vals = np.asarray(
        [line.split()[-4] for line in lines if "occ." in line],
        dtype=float,
    )
    nspin = len(energy_vals) // (nkpts * nbands)
    energy_vals.resize(nspin, nkpts, nbands)
    energies = energy_vals[spin, kpoint, :].copy()

    weight_vals = np.asarray(
        [line.split()[-1]
         for line in lines
         if not re.search(r"[a-zA-Z]", line)],
        dtype=float,
    )
    nspin_w = len(weight_vals) // (nkpts * nbands * nions)
    if nspin_w != nspin:
        raise ValueError(
            f"Inconsistent PROCAR spin count in {procar}: "
            f"energy nspin={nspin}, weight nspin={nspin_w}"
        )

    weight_vals.resize(nspin, nkpts, nbands, nions)
    ion_weights = weight_vals[spin, kpoint, :, :].copy()

    return energies, ion_weights


def parse_outcar_scalars(outcar):
    """
    Return final E-fermi and final TOTEN found in OUTCAR.
    Missing values return NaN.
    """
    efermi = np.nan
    toten = np.nan

    if not os.path.isfile(outcar):
        return efermi, toten

    re_fermi = re.compile(r"E-fermi\s*:\s*([-+0-9Ee.]+)")
    re_toten = re.compile(r"free\s+energy\s+TOTEN\s*=\s*([-+0-9Ee.]+)")

    with open(outcar, "r", errors="replace") as f:
        for line in f:
            m = re_fermi.search(line)
            if m:
                try:
                    efermi = float(m.group(1))
                except ValueError:
                    pass
            m = re_toten.search(line)
            if m:
                try:
                    toten = float(m.group(1))
                except ValueError:
                    pass

    return efermi, toten


# ----------------------------------------------------------------------
# Analysis helpers
# ----------------------------------------------------------------------

def weighted_centroid(energies, weights, min_total_weight=1e-12):
    s = np.sum(weights)
    if s <= min_total_weight:
        return np.nan
    return float(np.sum(energies * weights) / s)


def weighted_spread(energies, weights, centroid=None):
    s = np.sum(weights)
    if s <= 1e-12:
        return np.nan
    if centroid is None or not np.isfinite(centroid):
        centroid = weighted_centroid(energies, weights)
    var = np.sum(weights * (energies - centroid) ** 2) / s
    return float(np.sqrt(max(var, 0.0)))


def dominant_fragment(fp, fb, fc):
    vals = {"PVSK": fp, "BCF": fb, "PCBM": fc}
    return max(vals, key=vals.get)


def same_sign_fraction(deltas, eps=1e-10):
    finite = np.asarray([x for x in deltas if np.isfinite(x)])
    if finite.size == 0:
        return np.nan
    finite = finite[np.abs(finite) > eps]
    if finite.size == 0:
        return 0.0
    npos = np.count_nonzero(finite > 0)
    nneg = np.count_nonzero(finite < 0)
    return max(npos, nneg) / finite.size


def parse_args():
    ap = argparse.ArgumentParser(
        description="Diagnose common-mode fragment-level KS energy excursions."
    )
    ap.add_argument("--band-min", type=int, default=976)
    ap.add_argument("--band-max", type=int, default=981)
    ap.add_argument("--spin", type=int, default=0)
    ap.add_argument("--kpoint", type=int, default=0)

    ap.add_argument(
        "--pvsk-reference-band",
        type=int,
        default=981,
        help=(
            "Fixed VASP band used as a simple PVSK energy reference. "
            "Default 981 based on the current six-state window. "
            "A character-weighted PVSK centroid is also reported independently."
        ),
    )

    ap.add_argument(
        "--pcbm-character-min",
        type=float,
        default=0.80,
        help="Minimum PCBM character for a band to enter the rigid-manifold test.",
    )
    ap.add_argument(
        "--shift-threshold",
        type=float,
        default=0.12,
        help="Minimum one-frame PCBM centroid shift (eV) for candidate events.",
    )
    ap.add_argument(
        "--rigidity-ratio-max",
        type=float,
        default=0.40,
        help=(
            "Candidate requires change in PCBM internal spread / centroid shift "
            "to be below this ratio."
        ),
    )
    ap.add_argument(
        "--same-sign-min",
        type=float,
        default=0.75,
        help="Minimum fraction of PCBM-like bands moving in the same direction.",
    )

    ap.add_argument("--frame-min", type=int, default=None)
    ap.add_argument("--frame-max", type=int, default=None)

    ap.add_argument("--csv", default="fragment_level_shifts.csv")
    ap.add_argument("--events", default="fragment_level_shift_events.csv")
    ap.add_argument("--png", default="fragment_level_shifts.png")

    return ap.parse_args()


def main():
    args = parse_args()

    frame_dirs = detect_frame_dirs(".")
    poscar = find_poscar(".", frame_dirs)
    symbols = read_poscar_symbols(poscar)
    pvsk_idx, bcf_idx, pcbm_idx = build_groups_from_rules(symbols)

    band_numbers = list(range(args.band_min, args.band_max + 1))
    rows = []

    for d in frame_dirs:
        fr = frame_label(d)

        if args.frame_min is not None and fr < args.frame_min:
            continue
        if args.frame_max is not None and fr > args.frame_max:
            continue

        energies_all, ion_weights = parse_procar(
            os.path.join(d, "PROCAR"),
            spin=args.spin,
            kpoint=args.kpoint,
        )

        Wp = ion_weights[:, pvsk_idx].sum(axis=1)
        Wb = ion_weights[:, bcf_idx].sum(axis=1)
        Wc = ion_weights[:, pcbm_idx].sum(axis=1)
        Wtot = Wp + Wb + Wc + 1e-15

        Fp = Wp / Wtot
        Fb = Wb / Wtot
        Fc = Wc / Wtot

        idx = [b - 1 for b in band_numbers if 1 <= b <= len(energies_all)]
        bnums_valid = [b for b in band_numbers if 1 <= b <= len(energies_all)]

        E = energies_all[idx]
        fp = Fp[idx]
        fb = Fb[idx]
        fc = Fc[idx]

        # Fragment-weighted centroids within selected band window
        pvsk_centroid = weighted_centroid(E, fp)
        bcf_centroid = weighted_centroid(E, fb)
        pcbm_centroid = weighted_centroid(E, fc)

        pvsk_spread = weighted_spread(E, fp, pvsk_centroid)
        bcf_spread = weighted_spread(E, fb, bcf_centroid)
        pcbm_spread = weighted_spread(E, fc, pcbm_centroid)

        # Strongly PCBM-localized subset for rigid-manifold tests
        pcbm_mask = fc >= args.pcbm_character_min
        pcbm_local_energies = E[pcbm_mask]
        pcbm_local_bands = np.asarray(bnums_valid, dtype=int)[pcbm_mask]

        if pcbm_local_energies.size >= 2:
            pcbm_local_mean = float(np.mean(pcbm_local_energies))
            pcbm_local_std = float(np.std(pcbm_local_energies))
            pcbm_local_range = float(
                np.max(pcbm_local_energies) - np.min(pcbm_local_energies)
            )
        elif pcbm_local_energies.size == 1:
            pcbm_local_mean = float(pcbm_local_energies[0])
            pcbm_local_std = 0.0
            pcbm_local_range = 0.0
        else:
            pcbm_local_mean = np.nan
            pcbm_local_std = np.nan
            pcbm_local_range = np.nan

        # Simple fixed PVSK reference band plus character-weighted PVSK centroid
        ref_ib = args.pvsk_reference_band - 1
        if 0 <= ref_ib < len(energies_all):
            pvsk_ref_fixed = float(energies_all[ref_ib])
            pvsk_ref_fixed_char = float(Fp[ref_ib])
        else:
            pvsk_ref_fixed = np.nan
            pvsk_ref_fixed_char = np.nan

        efermi, toten = parse_outcar_scalars(os.path.join(d, "OUTCAR"))

        row = {
            "frame": fr,
            "efermi": efermi,
            "toten": toten,
            "pvsk_centroid": pvsk_centroid,
            "bcf_centroid": bcf_centroid,
            "pcbm_centroid": pcbm_centroid,
            "pvsk_spread": pvsk_spread,
            "bcf_spread": bcf_spread,
            "pcbm_spread": pcbm_spread,
            "pcbm_local_mean": pcbm_local_mean,
            "pcbm_local_std": pcbm_local_std,
            "pcbm_local_range": pcbm_local_range,
            "pcbm_local_count": int(pcbm_local_energies.size),
            "pcbm_local_bands": ";".join(map(str, pcbm_local_bands.tolist())),
            "pvsk_ref_fixed": pvsk_ref_fixed,
            "pvsk_ref_fixed_char": pvsk_ref_fixed_char,
            "pcbm_minus_pvsk_centroid": (
                pcbm_centroid - pvsk_centroid
                if np.isfinite(pcbm_centroid) and np.isfinite(pvsk_centroid)
                else np.nan
            ),
            "bcf_minus_pvsk_centroid": (
                bcf_centroid - pvsk_centroid
                if np.isfinite(bcf_centroid) and np.isfinite(pvsk_centroid)
                else np.nan
            ),
            "pcbm_local_minus_pvsk_fixed": (
                pcbm_local_mean - pvsk_ref_fixed
                if np.isfinite(pcbm_local_mean) and np.isfinite(pvsk_ref_fixed)
                else np.nan
            ),
        }

        # Store per-band quantities
        for b, e, a, bb, c in zip(bnums_valid, E, fp, fb, fc):
            row[f"E_{b}"] = float(e)
            row[f"PVSK_{b}"] = float(a)
            row[f"BCF_{b}"] = float(bb)
            row[f"PCBM_{b}"] = float(c)
            row[f"DOM_{b}"] = dominant_fragment(a, bb, c)

        rows.append(row)

    rows.sort(key=lambda r: r["frame"])

    # Consecutive-frame differences + candidate classification
    events = []
    previous = None

    for row in rows:
        row["d_pcbm_local_mean"] = np.nan
        row["d_pcbm_local_std"] = np.nan
        row["d_pcbm_centroid"] = np.nan
        row["d_pcbm_minus_pvsk_centroid"] = np.nan
        row["same_sign_fraction"] = np.nan
        row["rigidity_ratio"] = np.nan
        row["candidate_common_mode"] = False

        if previous is not None and row["frame"] == previous["frame"] + 1:
            for key in [
                "pcbm_local_mean",
                "pcbm_local_std",
                "pcbm_centroid",
                "pcbm_minus_pvsk_centroid",
            ]:
                a = row.get(key, np.nan)
                b = previous.get(key, np.nan)
                row["d_" + key] = (
                    a - b if np.isfinite(a) and np.isfinite(b) else np.nan
                )

            # Compare shifts of bands that are strongly PCBM-like in BOTH frames
            deltas = []
            shared_bands = []
            for b in band_numbers:
                c_now = row.get(f"PCBM_{b}", np.nan)
                c_prev = previous.get(f"PCBM_{b}", np.nan)
                e_now = row.get(f"E_{b}", np.nan)
                e_prev = previous.get(f"E_{b}", np.nan)

                if (
                    np.isfinite(c_now) and np.isfinite(c_prev)
                    and c_now >= args.pcbm_character_min
                    and c_prev >= args.pcbm_character_min
                    and np.isfinite(e_now) and np.isfinite(e_prev)
                ):
                    deltas.append(e_now - e_prev)
                    shared_bands.append(b)

            ssf = same_sign_fraction(deltas)
            row["same_sign_fraction"] = ssf

            shift = row["d_pcbm_local_mean"]
            dspread = row["d_pcbm_local_std"]

            if np.isfinite(shift) and abs(shift) > 1e-12 and np.isfinite(dspread):
                rr = abs(dspread) / abs(shift)
            else:
                rr = np.nan

            row["rigidity_ratio"] = rr

            candidate = (
                np.isfinite(shift)
                and abs(shift) >= args.shift_threshold
                and np.isfinite(ssf)
                and ssf >= args.same_sign_min
                and np.isfinite(rr)
                and rr <= args.rigidity_ratio_max
                and len(shared_bands) >= 2
            )

            row["candidate_common_mode"] = bool(candidate)

            if candidate:
                events.append({
                    "frame_prev": previous["frame"],
                    "frame": row["frame"],
                    "pcbm_local_mean_prev": previous["pcbm_local_mean"],
                    "pcbm_local_mean": row["pcbm_local_mean"],
                    "d_pcbm_local_mean": shift,
                    "pcbm_local_std_prev": previous["pcbm_local_std"],
                    "pcbm_local_std": row["pcbm_local_std"],
                    "d_pcbm_local_std": dspread,
                    "same_sign_fraction": ssf,
                    "rigidity_ratio": rr,
                    "shared_pcbm_bands": ";".join(map(str, shared_bands)),
                    "d_pcbm_minus_pvsk_centroid": row[
                        "d_pcbm_minus_pvsk_centroid"
                    ],
                })

        previous = row

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------

    if not rows:
        raise RuntimeError("No frames remained after filtering.")

    fieldnames = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    event_fields = [
        "frame_prev",
        "frame",
        "pcbm_local_mean_prev",
        "pcbm_local_mean",
        "d_pcbm_local_mean",
        "pcbm_local_std_prev",
        "pcbm_local_std",
        "d_pcbm_local_std",
        "same_sign_fraction",
        "rigidity_ratio",
        "shared_pcbm_bands",
        "d_pcbm_minus_pvsk_centroid",
    ]

    with open(args.events, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=event_fields)
        writer.writeheader()
        writer.writerows(events)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    frames = np.asarray([r["frame"] for r in rows], dtype=int)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11.5, 10.5),
        sharex=True,
        constrained_layout=True,
    )

    # 1. Raw KS energies
    ax = axes[0]
    for b in band_numbers:
        vals = np.asarray([r.get(f"E_{b}", np.nan) for r in rows], dtype=float)
        ax.plot(frames, vals, lw=0.8, label=f"band {b}")
    ax.set_ylabel("KS energy [eV]")
    ax.set_title("Selected raw KS eigenvalues")
    ax.grid(ls="--", lw=0.4, alpha=0.35)
    ax.legend(ncol=3, fontsize=7, frameon=False)

    # 2. Fragment alignments relative to PVSK centroid
    ax = axes[1]
    pcbm_rel = np.asarray(
        [r["pcbm_minus_pvsk_centroid"] for r in rows], dtype=float
    )
    bcf_rel = np.asarray(
        [r["bcf_minus_pvsk_centroid"] for r in rows], dtype=float
    )
    pcbm_fixed_rel = np.asarray(
        [r["pcbm_local_minus_pvsk_fixed"] for r in rows], dtype=float
    )

    ax.plot(frames, pcbm_rel, lw=1.0, label="PCBM centroid - PVSK centroid")
    ax.plot(frames, bcf_rel, lw=1.0, label="BCF centroid - PVSK centroid")
    ax.plot(
        frames,
        pcbm_fixed_rel,
        lw=0.8,
        alpha=0.75,
        label="PCBM-local mean - fixed PVSK reference",
    )
    ax.set_ylabel("Relative energy [eV]")
    ax.set_title("Fragment-level alignment")
    ax.grid(ls="--", lw=0.4, alpha=0.35)
    ax.legend(fontsize=8, frameon=False)

    # 3. PCBM internal spread
    ax = axes[2]
    pcbm_std = np.asarray([r["pcbm_local_std"] for r in rows], dtype=float)
    pcbm_range = np.asarray([r["pcbm_local_range"] for r in rows], dtype=float)
    ax.plot(frames, pcbm_std, lw=1.0, label="PCBM-local std")
    ax.plot(frames, pcbm_range, lw=0.9, label="PCBM-local range")
    ax.set_ylabel("Internal spread [eV]")
    ax.set_title("Internal PCBM manifold spacing")
    ax.grid(ls="--", lw=0.4, alpha=0.35)
    ax.legend(fontsize=8, frameon=False)

    # 4. Common-mode diagnostic
    ax = axes[3]
    dmean = np.asarray(
        [r["d_pcbm_local_mean"] for r in rows], dtype=float
    )
    dstd = np.asarray(
        [r["d_pcbm_local_std"] for r in rows], dtype=float
    )

    ax.plot(frames, dmean, lw=0.9, label="Δ PCBM-local mean")
    ax.plot(frames, dstd, lw=0.9, label="Δ PCBM-local std")

    if events:
        ev_frames = np.asarray([e["frame"] for e in events], dtype=int)
        ev_vals = np.asarray([e["d_pcbm_local_mean"] for e in events], dtype=float)
        ax.scatter(
            ev_frames,
            ev_vals,
            s=22,
            marker="o",
            label="candidate common-mode excursion",
            zorder=4,
        )

    ax.axhline(args.shift_threshold, lw=0.6, ls="--", alpha=0.5)
    ax.axhline(-args.shift_threshold, lw=0.6, ls="--", alpha=0.5)

    ax.set_ylabel("One-frame change [eV]")
    ax.set_xlabel("MD frame")
    ax.set_title("Common-mode vs internal-spacing change")
    ax.grid(ls="--", lw=0.4, alpha=0.35)
    ax.legend(fontsize=8, frameon=False)

    fig.savefig(args.png, dpi=400, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------------------

    print(f"Analyzed {len(rows)} frames.")
    print(f"Wrote: {args.csv}")
    print(f"Wrote: {args.events}")
    print(f"Wrote: {args.png}")
    print()

    if events:
        print("Candidate PCBM common-mode excursions:")
        print(
            " frame_prev -> frame    Δmean/eV    Δstd/eV    "
            "same-sign    rigidity    shared PCBM bands"
        )
        for e in events:
            print(
                f" {e['frame_prev']:>4d} -> {e['frame']:<4d}    "
                f"{e['d_pcbm_local_mean']:>+9.4f}    "
                f"{e['d_pcbm_local_std']:>+8.4f}    "
                f"{e['same_sign_fraction']:>8.3f}    "
                f"{e['rigidity_ratio']:>8.3f}    "
                f"{e['shared_pcbm_bands']}"
            )
    else:
        print(
            "No candidate common-mode excursions met the current thresholds. "
            "Try --shift-threshold 0.08 or inspect the full CSV."
        )


if __name__ == "__main__":
    main()