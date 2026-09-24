#!/usr/bin/env python3
"""
plot_B_avoided_crossings_clean.py

Cleaner publication-style plotter for the B1-B4 avoided-crossing windows in
trajectory B.

Compared with the earlier verbose generator, this version:
  * removes the long diagnostic subtitles,
  * removes per-panel text clutter,
  * keeps only a compact panel title (B1-B4),
  * optionally removes faint context bands,
  * keeps shaded event windows,
  * colors highlighted bands by instantaneous fragment character
    (PVSK / BCF / PCBM),
  * optionally writes a separate plain-text summary of minimum gaps.

Run from the trajectory-B root directory containing numbered daughter folders:

    python plot_B_avoided_crossings_clean.py

Useful options:
    python plot_B_avoided_crossings_clean.py --padding 20
    python plot_B_avoided_crossings_clean.py --no-context
    python plot_B_avoided_crossings_clean.py --summary B_clean_summary.txt
"""

import os
import re
import argparse
from glob import glob
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib as mpl
mpl.use("agg")
mpl.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


EVENTS = {
    "B1": {"start": 1488, "end": 1492, "bands": [978, 979]},
    "B2": {"start": 1687, "end": 1696, "bands": [978, 979, 980]},
    "B3": {"start": 1801, "end": 1811, "bands": [978, 980]},
    "B4": {"start": 1846, "end": 1855, "bands": [978, 979]},
}

COL_PVSK = np.array([0.20, 0.35, 0.90])
COL_BCF  = np.array([0.10, 0.70, 0.30])
COL_PCBM = np.array([0.95, 0.55, 0.10])


def WeightFromPro(infile="PROCAR", whichAtom=None, spd=None):
    if not os.path.isfile(infile):
        raise FileNotFoundError(infile)

    with open(infile, "r") as f:
        FileContents = [line for line in f if line.strip()]

    nkpts, nbands, nions = [
        int(xx) for xx in re.sub(r"[^0-9]", " ", FileContents[1]).split()
    ]

    if spd:
        Weights = np.asarray(
            [line.split()[1:-1] for line in FileContents if not re.search(r"[a-zA-Z]", line)],
            dtype=float,
        )
        Weights = np.sum(Weights[:, spd], axis=1)
    else:
        Weights = np.asarray(
            [line.split()[-1] for line in FileContents if not re.search(r"[a-zA-Z]", line)],
            dtype=float,
        )

    nspin = Weights.shape[0] // (nkpts * nbands * nions)
    Weights.resize(nspin, nkpts, nbands, nions)

    Energies = np.asarray(
        [line.split()[-4] for line in FileContents if "occ." in line],
        dtype=float,
    )
    Energies.resize(nspin, nkpts, nbands)

    if whichAtom is None:
        return Energies, np.sum(Weights, axis=-1)
    return Energies, np.sum(Weights[:, :, :, whichAtom], axis=-1)


def parallel_wht(runDirs, whichAtoms, nproc=None):
    nproc = cpu_count() if nproc is None else min(nproc, cpu_count())
    with Pool(processes=nproc) as pool:
        results = [
            pool.apply_async(
                WeightFromPro,
                (os.path.join(rd, "PROCAR"), whichAtoms, None)
            )
            for rd in runDirs
        ]
        enr, wht = [], []
        for r in results:
            e, w = r.get()
            enr.append(e)
            wht.append(w)
    return np.asarray(enr), np.asarray(wht)


def read_poscar_symbols(poscar="POSCAR"):
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

    expanded = []
    for sym, n in zip(symbols, counts):
        expanded.extend([sym] * n)
    return np.asarray(expanded)


def build_groups_from_rules(symbols):
    n = len(symbols)
    idx_all = np.arange(n, dtype=int)
    c_idx = idx_all[symbols == "C"]

    if len(c_idx) < 45:
        raise ValueError(f"Expected at least 45 carbon atoms; found {len(c_idx)}.")

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


def detect_frame_dirs(prefix="."):
    dirs = sorted(
        d for d in glob(os.path.join(prefix, "[0-9][0-9][0-9][0-9]"))
        if os.path.isdir(d)
    )
    dirs = [d for d in dirs if os.path.isfile(os.path.join(d, "PROCAR"))]
    if not dirs:
        raise FileNotFoundError("No daughter frame directories with PROCAR were found.")
    return dirs


def auto_detect_poscar(prefix=".", runDirs=None):
    root_poscar = os.path.join(prefix, "POSCAR")
    if os.path.isfile(root_poscar):
        return root_poscar
    for d in runDirs or []:
        cand = os.path.join(d, "POSCAR")
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError("Could not find POSCAR in the root or frame directories.")


def load_or_compute(prefix, cache_npz, whichS, whichK, nproc):
    runDirs = detect_frame_dirs(prefix)
    nsw = len(runDirs)

    if os.path.isfile(cache_npz):
        data = np.load(cache_npz)
        Enr = data["Enr"]
        W_pvsk = data["W_pvsk"]
        W_bcf = data["W_bcf"]
        W_pcbm = data["W_pcbm"]
        if Enr.shape[0] != nsw:
            raise RuntimeError(
                f"Cache has {Enr.shape[0]} frames but {nsw} frame directories were found."
            )
        print(f"Loaded cache: {cache_npz}")
        return Enr, W_pvsk, W_bcf, W_pcbm

    poscar_path = auto_detect_poscar(prefix, runDirs)
    print("No cache found; recomputing from PROCAR.")
    print("Using POSCAR:", poscar_path)

    symbols = read_poscar_symbols(poscar_path)
    pvsk_idx, bcf_idx, pcbm_idx = build_groups_from_rules(symbols)

    Enr, W_pvsk = parallel_wht(runDirs, pvsk_idx, nproc=nproc)
    _, W_bcf = parallel_wht(runDirs, bcf_idx, nproc=nproc)
    _, W_pcbm = parallel_wht(runDirs, pcbm_idx, nproc=nproc)

    Enr = Enr[:, whichS, whichK, :]
    W_pvsk = W_pvsk[:, whichS, whichK, :]
    W_bcf = W_bcf[:, whichS, whichK, :]
    W_pcbm = W_pcbm[:, whichS, whichK, :]

    np.savez_compressed(
        cache_npz,
        Enr=Enr,
        W_pvsk=W_pvsk,
        W_bcf=W_bcf,
        W_pcbm=W_pcbm,
    )
    print("Saved cache:", cache_npz)
    return Enr, W_pvsk, W_bcf, W_pcbm


def character_rgb(F_pvsk, F_bcf, F_pcbm):
    return (
        F_pvsk[..., None] * COL_PVSK
        + F_bcf[..., None] * COL_BCF
        + F_pcbm[..., None] * COL_PCBM
    )


def local_ylim(Enr, indices, lo, hi, pad=0.08):
    block = Enr[lo:hi+1, :][:, indices]
    ymin = float(np.nanmin(block))
    ymax = float(np.nanmax(block))
    span = max(ymax - ymin, 0.04)
    return ymin - pad * span, ymax + pad * span


def compute_gap_summary(Enr, event):
    target_idx = [b - 1 for b in event["bands"]]
    ev_lo, ev_hi = event["start"], event["end"]
    out = []
    for i in range(len(target_idx)):
        for j in range(i + 1, len(target_idx)):
            gap = np.abs(
                Enr[ev_lo:ev_hi+1, target_idx[i]]
                - Enr[ev_lo:ev_hi+1, target_idx[j]]
            )
            if gap.size:
                k = int(np.argmin(gap))
                out.append({
                    "pair": (event["bands"][i], event["bands"][j]),
                    "min_gap_meV": 1000.0 * float(gap[k]),
                    "frame": ev_lo + k,
                })
    return out


def plot_event(
    ax,
    event_name,
    event,
    Enr,
    RGB,
    padding,
    band_min,
    band_max,
    dot_size,
    show_context,
    annotate_band_labels,
):
    nframes, nbands = Enr.shape
    context_numbers = [
        b for b in range(band_min, band_max + 1)
        if 1 <= b <= nbands
    ]
    target_numbers = [b for b in event["bands"] if 1 <= b <= nbands]

    context_idx = [b - 1 for b in context_numbers]
    target_idx = [b - 1 for b in target_numbers]

    lo = max(0, event["start"] - padding)
    hi = min(nframes - 1, event["end"] + padding)
    frames = np.arange(lo, hi + 1)

    if show_context:
        for ib in context_idx:
            ax.plot(
                frames,
                Enr[lo:hi+1, ib],
                color="0.83",
                lw=0.7,
                alpha=0.7,
                zorder=1,
            )

    for bnum, ib in zip(target_numbers, target_idx):
        y = Enr[lo:hi+1, ib]
        c = RGB[lo:hi+1, ib, :]

        ax.plot(
            frames,
            y,
            color="0.22",
            lw=0.9,
            alpha=0.75,
            zorder=2,
        )
        ax.scatter(
            frames,
            y,
            c=c,
            s=dot_size,
            edgecolors="none",
            alpha=0.95,
            zorder=3,
        )

        if annotate_band_labels:
            j = min(3, len(frames) - 1)
            ax.text(
                frames[j],
                y[j],
                f"{bnum}",
                fontsize=8,
                va="bottom",
                ha="left",
                color="0.15",
                zorder=4,
            )

    ax.axvspan(event["start"], event["end"], color="0.88", alpha=0.7, zorder=0)
    ax.axvline(event["start"], color="0.55", lw=0.6, ls="--", alpha=0.8)
    ax.axvline(event["end"], color="0.55", lw=0.6, ls="--", alpha=0.8)

    ylo, yhi = local_ylim(Enr, target_idx, lo, hi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(lo, hi)

    ax.set_title(event_name, fontsize=11, pad=6)
    ax.set_xlabel("MD frame")
    ax.set_ylabel("KS energy [eV]")
    ax.grid(ls="--", lw=0.35, alpha=0.30)


def main():
    ap = argparse.ArgumentParser(description="Cleaner avoided-crossing figure generator for trajectory B.")
    ap.add_argument("--prefix", default=".")
    ap.add_argument("--cache", default="pvsk_bcf_pcbm_weights.npz")
    ap.add_argument("--spin", type=int, default=0)
    ap.add_argument("--kpoint", type=int, default=0)
    ap.add_argument("--nproc", type=int, default=24)
    ap.add_argument("--padding", type=int, default=20)
    ap.add_argument("--band-min", type=int, default=976)
    ap.add_argument("--band-max", type=int, default=981)
    ap.add_argument("--dot-size", type=float, default=18.0)
    ap.add_argument("--no-context", action="store_true", help="Do not draw faint context bands.")
    ap.add_argument("--no-band-labels", action="store_true", help="Do not label highlighted band numbers.")
    ap.add_argument("--out", default="B_avoided_crossings_clean")
    ap.add_argument("--summary", default="B_avoided_crossings_clean_summary.txt")
    args = ap.parse_args()

    Enr, W_pvsk, W_bcf, W_pcbm = load_or_compute(
        args.prefix,
        args.cache,
        args.spin,
        args.kpoint,
        args.nproc,
    )

    W_tot = W_pvsk + W_bcf + W_pcbm + 1e-15
    F_pvsk = W_pvsk / W_tot
    F_bcf  = W_bcf / W_tot
    F_pcbm = W_pcbm / W_tot
    RGB = character_rgb(F_pvsk, F_bcf, F_pcbm)

    fig, axes = plt.subplots(
        2, 2, figsize=(10.5, 7.4), constrained_layout=True
    )
    axes = axes.ravel()

    summary_lines = []
    summary_lines.append("Trajectory B avoided-crossing summary")
    summary_lines.append("=" * 60)

    for ax, (name, event) in zip(axes, EVENTS.items()):
        plot_event(
            ax=ax,
            event_name=name,
            event=event,
            Enr=Enr,
            RGB=RGB,
            padding=args.padding,
            band_min=args.band_min,
            band_max=args.band_max,
            dot_size=args.dot_size,
            show_context=(not args.no_context),
            annotate_band_labels=(not args.no_band_labels),
        )

        gaps = compute_gap_summary(Enr, event)
        summary_lines.append(f"{name}  frames {event['start']}-{event['end']}")
        for g in gaps:
            b1, b2 = g["pair"]
            summary_lines.append(
                f"  bands {b1}/{b2}: min gap = {g['min_gap_meV']:.1f} meV at frame {g['frame']}"
            )

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=COL_PVSK, markeredgecolor="none",
               markersize=6.5, label="perovskite character"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=COL_BCF, markeredgecolor="none",
               markersize=6.5, label="BCF character"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=COL_PCBM, markeredgecolor="none",
               markersize=6.5, label="PCBM character"),
    ]
    if not args.no_context:
        legend_handles.append(
            Line2D([0], [0], color="0.83", lw=1.0, label="other bands")
        )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=9,
    )

    fig.suptitle(
        "Trajectory B: local BCF/PCBM character exchange",
        fontsize=13,
        y=1.05,
    )

    png = args.out + ".png"
    pdf = args.out + ".pdf"
    txt = args.summary

    fig.savefig(png, dpi=500, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    with open(txt, "w") as f:
        f.write("\n".join(summary_lines))
        f.write("\n")

    print("Wrote:", png)
    print("Wrote:", pdf)
    print("Wrote:", txt)
    print("")
    print("This clean version suppresses the long on-figure diagnostics.")
    print("Gap summaries were written to the text file instead.")


if __name__ == "__main__":
    main()