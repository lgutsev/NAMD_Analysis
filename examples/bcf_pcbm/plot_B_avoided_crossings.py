#!/usr/bin/env python3
"""
plot_B_avoided_crossings.py

B-specific generator for publication-quality zooms of the four
BCF/PCBM avoided-crossing / character-exchange regions in trajectory B.

Default event windows:
    B1: 1488-1492
    B2: 1687-1696
    B3: 1801-1811
    B4: 1846-1855

Default band mapping assumes the six-state NAMD manifold corresponds to
bands 976-981:
    B1: 978/979
    B2: 978/979/980
    B3: 978/980
    B4: 978/979

The script:
  * loads pvsk_bcf_pcbm_weights.npz if present;
  * otherwise recomputes the fragment weights from PROCAR files;
  * uses RAW KS energies by default (no smoothing), because smoothing can
    shift or create the appearance of a local avoided crossing;
  * plots faint context bands plus highlighted BCF/PCBM-relevant bands;
  * colors highlighted points by instantaneous PVSK/BCF/PCBM character;
  * shades the exact event window and adds a shared fragment-color legend;
  * writes both PNG and PDF.

Run from the trajectory-B directory:
    python plot_B_avoided_crossings.py

Typical options:
    python plot_B_avoided_crossings.py --padding 20
    python plot_B_avoided_crossings.py --band-min 976 --band-max 981
    python plot_B_avoided_crossings.py --cache pvsk_bcf_pcbm_weights.npz

Scientific plotting note:
The default plot intentionally does NOT smooth the energy traces.
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


# ---------------------------
# Defaults for trajectory B
# ---------------------------

EVENTS = {
    "B1": {"start": 1488, "end": 1492, "bands": [978, 979]},
    "B2": {"start": 1687, "end": 1696, "bands": [978, 979, 980]},
    "B3": {"start": 1801, "end": 1811, "bands": [978, 980]},
    "B4": {"start": 1846, "end": 1855, "bands": [978, 979]},
}

COL_PVSK = np.array([0.20, 0.35, 0.90])
COL_BCF  = np.array([0.10, 0.70, 0.30])
COL_PCBM = np.array([0.95, 0.55, 0.10])


# ---------------------------
# PROCAR parsing
# ---------------------------

def WeightFromPro(infile="PROCAR", whichAtom=None, spd=None):
    """Return energies and selected atom-projected PROCAR weights."""
    if not os.path.isfile(infile):
        raise FileNotFoundError(infile)

    with open(infile, "r") as f:
        FileContents = [line for line in f if line.strip()]

    nkpts, nbands, nions = [
        int(xx)
        for xx in re.sub(r"[^0-9]", " ", FileContents[1]).split()
    ]

    if spd:
        Weights = np.asarray(
            [line.split()[1:-1]
             for line in FileContents
             if not re.search(r"[a-zA-Z]", line)],
            dtype=float,
        )
        Weights = np.sum(Weights[:, spd], axis=1)
    else:
        Weights = np.asarray(
            [line.split()[-1]
             for line in FileContents
             if not re.search(r"[a-zA-Z]", line)],
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


# ---------------------------
# POSCAR/group assignment
# ---------------------------

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
        coord_line = 6
    else:
        symbols = toks5
        counts = [int(x) for x in lines[6].split()]
        coord_line = 7

    if len(symbols) != len(counts):
        raise ValueError(
            f"POSCAR parse mismatch: symbols={symbols}, counts={counts}"
        )

    expanded = []
    for sym, n in zip(symbols, counts):
        expanded.extend([sym] * n)

    coord_tag = lines[coord_line].lower()
    if coord_tag.startswith("s"):
        coord_tag = lines[coord_line + 1].lower()
    if not (coord_tag.startswith("d") or coord_tag.startswith("c")):
        print(
            "WARNING: could not confirm Direct/Cartesian line; "
            f"saw '{lines[coord_line]}'"
        )

    return np.asarray(expanded)


def build_groups_from_rules(symbols):
    """
    Same fragment assignment rules as tdksen_bcf_pcbm_smooth_nolegend.py:
      first 27 C -> PVSK
      next 18 C -> BCF
      remaining C -> PCBM
      O -> PCBM
      B/F -> BCF
      all other atoms -> PVSK
    """
    n = len(symbols)
    idx_all = np.arange(n, dtype=int)

    c_idx = idx_all[symbols == "C"]
    if len(c_idx) < 45:
        raise ValueError(
            f"Not enough carbon atoms for the configured grouping; found {len(c_idx)}."
        )

    pvsk_c = c_idx[:27]
    bcf_c = c_idx[27:45]
    pcbm_c = c_idx[45:]

    pvsk_idx = set(pvsk_c.tolist())
    bcf_idx = set(bcf_c.tolist())
    pcbm_idx = set(pcbm_c.tolist())

    bcf_idx.update(idx_all[np.isin(symbols, ["B", "F"])].tolist())
    pcbm_idx.update(idx_all[symbols == "O"].tolist())

    assigned = pvsk_idx | bcf_idx | pcbm_idx
    pvsk_idx.update(i for i in idx_all.tolist() if i not in assigned)

    pvsk_idx = np.asarray(sorted(pvsk_idx), dtype=int)
    bcf_idx = np.asarray(sorted(bcf_idx), dtype=int)
    pcbm_idx = np.asarray(sorted(pcbm_idx), dtype=int)

    assert len(set(pvsk_idx) & set(bcf_idx)) == 0
    assert len(set(pvsk_idx) & set(pcbm_idx)) == 0
    assert len(set(bcf_idx) & set(pcbm_idx)) == 0
    assert len(pvsk_idx) + len(bcf_idx) + len(pcbm_idx) == n

    return pvsk_idx, bcf_idx, pcbm_idx


# ---------------------------
# File discovery/cache
# ---------------------------

def detect_frame_dirs(prefix="."):
    dirs = sorted(
        d for d in glob(os.path.join(prefix, "[0-9][0-9][0-9][0-9]"))
        if os.path.isdir(d)
    )
    dirs = [
        d for d in dirs
        if os.path.isfile(os.path.join(d, "PROCAR"))
    ]
    if not dirs:
        raise FileNotFoundError(
            "No 4-digit frame directories containing PROCAR were found."
        )
    return dirs


def auto_detect_poscar(prefix=".", runDirs=None):
    root_poscar = os.path.join(prefix, "POSCAR")
    if os.path.isfile(root_poscar):
        return root_poscar

    for d in runDirs or []:
        cand = os.path.join(d, "POSCAR")
        if os.path.isfile(cand):
            return cand

    raise FileNotFoundError(
        "Could not find POSCAR in the root or frame directories."
    )


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
                f"Cache has {Enr.shape[0]} frames but {nsw} PROCAR frame "
                "directories were found. Delete/rebuild the cache."
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


# ---------------------------
# Plotting
# ---------------------------

def local_ylim(Enr, indices, lo, hi, pad=0.06):
    block = Enr[lo:hi+1, :][:, indices]
    ymin = float(np.nanmin(block))
    ymax = float(np.nanmax(block))
    span = max(ymax - ymin, 0.04)
    return ymin - pad * span, ymax + pad * span


def character_rgb(F_pvsk, F_bcf, F_pcbm):
    return (
        F_pvsk[..., None] * COL_PVSK
        + F_bcf[..., None] * COL_BCF
        + F_pcbm[..., None] * COL_PCBM
    )


def plot_event(
    ax,
    event_name,
    event,
    Enr,
    RGB,
    F_bcf,
    F_pcbm,
    padding,
    band_min,
    band_max,
    dot_size,
):
    nframes, nbands = Enr.shape

    # Convert 1-based VASP band labels to 0-based array indices.
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

    # Faint context bands.
    for ib in context_idx:
        ax.plot(
            frames,
            Enr[lo:hi+1, ib],
            color="0.72",
            lw=0.7,
            alpha=0.65,
            zorder=1,
        )

    # Highlight target bands using instantaneous fragment character.
    for bnum, ib in zip(target_numbers, target_idx):
        y = Enr[lo:hi+1, ib]
        c = RGB[lo:hi+1, ib, :]

        ax.plot(
            frames,
            y,
            color="0.18",
            lw=0.8,
            alpha=0.65,
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

        j = min(2, len(frames) - 1)
        ax.text(
            frames[j],
            y[j],
            f"{bnum}",
            fontsize=8,
            va="bottom",
            ha="left",
            color="0.15",
            zorder=5,
        )

    # Exact event interval.
    ax.axvspan(
        event["start"],
        event["end"],
        color="0.85",
        alpha=0.45,
        zorder=0,
    )
    ax.axvline(event["start"], color="0.45", lw=0.6, ls="--", alpha=0.7)
    ax.axvline(event["end"], color="0.45", lw=0.6, ls="--", alpha=0.7)

    ylo, yhi = local_ylim(Enr, target_idx, lo, hi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(lo, hi)

    ev_lo = max(0, event["start"])
    ev_hi = min(nframes - 1, event["end"])
    gap_strings = []
    for i in range(len(target_idx)):
        for j in range(i + 1, len(target_idx)):
            gap = np.abs(
                Enr[ev_lo:ev_hi+1, target_idx[i]]
                - Enr[ev_lo:ev_hi+1, target_idx[j]]
            )
            if gap.size:
                k = int(np.argmin(gap))
                min_gap = float(gap[k])
                min_frame = ev_lo + k
                gap_strings.append(
                    f"{target_numbers[i]}/{target_numbers[j]}: "
                    f"{1000.0*min_gap:.1f} meV @ {min_frame}"
                )

    char_note = ""
    if len(target_idx) >= 2:
        gap = np.abs(
            Enr[ev_lo:ev_hi+1, target_idx[0]]
            - Enr[ev_lo:ev_hi+1, target_idx[1]]
        )
        if gap.size:
            k = int(np.argmin(gap))
            fr = ev_lo + k
            a = target_idx[0]
            b = target_idx[1]
            char_note = (
                f"  |  {target_numbers[0]}: "
                f"BCF {F_bcf[fr,a]:.2f}, PCBM {F_pcbm[fr,a]:.2f}; "
                f"{target_numbers[1]}: "
                f"BCF {F_bcf[fr,b]:.2f}, PCBM {F_pcbm[fr,b]:.2f}"
            )

    subtitle = "; ".join(gap_strings)
    ax.set_title(
        f"{event_name}  (frames {event['start']}-{event['end']})\n"
        f"{subtitle}{char_note}",
        fontsize=9,
    )
    ax.set_xlabel("MD frame")
    ax.set_ylabel("KS energy [eV]")
    ax.grid(ls="--", lw=0.4, alpha=0.35)


def main():
    parser = argparse.ArgumentParser(
        description="Trajectory-B avoided-crossing zoom generator."
    )
    parser.add_argument("--prefix", default=".")
    parser.add_argument(
        "--cache",
        default="pvsk_bcf_pcbm_weights.npz",
        help="Existing projection cache; recomputed from PROCAR if absent.",
    )
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--kpoint", type=int, default=0)
    parser.add_argument("--nproc", type=int, default=48)
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Frames shown before/after each exact event window.",
    )
    parser.add_argument(
        "--band-min",
        type=int,
        default=976,
        help="Lowest 1-based VASP band shown as faint context.",
    )
    parser.add_argument(
        "--band-max",
        type=int,
        default=981,
        help="Highest 1-based VASP band shown as faint context.",
    )
    parser.add_argument(
        "--dot-size",
        type=float,
        default=18.0,
        help="Projected-character marker size.",
    )
    parser.add_argument(
        "--out",
        default="B_avoided_crossings_zoom",
        help="Output basename without extension.",
    )
    args = parser.parse_args()

    Enr, W_pvsk, W_bcf, W_pcbm = load_or_compute(
        args.prefix,
        args.cache,
        args.spin,
        args.kpoint,
        args.nproc,
    )

    W_tot = W_pvsk + W_bcf + W_pcbm + 1e-15
    F_pvsk = W_pvsk / W_tot
    F_bcf = W_bcf / W_tot
    F_pcbm = W_pcbm / W_tot
    RGB = character_rgb(F_pvsk, F_bcf, F_pcbm)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.0, 7.5),
        constrained_layout=True,
    )
    axes = axes.ravel()

    for ax, (name, event) in zip(axes, EVENTS.items()):
        plot_event(
            ax=ax,
            event_name=name,
            event=event,
            Enr=Enr,
            RGB=RGB,
            F_bcf=F_bcf,
            F_pcbm=F_pcbm,
            padding=args.padding,
            band_min=args.band_min,
            band_max=args.band_max,
            dot_size=args.dot_size,
        )

    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=COL_PVSK, markeredgecolor="none",
            markersize=7, label="perovskite character"
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=COL_BCF, markeredgecolor="none",
            markersize=7, label="BCF character"
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=COL_PCBM, markeredgecolor="none",
            markersize=7, label="PCBM character"
        ),
        Line2D(
            [0], [0], color="0.72", lw=1.2,
            label="other bands (context)"
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.025),
        fontsize=9,
    )
    fig.suptitle(
        "Trajectory B: local BCF/PCBM character exchange at avoided-crossing regions",
        fontsize=13,
        y=1.055,
    )

    png = args.out + ".png"
    pdf = args.out + ".pdf"
    fig.savefig(png, dpi=500, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:", png)
    print("Wrote:", pdf)
    print("")
    print("IMPORTANT: energies are plotted raw (unsmoothed) by default.")
    print("Event windows:")
    for name, ev in EVENTS.items():
        print(
            f"  {name}: {ev['start']}-{ev['end']}, "
            f"highlight bands {','.join(map(str, ev['bands']))}"
        )


if __name__ == "__main__":
    main()
