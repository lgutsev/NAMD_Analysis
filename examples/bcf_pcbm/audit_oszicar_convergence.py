#!/usr/bin/env python3
"""
audit_oszicar_convergence.py

Fast OSZICAR-first convergence audit for a trajectory consisting of numbered
daughter directories such as:

    0001/
    0002/
    ...
    1999/

Run from the ROOT directory containing those daughter folders:

    python audit_oszicar_convergence.py

The script scans OSZICAR rather than OUTCAR, so a full ~2000-frame audit is
lightweight. INCAR is used only to obtain NELM and EDIFF when available.

Outputs
-------
oszicar_convergence_audit.csv
oszicar_convergence_summary.txt

Classification
--------------
PASS
    Final electronic cycle ends below NELM and |dE| <= EDIFF.

WARN
    OSZICAR is readable but the convergence evidence is ambiguous
    (for example dE cannot be parsed).

FAIL
    Missing/empty OSZICAR, final electronic cycle reaches NELM, or
    |dE| > EDIFF.

Notes
-----
* The parser recognizes standard VASP electronic-step lines such as DAV:, RMM:,
  and CG:.
* For a static calculation, the final electronic cycle is normally the only
  electronic cycle in OSZICAR.
* If ionic-summary lines are present, the script separates electronic cycles
  using those summaries and audits the final one.
* NELM/EDIFF are searched in:
      1) root INCAR
      2) daughter INCAR
  and otherwise fall back to CLI defaults (NELM=60, EDIFF=1e-4 unless changed).
  For your current trajectory, pass --nelm 120 --ediff 1e-6 if those are the
  known production settings and INCAR is not present.
"""

import os
import re
import csv
import argparse
from glob import glob


ELEC_RE = re.compile(
    r"^\s*(DAV|RMM|CG)\s*:\s*(\d+)\s+"
    r"([-+0-9.Ee]+)\s+([-+0-9.Ee]+)"
)

IONIC_RE = re.compile(
    r"^\s*\d+\s+F=\s*([-+0-9.Ee]+)"
)

KEYVAL_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s*=\s*([^!#;]+)"
)


def detect_frame_dirs(root="."):
    dirs = sorted(
        d for d in glob(os.path.join(root, "[0-9][0-9][0-9][0-9]"))
        if os.path.isdir(d)
    )
    if not dirs:
        raise FileNotFoundError(
            "No four-digit daughter directories found in the current directory."
        )
    return dirs


def frame_label(path):
    return int(os.path.basename(os.path.normpath(path)))


def parse_incar(path):
    vals = {}
    if not os.path.isfile(path):
        return vals

    with open(path, "r", errors="replace") as f:
        for line in f:
            m = KEYVAL_RE.match(line)
            if not m:
                continue
            key = m.group(1).upper()
            val = m.group(2).strip().split()[0]
            vals[key] = val
    return vals


def get_setting(root_incar, daughter_dir, key, cli_value, cast):
    """
    Priority:
      explicit CLI value > root INCAR > daughter INCAR > None
    """
    if cli_value is not None:
        return cli_value, "CLI"

    root_vals = parse_incar(root_incar)
    if key in root_vals:
        try:
            return cast(root_vals[key]), "root INCAR"
        except Exception:
            pass

    daughter_incar = os.path.join(daughter_dir, "INCAR")
    daughter_vals = parse_incar(daughter_incar)
    if key in daughter_vals:
        try:
            return cast(daughter_vals[key]), "daughter INCAR"
        except Exception:
            pass

    return None, "not found"


def parse_oszicar(path):
    rec = {
        "exists": False,
        "size_bytes": 0,
        "n_electronic_cycles": 0,
        "final_iteration": None,
        "final_algorithm": None,
        "final_energy": None,
        "final_dE": None,
        "final_dE_abs": None,
        "final_ionic_energy": None,
        "n_ionic_summaries": 0,
        "parse_ok": False,
    }

    if not os.path.isfile(path):
        return rec

    rec["exists"] = True
    rec["size_bytes"] = os.path.getsize(path)
    if rec["size_bytes"] == 0:
        return rec

    cycles = []
    current = []
    ionic_energies = []

    with open(path, "r", errors="replace") as f:
        for line in f:
            m = ELEC_RE.match(line)
            if m:
                try:
                    current.append({
                        "algorithm": m.group(1),
                        "iteration": int(m.group(2)),
                        "energy": float(m.group(3)),
                        "dE": float(m.group(4)),
                    })
                except Exception:
                    pass
                continue

            m = IONIC_RE.match(line)
            if m:
                if current:
                    cycles.append(current)
                    current = []
                try:
                    ionic_energies.append(float(m.group(1)))
                except Exception:
                    pass

    # Static calculations or a trailing electronic cycle may not end with an
    # ionic-summary line.
    if current:
        cycles.append(current)

    rec["n_electronic_cycles"] = len(cycles)
    rec["n_ionic_summaries"] = len(ionic_energies)

    if ionic_energies:
        rec["final_ionic_energy"] = ionic_energies[-1]

    if cycles and cycles[-1]:
        last = cycles[-1][-1]
        rec["final_iteration"] = last["iteration"]
        rec["final_algorithm"] = last["algorithm"]
        rec["final_energy"] = last["energy"]
        rec["final_dE"] = last["dE"]
        rec["final_dE_abs"] = abs(last["dE"])
        rec["parse_ok"] = True

    return rec


def classify(rec, nelm, ediff):
    notes = []

    if not rec["exists"]:
        return "FAIL", ["OSZICAR missing"]

    if rec["size_bytes"] == 0:
        return "FAIL", ["OSZICAR empty"]

    if not rec["parse_ok"]:
        return "WARN", ["could not parse final electronic SCF line"]

    status = "PASS"

    if nelm is not None:
        if rec["final_iteration"] >= nelm:
            status = "FAIL"
            notes.append(
                f"final electronic iteration {rec['final_iteration']} reached/exceeded NELM={nelm}"
            )
    else:
        notes.append("NELM unavailable; iteration-limit check skipped")
        if status == "PASS":
            status = "WARN"

    if rec["final_dE_abs"] is None:
        notes.append("final dE unavailable")
        if status == "PASS":
            status = "WARN"
    elif ediff is not None:
        if rec["final_dE_abs"] > ediff:
            status = "FAIL"
            notes.append(
                f"|dE|={rec['final_dE_abs']:.6e} > EDIFF={ediff:.6e}"
            )
    else:
        notes.append("EDIFF unavailable; dE threshold check skipped")
        if status == "PASS":
            status = "WARN"

    if not notes and status == "PASS":
        notes.append("final SCF cycle satisfies iteration and dE checks")

    return status, notes


def main():
    ap = argparse.ArgumentParser(
        description="Fast OSZICAR convergence audit across numbered daughter directories."
    )
    ap.add_argument("--frame-min", type=int, default=None)
    ap.add_argument("--frame-max", type=int, default=None)

    ap.add_argument(
        "--nelm",
        type=int,
        default=None,
        help="Override NELM. If omitted, root/daughter INCAR is searched.",
    )
    ap.add_argument(
        "--ediff",
        type=float,
        default=None,
        help="Override EDIFF. If omitted, root/daughter INCAR is searched.",
    )

    ap.add_argument(
        "--csv",
        default="oszicar_convergence_audit.csv",
    )
    ap.add_argument(
        "--summary",
        default="oszicar_convergence_summary.txt",
    )
    args = ap.parse_args()

    frame_dirs = detect_frame_dirs(".")
    root_incar = os.path.join(".", "INCAR")

    rows = []

    for d in frame_dirs:
        fr = frame_label(d)

        if args.frame_min is not None and fr < args.frame_min:
            continue
        if args.frame_max is not None and fr > args.frame_max:
            continue

        nelm, nelm_source = get_setting(
            root_incar, d, "NELM", args.nelm, int
        )
        ediff, ediff_source = get_setting(
            root_incar, d, "EDIFF", args.ediff, float
        )

        rec = parse_oszicar(os.path.join(d, "OSZICAR"))
        status, notes = classify(rec, nelm, ediff)

        row = {
            "frame": fr,
            "directory": os.path.basename(d),
            "status": status,
            "nelm": nelm,
            "nelm_source": nelm_source,
            "ediff": ediff,
            "ediff_source": ediff_source,
            **rec,
            "notes": "; ".join(notes),
        }
        rows.append(row)

    rows.sort(key=lambda x: x["frame"])

    fieldnames = [
        "frame",
        "directory",
        "status",
        "exists",
        "size_bytes",
        "nelm",
        "nelm_source",
        "ediff",
        "ediff_source",
        "n_electronic_cycles",
        "n_ionic_summaries",
        "final_algorithm",
        "final_iteration",
        "final_energy",
        "final_dE",
        "final_dE_abs",
        "final_ionic_energy",
        "parse_ok",
        "notes",
    ]

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for row in rows:
        counts[row["status"]] += 1

    flagged = [r for r in rows if r["status"] != "PASS"]

    lines = []
    lines.append("OSZICAR convergence audit")
    lines.append("=" * 80)
    lines.append(f"Frames audited: {len(rows)}")
    lines.append(
        f"PASS: {counts['PASS']}   WARN: {counts['WARN']}   FAIL: {counts['FAIL']}"
    )
    lines.append("")

    if rows:
        nelm_sources = sorted(set(str(r["nelm_source"]) for r in rows))
        ediff_sources = sorted(set(str(r["ediff_source"]) for r in rows))
        lines.append(f"NELM source(s): {', '.join(nelm_sources)}")
        lines.append(f"EDIFF source(s): {', '.join(ediff_sources)}")
        lines.append("")

    if not flagged:
        lines.append("All audited snapshots passed the OSZICAR convergence checks.")
    else:
        lines.append("Flagged snapshots:")
        lines.append("")
        for r in flagged:
            lines.append(
                f"frame {r['frame']:04d}  {r['status']:4s}  "
                f"alg={r['final_algorithm']}  "
                f"iter={r['final_iteration']}  "
                f"NELM={r['nelm']}  "
                f"|dE|={r['final_dE_abs']}  "
                f"EDIFF={r['ediff']}"
            )
            lines.append(f"    {r['notes']}")

    lines.append("")
    lines.append("Interpretation")
    lines.append(
        "- PASS means the final parsed electronic cycle ended below NELM and "
        "its final |dE| did not exceed EDIFF."
    )
    lines.append(
        "- WARN means the file was present but one required convergence datum "
        "could not be established."
    )
    lines.append(
        "- FAIL means a clear convergence problem was found: missing/empty "
        "OSZICAR, NELM reached, or |dE| > EDIFF."
    )

    summary_text = "\n".join(lines)

    with open(args.summary, "w") as f:
        f.write(summary_text)
        f.write("\n")

    print(summary_text)
    print("")
    print("Wrote:")
    print(" ", args.csv)
    print(" ", args.summary)


if __name__ == "__main__":
    main()