#!/usr/bin/env python3
"""
audit_vasp_convergence.py

Lightweight convergence audit for a trajectory made of numbered daughter
directories (e.g. 0001/, 0002/, ...), each containing an OUTCAR.

Run from the ROOT directory containing the daughter directories:

    python audit_vasp_convergence.py

Optional restricted frame range:

    python audit_vasp_convergence.py --frame-min 1485 --frame-max 1515

Outputs:
    vasp_convergence_audit.csv
    vasp_convergence_summary.txt

What is checked
---------------
For each OUTCAR:
  * file exists and is non-empty
  * OUTCAR contains a normal VASP footer / timing section
  * final electronic iteration count can be identified
  * NELM is parsed from OUTCAR
  * final SCF cycle did not hit NELM
  * final total-energy change is compared to EDIFF where available
  * final "reached required accuracy" marker is detected where present
  * ionic-step count is reported where relevant
  * E-fermi and final TOTEN are recorded for correlation with spectral anomalies

Important
---------
For single-point calculations, VASP may not print the exact same convergence
phrases as an ionic relaxation. Therefore the classification uses multiple
signals rather than one phrase.

Classification:
  PASS     : no convergence/finalization red flags found
  WARN     : ambiguous or incomplete evidence, but no clear failure
  FAIL     : clear non-convergence, NELM hit, truncated OUTCAR, or missing data
"""

import os
import re
import csv
import argparse
from glob import glob


def detect_frame_dirs(root="."):
    dirs = sorted(
        d for d in glob(os.path.join(root, "[0-9][0-9][0-9][0-9]"))
        if os.path.isdir(d)
    )
    if not dirs:
        raise FileNotFoundError(
            "No four-digit daughter directories were found in the current directory."
        )
    return dirs


def frame_label(path):
    return int(os.path.basename(os.path.normpath(path)))


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def parse_outcar(path):
    rec = {
        "exists": False,
        "size_bytes": 0,
        "has_footer": False,
        "reached_required_accuracy": False,
        "nelm": None,
        "ediff": None,
        "final_toten": None,
        "final_efermi": None,
        "ionic_steps": 0,
        "electronic_iterations_last_ionic": None,
        "max_electronic_iterations_any_ionic": None,
        "hit_nelm_last_ionic": False,
        "hit_nelm_any_ionic": False,
        "last_energy_change": None,
        "last_energy_change_abs": None,
        "energy_change_within_ediff": None,
        "has_aborting_message": False,
        "has_error_message": False,
        "status": "FAIL",
        "notes": [],
    }

    if not os.path.isfile(path):
        rec["notes"].append("OUTCAR missing")
        return rec

    rec["exists"] = True
    rec["size_bytes"] = os.path.getsize(path)

    if rec["size_bytes"] == 0:
        rec["notes"].append("OUTCAR empty")
        return rec

    re_nelm = re.compile(r"\bNELM\s*=\s*(\d+)")
    re_ediff = re.compile(r"\bEDIFF\s*=\s*([-+0-9.Ee]+)")
    re_toten = re.compile(r"free\s+energy\s+TOTEN\s*=\s*([-+0-9.Ee]+)")
    re_fermi = re.compile(r"E-fermi\s*:\s*([-+0-9.Ee]+)")
    re_iter = re.compile(r"^\s*(?:DAV|RMM|CG)\s*:\s*(\d+)\s+")
    re_ionic = re.compile(r"^\s*Iteration\s+(\d+)\(\s*(\d+)\)")
    re_dE = re.compile(r"energy-change\s*\(2\. order\)\s*:\s*([-+0-9.Ee]+)")
    re_dE_alt = re.compile(r"\bdE\s*=\s*([-+0-9.Ee]+)")

    current_ionic = None
    iter_counts = {}
    sequential_iter_count = 0
    last_iter_num = None
    last_dE = None

    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                low = line.lower()

                if "general timing and accounting informations for this job" in low:
                    rec["has_footer"] = True

                if "reached required accuracy" in low:
                    rec["reached_required_accuracy"] = True

                if "aborting loop because ediff is reached" in low:
                    # This is a normal electronic-convergence message in VASP.
                    pass

                if "error" in low and "error f=" not in low:
                    # Keep this deliberately conservative; many benign strings
                    # contain the word error, so only record presence.
                    rec["has_error_message"] = True

                if "aborting" in low and "ediff is reached" not in low:
                    rec["has_aborting_message"] = True

                m = re_nelm.search(line)
                if m:
                    rec["nelm"] = int(m.group(1))

                m = re_ediff.search(line)
                if m:
                    rec["ediff"] = safe_float(m.group(1))

                m = re_toten.search(line)
                if m:
                    rec["final_toten"] = safe_float(m.group(1))

                m = re_fermi.search(line)
                if m:
                    rec["final_efermi"] = safe_float(m.group(1))

                m = re_ionic.search(line)
                if m:
                    current_ionic = int(m.group(1))
                    rec["ionic_steps"] = max(rec["ionic_steps"], current_ionic)
                    iter_counts.setdefault(current_ionic, 0)
                    sequential_iter_count = 0
                    continue

                m = re_iter.search(line)
                if m:
                    it = int(m.group(1))
                    last_iter_num = it
                    sequential_iter_count += 1
                    key = current_ionic if current_ionic is not None else 0
                    iter_counts[key] = max(iter_counts.get(key, 0), it)
                    continue

                m = re_dE.search(line)
                if m:
                    last_dE = safe_float(m.group(1))
                    continue

                m = re_dE_alt.search(line)
                if m and "toten" not in low:
                    val = safe_float(m.group(1))
                    if val is not None:
                        last_dE = val

    except Exception as exc:
        rec["notes"].append(f"parse exception: {exc}")
        return rec

    if iter_counts:
        last_key = sorted(iter_counts.keys())[-1]
        rec["electronic_iterations_last_ionic"] = iter_counts[last_key]
        rec["max_electronic_iterations_any_ionic"] = max(iter_counts.values())

    if rec["nelm"] is not None:
        if (
            rec["electronic_iterations_last_ionic"] is not None
            and rec["electronic_iterations_last_ionic"] >= rec["nelm"]
        ):
            rec["hit_nelm_last_ionic"] = True

        if (
            rec["max_electronic_iterations_any_ionic"] is not None
            and rec["max_electronic_iterations_any_ionic"] >= rec["nelm"]
        ):
            rec["hit_nelm_any_ionic"] = True

    rec["last_energy_change"] = last_dE
    if last_dE is not None:
        rec["last_energy_change_abs"] = abs(last_dE)

    if rec["ediff"] is not None and rec["last_energy_change_abs"] is not None:
        rec["energy_change_within_ediff"] = (
            rec["last_energy_change_abs"] <= rec["ediff"]
        )

    # Classification
    clear_fail = False
    ambiguous = False

    if not rec["has_footer"]:
        clear_fail = True
        rec["notes"].append("normal VASP footer/timing section not found")

    if rec["hit_nelm_last_ionic"]:
        clear_fail = True
        rec["notes"].append("final electronic cycle reached NELM")

    if rec["has_aborting_message"]:
        ambiguous = True
        rec["notes"].append("non-EDIFF aborting message detected")

    if rec["electronic_iterations_last_ionic"] is None:
        ambiguous = True
        rec["notes"].append("could not identify final electronic iteration count")

    if rec["ediff"] is None:
        ambiguous = True
        rec["notes"].append("EDIFF not parsed")

    if rec["last_energy_change_abs"] is None:
        ambiguous = True
        rec["notes"].append("final electronic energy-change metric not parsed")
    elif rec["ediff"] is not None and not rec["energy_change_within_ediff"]:
        # This metric is useful but not universally printed in the same way,
        # so treat as warning unless NELM/footer also fail.
        ambiguous = True
        rec["notes"].append(
            f"last parsed energy change {rec['last_energy_change_abs']:.3e} "
            f"> EDIFF {rec['ediff']:.3e}"
        )

    if clear_fail:
        rec["status"] = "FAIL"
    elif ambiguous:
        rec["status"] = "WARN"
    else:
        rec["status"] = "PASS"

    if rec["reached_required_accuracy"]:
        rec["notes"].append("VASP 'reached required accuracy' marker present")

    return rec


def main():
    ap = argparse.ArgumentParser(
        description="Audit VASP OUTCAR convergence across numbered daughter directories."
    )
    ap.add_argument("--frame-min", type=int, default=None)
    ap.add_argument("--frame-max", type=int, default=None)
    ap.add_argument("--csv", default="vasp_convergence_audit.csv")
    ap.add_argument("--summary", default="vasp_convergence_summary.txt")
    args = ap.parse_args()

    frame_dirs = detect_frame_dirs(".")
    rows = []

    for d in frame_dirs:
        fr = frame_label(d)

        if args.frame_min is not None and fr < args.frame_min:
            continue
        if args.frame_max is not None and fr > args.frame_max:
            continue

        outcar = os.path.join(d, "OUTCAR")
        rec = parse_outcar(outcar)
        rec["frame"] = fr
        rec["directory"] = os.path.basename(d)
        rows.append(rec)

    rows.sort(key=lambda r: r["frame"])

    fieldnames = [
        "frame",
        "directory",
        "status",
        "exists",
        "size_bytes",
        "has_footer",
        "reached_required_accuracy",
        "nelm",
        "ediff",
        "electronic_iterations_last_ionic",
        "max_electronic_iterations_any_ionic",
        "hit_nelm_last_ionic",
        "hit_nelm_any_ionic",
        "last_energy_change",
        "last_energy_change_abs",
        "energy_change_within_ediff",
        "ionic_steps",
        "final_toten",
        "final_efermi",
        "has_aborting_message",
        "has_error_message",
        "notes",
    ]

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["notes"] = "; ".join(row["notes"])
            writer.writerow({k: out.get(k, "") for k in fieldnames})

    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for row in rows:
        counts[row["status"]] += 1

    problem_rows = [r for r in rows if r["status"] != "PASS"]

    lines = []
    lines.append("VASP convergence audit")
    lines.append("=" * 80)
    lines.append(f"Frames audited: {len(rows)}")
    lines.append(
        f"PASS: {counts['PASS']}   WARN: {counts['WARN']}   FAIL: {counts['FAIL']}"
    )
    lines.append("")

    if not problem_rows:
        lines.append("No convergence/finalization red flags detected.")
    else:
        lines.append("Frames requiring attention:")
        lines.append("")
        for r in problem_rows:
            lines.append(
                f"frame {r['frame']:04d}  {r['status']:4s}  "
                f"iters={r['electronic_iterations_last_ionic']}  "
                f"NELM={r['nelm']}  "
                f"|dE|={r['last_energy_change_abs']}  "
                f"EDIFF={r['ediff']}"
            )
            for note in r["notes"]:
                lines.append(f"    - {note}")

    lines.append("")
    lines.append("Interpretation notes")
    lines.append("- PASS means no obvious electronic-convergence or file-finalization red flag.")
    lines.append("- WARN means the OUTCAR is not clearly failed, but one diagnostic was ambiguous.")
    lines.append("- FAIL means a clear failure criterion was encountered.")
    lines.append(
        "- The 'reached required accuracy' phrase is mainly an ionic-relaxation marker and "
        "is not required for a static single-point PASS."
    )
    lines.append(
        "- For the manuscript, the strongest audit statement should be based on the absence "
        "of FAIL/WARN frames in the analyzed snapshots, especially around the spectral excursions."
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