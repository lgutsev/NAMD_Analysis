"""CLIs for the full-space validation of PROCAR fragment fractions.

``character-fullspace-prepare`` chooses frames, derives the INCARs from the
production run and writes a batch script; it runs no VASP.
``character-fullspace-compare`` reads what that run produced and sets the
full-space fragment fractions beside the PROCAR ones.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .character import AtomGroupMap, CharacterError, _load_projection_manifest
from .crossings import CrossingError, read_projection_character
from .fullspace import (
    SELECTION_RULES,
    FullspaceError,
    compare_sample,
    comparison_header,
    derive_incars,
    fragment_fractions,
    nearest_atom_integrals,
    procar_reference,
    readings,
    render_job,
    rerun_check,
    select_frames,
)
from .io.procar import ProcarFormatError
from .io.vasp_settings import VaspSettingsError, read_incar, render_incar
from .io.volumetric import VolumetricFormatError, read_bader_acf, read_volumetric
from .projection_robustness import (
    BRACKET_ASSUMPTIONS,
    DEFAULT_NORMALIZED_THRESHOLDS,
    ProjectionTable,
    RobustnessError,
    audit,
    parse_grid,
    parse_mark,
)
from .provenance import environment, fingerprint
from .report import prepare_output, write_csv, write_json
from .robustness_cli import _method, _windows

MANIFEST_NAME = "validation_manifest.json"


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------


def build_prepare_parser(
    prog: str = "namd-analysis character-fullspace-prepare",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Prepare an independent full-space check of PROCAR fragment fractions: "
            "choose frames of a focus band, derive a single-point SCF and a "
            "band-decomposed PARCHG step from the production INCAR, and write a "
            "batch script that partitions each band density over Bader basins of "
            "the all-electron reference density. Runs no VASP."
        ),
    )
    parser.add_argument("--projection-character", required=True,
                        help="projection_character.csv of the character run")
    parser.add_argument("--projection-manifest", required=True,
                        help="its projection manifest: each frame's PROCAR directory is the "
                             "production frame directory POSCAR/POTCAR/KPOINTS are taken from")
    parser.add_argument("--atom-groups", required=True,
                        help="the atom-group JSON; copied into the campaign for the comparison")
    parser.add_argument("--band", type=int, required=True, help="focus band, e.g. 981")
    parser.add_argument("--bands", default=None,
                        help="all bands to write PARCHGs for (IBAND), comma-separated; must "
                             "include --band. Well-behaved bands make a method control, e.g. "
                             "981,977,978. Default: --band only")
    parser.add_argument("--pair", default="BCF,PCBM")
    parser.add_argument("--base-incar", default=None,
                        help="the production INCAR (default: the INCAR in the first selected "
                             "frame's directory)")
    parser.add_argument("--worst", type=int, default=3, help="lowest-capture frames (default 3)")
    parser.add_argument("--mixed", type=int, default=2,
                        help="frames where PROCAR reports the most mixing (default 2)")
    parser.add_argument("--controls", type=int, default=1,
                        help="highest-capture control frames (default 1)")
    parser.add_argument("--typical", type=int, default=1,
                        help="median-capture frames (default 1)")
    parser.add_argument("--include", action="append", default=None, metavar="LABEL=F1,F2",
                        help="frames to include by name, e.g. issue4_low_capture=1273,1286,1302")
    parser.add_argument("--episode", action="append", default=None, metavar="NAME=FIRST:LAST")
    parser.add_argument("--control-episode", action="append", default=None,
                        metavar="NAME=FIRST:LAST")
    parser.add_argument("--profile", default=None,
                        help="production_profile.json, for --configuration's windows")
    parser.add_argument("--configuration", default=None)
    parser.add_argument("--account", default=None, help="SBATCH --account")
    parser.add_argument("--partition", default=None, help="SBATCH --partition")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--ntasks-per-node", type=int, default=48)
    parser.add_argument("--walltime", default="24:00:00")
    parser.add_argument("--out", required=True, help="new campaign directory")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _bands(args) -> List[int]:
    if not args.bands:
        return [int(args.band)]
    try:
        bands = list(dict.fromkeys(int(b) for b in args.bands.split(",") if b.strip()))
    except ValueError:
        raise FullspaceError("--bands must be comma-separated integers") from None
    if int(args.band) not in bands:
        raise FullspaceError(f"--bands {bands} must include the focus band {args.band}")
    return bands


def _method_args(frame_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(vasp_dir=str(frame_dir), incar=None, outcar=None,
                              vasprun=None, procar=None)


def prepare_main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_prepare_parser().parse_args(argv)
    try:
        bands = _bands(args)
        groups = AtomGroupMap.from_json(args.atom_groups)
        series = read_projection_character(args.projection_character)
        table = ProjectionTable.from_character_series(series, source=args.projection_character)
        if sorted(groups.names) != sorted(table.groups):
            raise FullspaceError(
                f"the atom groups {groups.names} do not match the projection table's "
                f"groups {table.groups}"
            )
        for band in bands:
            table.band_position(band)
        result = audit(table, pair=[p.strip() for p in args.pair.split(",")])
        windows = _windows(args)
        include = [parse_mark(spec) for spec in (args.include or [])]
        selected = select_frames(
            result, args.band, windows, include,
            n_worst=args.worst, n_mixed=args.mixed,
            n_controls=args.controls, n_typical=args.typical,
        )
        manifest = _load_projection_manifest(args.projection_manifest)
        procars = dict(manifest.frames)
        missing = [v.frame for v in selected if v.frame not in procars]
        if missing:
            raise FullspaceError(f"the projection manifest has no PROCAR for frames {missing}")
        for entry in selected:
            entry.source_dir = str(Path(procars[entry.frame]).resolve().parent)
        base_path = (
            Path(args.base_incar) if args.base_incar
            else Path(selected[0].source_dir) / "INCAR"
        )
        base = read_incar(base_path)
        frame_incars: Dict[int, str] = {}
        for entry in selected:
            own = Path(entry.source_dir) / "INCAR"
            if own.is_file():
                tags = read_incar(own)
                if tags != base:
                    differing = sorted(
                        k for k in set(tags) | set(base) if tags.get(k) != base.get(k)
                    )
                    raise FullspaceError(
                        f"frame {entry.frame}'s INCAR differs from {base_path} in {differing}; "
                        "one derived INCAR cannot stand for both"
                    )
                frame_incars[entry.frame] = str(own)
        scf, parchg, changes = derive_incars(base, bands)
        method = _method(_method_args(Path(selected[0].source_dir)))
        reference = procar_reference(table, [v.frame for v in selected], bands, result.pair)
    except (FullspaceError, RobustnessError, CrossingError, CharacterError,
            VaspSettingsError, ProcarFormatError, ValueError, OSError) as exc:
        print(f"error: {exc}")
        return 2

    out = prepare_output(args.out, overwrite=args.overwrite)
    shutil.copyfile(args.atom_groups, out / "atom_groups.json")
    header = (
        "Derived by namd-analysis character-fullspace-prepare from\n"
        f"{base_path}\nSee validation_manifest.json for every change and why."
    )
    for entry in selected:
        work = out / f"frame_{entry.frame}"
        work.mkdir()
        (work / "INCAR.scf").write_text(render_incar(scf, header + "\nStep 1: SCF"))
        (work / "INCAR.parchg").write_text(render_incar(parchg, header + "\nStep 2: PARCHG"))
    lines = ["frame\tsource_dir"] + [f"{v.frame}\t{v.source_dir}" for v in selected]
    (out / "frames.tsv").write_text("\n".join(lines) + "\n")
    job = render_job(args.band, bands, len(selected), args.account, args.partition,
                     args.nodes, args.ntasks_per_node, args.walltime,
                     campaign=str(out.resolve()))
    job_path = out / "run_fullspace_validation.sbatch"
    job_path.write_text(job)
    job_path.chmod(0o755)

    payload: Dict[str, Any] = {
        "command": "character-fullspace-prepare",
        "environment": environment(),
        "inputs": fingerprint(
            [Path(args.projection_character), Path(args.projection_manifest),
             Path(args.atom_groups), base_path]
        ),
        "focus_band": int(args.band),
        "bands": bands,
        "pair": list(result.pair),
        "groups": list(table.groups),
        "frames": [
            {"frame": v.frame, "reasons": v.reasons, "source_dir": v.source_dir,
             "production_procar": str(procars[v.frame])}
            for v in selected
        ],
        "selection_rules": SELECTION_RULES,
        "selection_counts": {
            "worst": args.worst, "mixed": args.mixed,
            "controls": args.controls, "typical": args.typical,
        },
        "windows": [
            {"name": w.name, "first": w.first, "last": w.last, "role": w.role} for w in windows
        ],
        "base_incar": str(base_path),
        "frame_incars_checked_equal_to_base": frame_incars,
        "incar_changes": changes,
        "projection_method": method,
        "procar_reference": reference,
        "partition": {
            "primary": "Bader basins of AECCAR0 + AECCAR2 (bader -ref CHGCAR_sum -vac off)",
            "optional": (
                "nearest-atom (Voronoi) cells, computed by character-fullspace-compare "
                "--voronoi from the PARCHG files themselves"
            ),
            "why": (
                "both partitions cover the whole cell, so the interstitial and vacuum "
                "density the PAW projections miss is assigned to a fragment rather than "
                "divided away. The Bader basins come from the total density of the same "
                "frame, not from the band being measured, so the partition is fixed "
                "before the band density is integrated over it"
            ),
        },
        "expected_outputs_per_frame": (
            ["PROCAR.rerun", "OUTCAR.scf"]
            + [f"ACF_{b}.dat" for b in bands] + [f"PARCHG_{b}" for b in bands]
        ),
        "bracket_assumptions": BRACKET_ASSUMPTIONS,
        "next_step": (
            "export VASP_CMD='srun vasp_gam' (as appropriate), load bader and the VTST "
            "scripts, then: sbatch run_fullspace_validation.sbatch. When every frame "
            "directory holds DONE: namd-analysis character-fullspace-compare --campaign "
            f"{out} --out <new directory>"
        ),
    }
    write_json(out / MANIFEST_NAME, payload)

    print(f"{len(selected)} frame(s) for band {args.band} (IBAND {' '.join(map(str, bands))}):")
    for entry in selected:
        print(f"  {entry.frame}: {', '.join(entry.reasons)}")
    print(f"projection method: {method['method']}"
          + (f" (LORBIT={method['LORBIT']})" if method["LORBIT"] is not None else ""))
    if not args.account or not args.partition:
        print("note: --account/--partition not given; the batch script carries "
              "REQUIRED_EDIT placeholders and will be refused by sbatch until edited")
    print(f"wrote {out}")
    return 0


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------


def build_compare_parser(
    prog: str = "namd-analysis character-fullspace-compare",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Compare full-space fragment fractions of band densities (Bader basins, "
            "optionally nearest-atom cells) with the PROCAR normalized fractions, raw "
            "weights and allocation bracket, for a campaign written by "
            "character-fullspace-prepare."
        ),
    )
    parser.add_argument("--campaign", required=True,
                        help="the directory character-fullspace-prepare wrote")
    parser.add_argument("--voronoi", action="store_true",
                        help="also integrate each PARCHG over nearest-atom cells")
    parser.add_argument("--thresholds", default=None,
                        help="thresholds for the readings, as for character-robustness")
    parser.add_argument("--bracket-tolerance", type=float, default=0.0,
                        help="slack when testing whether a full-space fraction lies inside "
                             "the allocation bracket (default 0)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def compare_main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_compare_parser().parse_args(argv)
    campaign = Path(args.campaign)
    try:
        manifest = json.loads((campaign / MANIFEST_NAME).read_text(encoding="utf-8"))
        groups = AtomGroupMap.from_json(campaign / "atom_groups.json")
        thresholds = parse_grid(args.thresholds, DEFAULT_NORMALIZED_THRESHOLDS, "--thresholds")
        pair = tuple(manifest["pair"])
        natoms = sum(len(a) for a in groups.groups.values()) if groups.complete_atoms else None
        records: List[Dict[str, Any]] = []
        missing: List[Dict[str, Any]] = []
        reruns: List[Dict[str, Any]] = []
        for frame_entry in manifest["frames"]:
            frame = int(frame_entry["frame"])
            work = campaign / f"frame_{frame}"
            for band in manifest["bands"]:
                reference = manifest["procar_reference"][f"{frame}:{band}"]
                rerun = None
                rerun_path = work / "PROCAR.rerun"
                if rerun_path.is_file():
                    rerun = rerun_check(rerun_path, reference, groups.groups)
                    reruns.append({"frame": frame, "band": int(band), **rerun})
                else:
                    missing.append({"frame": frame, "band": int(band), "file": str(rerun_path),
                                    "consequence": "rerun reproduction check not possible"})
                methods = []
                acf = work / f"ACF_{band}.dat"
                if acf.is_file():
                    charges = read_bader_acf(acf, natoms=natoms)
                    methods.append(("bader", fragment_fractions(
                        charges.charges, groups.groups, charges.vacuum_charge)))
                else:
                    missing.append({"frame": frame, "band": int(band), "file": str(acf)})
                if args.voronoi:
                    parchg = work / f"PARCHG_{band}"
                    if parchg.is_file():
                        volume = read_volumetric(parchg)
                        methods.append(("voronoi", fragment_fractions(
                            nearest_atom_integrals(volume), groups.groups)))
                    else:
                        missing.append({"frame": frame, "band": int(band), "file": str(parchg)})
                for method, fullspace in methods:
                    sample = compare_sample(reference, fullspace, pair, args.bracket_tolerance)
                    sample.update(
                        frame=frame, band=int(band), method=method,
                        reasons=frame_entry["reasons"],
                        rerun_max_abs_raw_difference=(
                            None if rerun is None else rerun["max_abs_raw_difference"]),
                        rerun_captured_difference=(
                            None if rerun is None else rerun["captured_difference"]),
                    )
                    records.append(sample)
        if not records:
            raise FullspaceError(
                f"no validation output found under {campaign}: expected ACF_<band>.dat "
                "(or PARCHG_<band> with --voronoi) in the frame directories. Has the "
                "batch job run?"
            )
    except (FullspaceError, RobustnessError, CharacterError, VolumetricFormatError,
            ProcarFormatError, KeyError, ValueError, OSError) as exc:
        print(f"error: {exc}")
        return 2

    out = prepare_output(args.out, overwrite=args.overwrite)
    names = list(manifest["groups"])
    rows = []
    for r in records:
        rows.append(
            [r["frame"], r["band"], r["method"], ";".join(r["reasons"]), r["integral"],
             r["unassigned_fraction"], r["undeclared_atoms_fraction"],
             r["captured_projection"], r["uncaptured_weight"]]
            + [r["fullspace"][g] for g in names]
            + [r["procar_normalized"][g] for g in names]
            + [r["procar_raw"][g] for g in names]
            + [r["uncaptured_allocated"][g] for g in names]
            + [r["uncaptured_proportional_share"][g] for g in names]
            + [f"{pair[0]}/{pair[1]}", r["fullspace_pair_min"],
               r["procar_pair_min_normalized"], r["procar_pair_min_worst_case"],
               r["procar_pair_min_best_case"], r["fullspace_over_normalized_pair_min"],
               r["fullspace_within_allocation_bracket"],
               r["rerun_max_abs_raw_difference"], r["rerun_captured_difference"]]
        )
    write_csv(out / "fullspace_comparison.csv", comparison_header(names), rows)
    methods = sorted({r["method"] for r in records})
    payload = {
        "command": "character-fullspace-compare",
        "environment": environment(),
        "campaign": str(campaign),
        "inputs": fingerprint([campaign / MANIFEST_NAME, campaign / "atom_groups.json"]),
        "focus_band": manifest["focus_band"],
        "pair": list(pair),
        "methods": methods,
        "records": records,
        "missing_outputs": missing,
        "rerun_checks": reruns,
        "readings": [
            readings(records, int(band), method, thresholds)
            for band in manifest["bands"] for method in methods
        ],
        "definitions": {
            "fullspace_g": "the band density integrated over fragment g's atoms' cells, "
                           "divided by the band's whole integral (unassigned included)",
            "uncaptured_allocated_to_g": "(fullspace_g - raw_g) / (1 - captured): where the "
                                         "weight PROCAR missed actually went",
            "uncaptured_proportional_share_g": "raw_g / captured: where normalization put it",
            "fullspace_over_normalized_pair_min": "min(f_a, f_b) / min(w_a, w_b)",
        },
        "caveats": [
            "a PARCHG is the plane-wave part of |psi|^2 on the FFT grid; how VASP treats "
            "the one-centre PAW terms in it depends on version and settings, so inside "
            "the augmentation spheres it need not equal the all-electron density. Its "
            "integral is recorded per sample (fullspace_integral) so a departure from "
            "the expected normalization is visible",
            "if rerun_max_abs_raw_difference is not small, the validation's SCF did not "
            "reproduce the production projection, and band i of the rerun may be a "
            "different state from the band i the character analysis used",
            BRACKET_ASSUMPTIONS,
        ],
        "missing_outputs_note": (
            "every expected output that was absent is listed; its sample is absent from "
            "the comparison and from the readings, and is not estimated"
        ),
    }
    write_json(out / "fullspace_summary.json", payload)
    print(f"{len(records)} comparison(s) over {len(manifest['frames'])} frame(s); "
          f"methods: {', '.join(methods)}; missing outputs: {len(missing)}")
    print(f"wrote {out}")
    return 0
