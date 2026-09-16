"""CLI for ``namd-analysis character-prepare``.

Generates the three configuration files a character run needs, plus an audit
of how every value was decided, and optionally a batch script and an immediate
preflight.  It prints what was inferred separately from what came from a
preset and what still needs a human, because those are not the same kind of
claim and a reader has to be able to tell them apart.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

from .character import AtomGroupMap, CharacterError, preflight_report
from .populations import ConfigError, StateMap
from .prepare import PrepareError, SlurmOptions, prepare_campaign
from .presets import campaign_names, preset_names
from .provenance import environment
from .report import write_json

#: Default SHPROP filename pattern inside a campaign directory.
DEFAULT_SHPROP_GLOB = "SHPROP.*"


def build_parser(prog: str = "namd-analysis character-prepare") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Inspect a campaign and write state_map.json, atom_groups.json, "
            "projection_manifest.json and an audit of how each value was decided. "
            "Ambiguity is refused, not resolved: subsystem names and atom "
            "boundaries are never inferred, and a column layout is accepted only "
            "when exactly one reading of the table fits."
        ),
    )
    parser.add_argument(
        "--shprop-dir",
        required=True,
        help="directory holding the original SHPROP.* histories",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help=(
            f"SHPROP filenames within --shprop-dir (default: every {DEFAULT_SHPROP_GLOB}). "
            "Pass the ORIGINAL histories, never SHPROP.master"
        ),
    )
    parser.add_argument(
        "--projection-dir",
        required=True,
        help="campaign directory holding numbered frame subdirectories with PROCARs",
    )
    parser.add_argument(
        "--procar-name", default="PROCAR", help="filename inside each frame directory"
    )
    parser.add_argument(
        "--preset",
        default=None,
        help=f"campaign preset, one of {preset_names()}",
    )
    parser.add_argument(
        "--campaign",
        default=None,
        help="campaign letter within the preset (state order and atom partition differ)",
    )
    parser.add_argument(
        "--atom-groups",
        default=None,
        help="JSON partition to use instead of a preset's; validated against the PROCAR",
    )
    parser.add_argument(
        "--frame-mode",
        choices=("dish-cyclic", "linear"),
        required=True,
        help="how NAMD time points select frames; never inferred from filenames",
    )
    parser.add_argument(
        "--cycle-length",
        type=int,
        default=None,
        help="explicit cyclic period; overrides both NSW-1 and directory coverage",
    )
    parser.add_argument("--out", default=".", help="directory to write the configuration into")
    parser.add_argument(
        "--results-dir",
        default="results/character",
        help="output directory the generated batch script will pass to --out",
    )
    parser.add_argument(
        "--write-sbatch", action="store_true", help="also write run_character_test.sbatch"
    )
    parser.add_argument("--slurm-account", default=None)
    parser.add_argument("--slurm-partition", default=None)
    parser.add_argument("--slurm-memory", default="32G")
    parser.add_argument("--slurm-time", default="04:00:00")
    parser.add_argument("--slurm-cpus", type=int, default=1)
    parser.add_argument("--slurm-nodes", type=int, default=1)
    parser.add_argument("--slurm-tasks", type=int, default=1)
    parser.add_argument("--slurm-job-name", default="character")
    parser.add_argument(
        "--run-preflight",
        action="store_true",
        help="after writing, run the ordinary preflight against exactly these files",
    )
    parser.add_argument(
        "--shprop-chunk-rows",
        type=int,
        default=None,
        help="passed through to the generated batch script's analysis step",
    )
    return parser


def _shprop_files(directory: str, names: Optional[Sequence[str]]) -> List[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise PrepareError(f"{root}: SHPROP directory does not exist")
    if names:
        paths = [root / name for name in names]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise PrepareError(f"no such SHPROP file(s): {missing}")
    else:
        paths = sorted(root.glob(DEFAULT_SHPROP_GLOB))
        if not paths:
            raise PrepareError(
                f"{root}: no {DEFAULT_SHPROP_GLOB} files. Name them with --files."
            )
    master = [p.name for p in paths if p.name.lower().endswith(".master")]
    if master:
        raise PrepareError(
            f"{master} looks like an averaged master file. Character analysis needs "
            "the ORIGINAL histories: averaging before projection destroys the "
            "NAMDTINI alignment that selects each history's PROCAR frames."
        )
    return paths


def _print_summary(prepared, args) -> None:
    report = prepared.report
    shprop = report["shprop"]
    print("Prepared:")
    for path in report["generated_files"]:
        print(f"  {path}")
    print()
    print("Inferred from the SHPROP tables:")
    rationale = report["population_column_rationale"]
    print(
        f"  population columns {report['population_columns']['value']} "
        f"(time column {report['time_column']['value']}); "
        f"{len(rationale['candidates'])} candidate "
        "layout(s) tested, one survived"
    )
    if not rationale["sample_covers_whole_file"]:
        print(
            f"    tested against the first {rationale['sampled_rows_per_file']} of "
            f"{rationale['total_rows_per_file']} rows; the analysis validates the rest"
        )
    print(
        f"  basis BMIN={shprop['BMIN']} BMAX={shprop['BMAX']} "
        f"({shprop['basis_size']} states), {shprop['n_files']} file(s), "
        f"{shprop['rows']} rows x {shprop['columns']} columns"
    )
    print(f"  NAMDTINI {shprop['NAMDTINI']}, NSW {shprop['NSW']}")
    print()
    print("Inferred from the projection directory:")
    projection = report["projection"]
    print(
        f"  {projection['discovered_frames']} frame(s) "
        f"{projection['first_frame']}..{projection['last_frame']}, "
        f"contiguous={projection['contiguous']}, "
        f"padding={projection['zero_padding_width']}, "
        f"manifest={report['manifest']['detail']}"
    )
    procar = report["representative_procar"]
    print(
        f"  representative PROCAR (frame {procar['frame']}): {procar['n_ions']} ions, "
        f"{procar['n_bands']} bands, {procar['n_kpoints']} k-point(s)"
    )
    print()
    preset = report["preset"]["value"]
    if preset:
        print(f"From preset {preset} (a human's declaration, not inference):")
        print(f"  state groups {list(report['state_map']['payload']['groups'])}")
        validation = report["atom_groups"]["validation"]
        print(
            f"  atom groups {validation['ions_per_group']}, "
            f"{validation['assigned_ions']}/{validation['procar_ions']} ions assigned, "
            f"complete={validation['covers_exactly']}"
        )
    else:
        print("From preset: none given")
    print()
    cycle = report["cycle_length"]
    print("Cycle length:")
    print(
        f"  coverage proposes {cycle['coverage_derived_cycle_length']}, "
        f"NSW-1 gives {cycle['header_derived_periods_nsw_minus_1']}, "
        f"selected {cycle['selected']} (source {cycle['source']})"
    )
    print(f"  {cycle['note']}")
    print()
    print(f"Frame mode: {report['frame_mode']['value']} (explicit; never inferred)")
    for warning in report["warnings"]:
        print(f"  warning: {warning}")
    if prepared.unresolved:
        print()
        print("Still needs confirmation:")
        for item in prepared.unresolved:
            print(f"  - {item}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preset and args.campaign is None and len(campaign_names(args.preset)) > 1:
        print(
            f"error: preset {args.preset!r} covers campaigns "
            f"{campaign_names(args.preset)}; name one with --campaign"
        )
        return 2
    extra: List[str] = []
    if args.shprop_chunk_rows is not None:
        extra.append(f"--shprop-chunk-rows {args.shprop_chunk_rows}")
    try:
        paths = _shprop_files(args.shprop_dir, args.files)
        prepared = prepare_campaign(
            paths,
            args.projection_dir,
            args.out,
            args.frame_mode,
            preset=args.preset,
            campaign=args.campaign,
            cycle_length=args.cycle_length,
            atom_groups_json=args.atom_groups,
            slurm=SlurmOptions(
                account=args.slurm_account,
                partition=args.slurm_partition,
                nodes=args.slurm_nodes,
                tasks=args.slurm_tasks,
                cpus=args.slurm_cpus,
                memory=args.slurm_memory,
                time=args.slurm_time,
                job_name=args.slurm_job_name,
            ),
            write_sbatch=args.write_sbatch,
            results_dir=args.results_dir,
            procar_name=args.procar_name,
            extra_run_args=extra,
        )
    except (PrepareError, ConfigError, CharacterError, OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2

    out = Path(args.out)
    report_path = out / "prepare_report.json"
    write_json(report_path, {"environment": environment(), **prepared.report})
    prepared.written.append(report_path)
    prepared.report["generated_files"].append(str(report_path))
    _print_summary(prepared, args)

    status = 0 if prepared.ok else 3
    if args.run_preflight:
        print()
        print("Preflight:")
        try:
            state_map = StateMap.from_json(prepared.state_map_path)
            atom_groups = AtomGroupMap.from_json(prepared.atom_groups_path)
            report = preflight_report(
                [str(p) for p in _shprop_files(args.shprop_dir, args.files)],
                state_map,
                str(prepared.manifest_path),
                atom_groups,
                args.frame_mode,
            )
        except (CharacterError, ConfigError, PrepareError, ValueError) as exc:
            print(f"  could not run: {exc}")
            return max(status, 2)
        window = report.get("band_window") or {}
        frames = report.get("frames") or {}
        coverage = report.get("atom_coverage") or {}
        if window:
            print(f"  basis: {window['basis_size']} states")
        print(f"  SHPROP files: {report['n_shprop_files']}")
        if frames:
            print(
                f"  projection frames: {frames['frames_required_by_shprop']} required of "
                f"{frames['manifest_declared_frames']} declared"
            )
        if report.get("representative_procar"):
            print(f"  ions: {report['representative_procar']['n_ions']}")
        if coverage:
            print(
                "  atom coverage: "
                + ("complete" if coverage.get("complete_atoms") else "partial")
                + f" ({coverage['assigned_ions']}/{coverage['procar_ions']})"
            )
        for problem in report.get("problems", []):
            print(f"  PROBLEM: {problem}")
        for warning in report.get("warnings", []):
            print(f"  warning: {warning}")
        print(f"  result: {'OK' if report['ok'] else 'BLOCKED'}")
        if not report["ok"]:
            status = max(status, 2)
    return status
