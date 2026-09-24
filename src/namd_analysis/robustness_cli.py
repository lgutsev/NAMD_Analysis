"""CLI for ``namd-analysis character-robustness``.

Reads the ``projection_character.csv`` a ``character-populations`` run wrote
-- the authoritative result -- and audits whether apparent two-fragment mixing
survives the absolute projection weights.  It never rewrites, filters or
renormalizes that table; every output here is additional.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .character import AtomGroupMap, CharacterError, _load_projection_manifest
from .crossings import CrossingError, read_projection_character
from .ensemble import EnsembleError, parse_episode_windows
from .io.procar import ProcarFormatError
from .io.vasp_settings import VaspSettingsError
from .projection_provenance import (
    RESOLUTION_HEADER,
    find_vasp_files,
    procar_resolution_check,
    projection_method,
)
from .projection_robustness import (
    BRACKET_ASSUMPTIONS,
    DEFAULT_CAPTURE_MINIMA,
    DEFAULT_NORMALIZED_THRESHOLDS,
    DEFAULT_RAW_MINIMA,
    INTERPRETATION_NOTE,
    SENSITIVITY_HEADER,
    ProjectionTable,
    RobustnessError,
    audit,
    automatic_marks,
    by_band_header,
    by_band_rows,
    check_marks,
    check_windows,
    focus_header,
    focus_rows,
    focus_summary,
    parse_grid,
    parse_mark,
    question_block,
    sample_header,
    sample_rows,
    sensitivity_rows,
)
from .provenance import environment, fingerprint
from .report import prepare_output, write_csv, write_json

#: The reporting threshold ``AtomGroupMap`` uses when an atom-group file does
#: not set one.  Used only when neither --quality-threshold nor a character
#: report says otherwise, and the report names which it was.
DEFAULT_QUALITY_THRESHOLD = 0.5


def build_parser(prog: str = "namd-analysis character-robustness") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Audit whether apparent fragment mixing in projection_character.csv "
            "survives the absolute (unnormalized) PAW-projector weights. Reports "
            "raw and normalized weights for every frame and band, the dominant "
            "fragment before and after normalization, capture statistics and "
            "correlations, and mixed-sample counts over a grid of normalized, "
            "capture and raw thresholds. Nothing is filtered or repaired."
        ),
    )
    parser.add_argument("--projection-character", required=True,
                        help="projection_character.csv from a character-populations run")
    parser.add_argument("--character-report", default=None,
                        help="that run's report.json: supplies the reporting threshold and "
                             "atom groups, and is fingerprinted as provenance")
    parser.add_argument("--pair", default="BCF,PCBM",
                        help="the two groups whose mixing is audited (default BCF,PCBM)")
    parser.add_argument("--quality-threshold", type=float, default=None,
                        help="the existing captured-projection reporting threshold; read from "
                             "--character-report when omitted, else 0.5. Reported, never applied")
    parser.add_argument("--normalized-thresholds", default=None,
                        help="grid of thresholds on each pair fragment's normalized fraction, "
                             "'a,b,c' or 'START:STOP:STEP' "
                             f"(default {','.join(f'{v:g}' for v in DEFAULT_NORMALIZED_THRESHOLDS)})")
    parser.add_argument("--capture-minima", default=None,
                        help="grid of captured-projection minima used to condition counts "
                             "(default 0 and 0.30:0.60:0.02; 0 and the reporting threshold are "
                             "always included)")
    parser.add_argument("--raw-minima", default=None,
                        help="grid of minimum raw weight on each pair fragment "
                             f"(default {','.join(f'{v:g}' for v in DEFAULT_RAW_MINIMA)})")
    parser.add_argument("--strata", type=int, default=4,
                        help="capture quantile strata per band (default 4, quartiles)")
    parser.add_argument("--focus-band", type=int, action="append", default=None,
                        help="write the dedicated focus report for this band (repeatable), "
                             "e.g. --focus-band 981")
    parser.add_argument("--mark", action="append", default=None, metavar="LABEL=F1,F2,...",
                        help="frames to mark on every focus report, e.g. "
                             "issue4_low_capture=1273,1286,1302 (repeatable)")
    parser.add_argument("--auto-marks", type=int, default=3,
                        help="lowest-capture, highest-capture and most-mixed frames to mark "
                             "automatically per focus band (default 3; 0 disables)")
    parser.add_argument("--episode", action="append", default=None, metavar="NAME=FIRST:LAST",
                        help="a crossing window of THIS configuration (repeatable)")
    parser.add_argument("--control-episode", action="append", default=None,
                        metavar="NAME=FIRST:LAST", help="a control window (repeatable)")
    parser.add_argument("--profile", default=None,
                        help="production_profile.json: take the crossing and control windows "
                             "of --configuration from it")
    parser.add_argument("--configuration", default=None,
                        help="configuration label in --profile (A, B or C); only that "
                             "configuration's windows are used")
    parser.add_argument("--vasp-dir", default=None,
                        help="a production frame directory holding INCAR/OUTCAR/vasprun.xml/"
                             "PROCAR, read to record LORBIT and RWIGS (default with "
                             "--projection-manifest: the first manifest frame's directory)")
    parser.add_argument("--incar", default=None, help="production INCAR (overrides --vasp-dir)")
    parser.add_argument("--outcar", default=None, help="production OUTCAR (overrides --vasp-dir)")
    parser.add_argument("--vasprun", default=None,
                        help="production vasprun.xml (overrides --vasp-dir)")
    parser.add_argument("--procar", default=None,
                        help="a production PROCAR, for its orbital layout (overrides --vasp-dir)")
    parser.add_argument("--projection-manifest", default=None,
                        help="the character run's projection manifest; with --atom-groups, "
                             "re-reads the marked frames' PROCARs to check print resolution")
    parser.add_argument("--atom-groups", default=None,
                        help="atom-group JSON for the PROCAR resolution check")
    parser.add_argument("--resolution-frames", default=None,
                        help="frames for the PROCAR resolution check, comma-separated "
                             "(default: every marked frame of every focus band)")
    parser.add_argument("--no-figures", action="store_true", help="skip the figures")
    parser.add_argument("--out", required=True, help="new output directory")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_report(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RobustnessError(f"cannot read character report {path}: {exc}") from exc


def _quality_threshold(args, report: Optional[Dict[str, Any]]):
    if args.quality_threshold is not None:
        return float(args.quality_threshold), "--quality-threshold"
    if report is not None:
        value = (report.get("atom_groups") or {}).get("min_projection_weight")
        if value is not None:
            return float(value), f"{args.character_report} atom_groups.min_projection_weight"
    return DEFAULT_QUALITY_THRESHOLD, "AtomGroupMap default (no report or flag supplied)"


def _windows(args) -> List[Any]:
    named = list(args.episode or [])
    controls = list(args.control_episode or [])
    if args.profile:
        if not args.configuration:
            raise RobustnessError("--profile needs --configuration, so only that "
                                  "configuration's windows are used")
        try:
            profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RobustnessError(f"cannot read profile {args.profile}: {exc}") from exc
        configurations = profile.get("configurations") or {}
        if args.configuration not in configurations:
            raise RobustnessError(
                f"configuration {args.configuration!r} is not in {args.profile} "
                f"(it has {sorted(configurations)})"
            )
        entry = configurations[args.configuration]
        named += [f"{name}={span}" for name, span in (entry.get("episodes") or {}).items()]
        controls += [
            f"{name}={span}" for name, span in (entry.get("control_episodes") or {}).items()
        ]
    elif args.configuration:
        raise RobustnessError("--configuration only has meaning with --profile")
    try:
        return parse_episode_windows(None, named, controls)
    except EnsembleError as exc:
        raise RobustnessError(str(exc)) from exc


def _method(args) -> Dict[str, Any]:
    found: Dict[str, Optional[Path]] = {
        "incar": None, "outcar": None, "vasprun": None, "procar": None
    }
    if args.vasp_dir:
        directory = Path(args.vasp_dir)
        if not directory.is_dir():
            raise RobustnessError(f"--vasp-dir {directory} is not a directory")
        found.update(find_vasp_files(directory))
    for key in found:
        explicit = getattr(args, key)
        if explicit:
            path = Path(explicit)
            if not path.is_file():
                raise RobustnessError(f"--{key} {path} does not exist")
            found[key] = path
    record = projection_method(
        procar=found["procar"], incar=found["incar"],
        outcar=found["outcar"], vasprun=found["vasprun"],
    )
    record["searched_directory"] = args.vasp_dir
    return record


def _resolution(args, marks_by_band, bands) -> Optional[Dict[str, Any]]:
    if not args.projection_manifest and not args.atom_groups:
        return None
    if not (args.projection_manifest and args.atom_groups):
        raise RobustnessError(
            "the PROCAR resolution check needs both --projection-manifest and --atom-groups"
        )
    if args.resolution_frames:
        try:
            frames = sorted({int(f) for f in args.resolution_frames.split(",") if f.strip()})
        except ValueError:
            raise RobustnessError("--resolution-frames must be comma-separated integers") from None
    else:
        frames = sorted({f for marks in marks_by_band.values() for m in marks for f in m.frames})
    if not frames:
        raise RobustnessError(
            "no frames for the PROCAR resolution check: pass --resolution-frames, or "
            "--focus-band so frames are marked"
        )
    manifest = _load_projection_manifest(args.projection_manifest)
    groups = AtomGroupMap.from_json(args.atom_groups)
    paths = dict(manifest.frames)
    missing = [f for f in frames if f not in paths]
    if missing:
        raise RobustnessError(f"the projection manifest has no PROCAR for frames {missing}")
    checks = [
        procar_resolution_check(paths[f], [int(b) for b in bands], groups.groups, frame=f)
        for f in frames
    ]
    return {"frames": frames, "checks": checks}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = _load_report(args.character_report)
        threshold, threshold_source = _quality_threshold(args, report)
        series = read_projection_character(args.projection_character)
        table = ProjectionTable.from_character_series(series, source=args.projection_character)
        pair = [item.strip() for item in args.pair.split(",")]
        grids = {
            "normalized": parse_grid(args.normalized_thresholds, DEFAULT_NORMALIZED_THRESHOLDS,
                                     "--normalized-thresholds"),
            "capture": parse_grid(args.capture_minima, DEFAULT_CAPTURE_MINIMA, "--capture-minima"),
            "raw": parse_grid(args.raw_minima, DEFAULT_RAW_MINIMA, "--raw-minima"),
        }
        result = audit(
            table, pair=pair, quality_threshold=threshold,
            normalized_thresholds=grids["normalized"], capture_minima=grids["capture"],
            raw_minima=grids["raw"], n_strata=args.strata,
        )
        windows = _windows(args)
        if windows:
            check_windows(table, windows)
        user_marks = [parse_mark(spec) for spec in (args.mark or [])]
        focus_bands = list(dict.fromkeys(args.focus_band or []))
        marks_by_band = {}
        for band in focus_bands:
            table.band_position(band)
            marks = user_marks + automatic_marks(table, result.arrays, band, args.auto_marks)
            check_marks(table, marks)
            marks_by_band[band] = marks
        if args.projection_manifest and not any(
            (args.vasp_dir, args.incar, args.outcar, args.vasprun, args.procar)
        ):
            # A production frame directory is where its PROCAR is; take the
            # first one the manifest names rather than leave LORBIT unread.
            first = _load_projection_manifest(args.projection_manifest).frames[0][1]
            args.vasp_dir = str(Path(first).parent)
        method = _method(args)
        resolution = _resolution(args, marks_by_band, [int(b) for b in table.bands])
    except (RobustnessError, CrossingError, CharacterError, ProcarFormatError,
            VaspSettingsError, ValueError, OSError) as exc:
        print(f"error: {exc}")
        return 2

    out = prepare_output(args.out, overwrite=args.overwrite)
    taus = result.grids["normalized_thresholds"]
    write_csv(out / "robustness_samples.csv", sample_header(table.groups, taus),
              sample_rows(table, result.pair, result.arrays, taus))
    write_csv(out / "robustness_by_band.csv", by_band_header(table.groups), by_band_rows(result))
    write_csv(out / "mixing_sensitivity.csv", SENSITIVITY_HEADER,
              sensitivity_rows(table, result.pair, result.arrays,
                               result.grids["normalized_thresholds"],
                               result.grids["capture_minima"], result.grids["raw_minima"]))

    figures: Dict[str, List[str]] = {}
    focus_reports = []
    for band, marks in marks_by_band.items():
        write_csv(out / f"band_{band}_focus.csv", focus_header(table.groups, taus),
                  focus_rows(result, band, marks, windows))
        summary = focus_summary(result, band, marks, windows)
        write_json(out / f"band_{band}_focus.json", summary)
        focus_reports.append(f"band_{band}_focus.json")
        if not args.no_figures:
            from .robustness_plots import plot_focus_band, plot_sensitivity

            figures[f"band_{band}_focus"] = plot_focus_band(
                result, band, marks, windows, out / f"band_{band}_focus", threshold
            )
            figures[f"band_{band}_mixing_sensitivity"] = plot_sensitivity(
                result, band, out / f"band_{band}_mixing_sensitivity"
            )

    if resolution is not None:
        write_csv(out / "procar_resolution.csv", RESOLUTION_HEADER,
                  [row for check in resolution["checks"] for row in check["rows"]])

    inputs = [Path(args.projection_character)]
    if args.character_report:
        inputs.append(Path(args.character_report))
    if args.profile:
        inputs.append(Path(args.profile))
    payload = {
        "command": "character-robustness",
        "environment": environment(),
        "inputs": fingerprint(inputs),
        "authoritative_source": {
            "path": args.projection_character,
            "note": (
                "the projection_character.csv written by character-populations is the "
                "authoritative result of that method. This audit reads it and writes "
                "only new files; no weight in it is changed, filtered or renormalized"
            ),
            "character_run_environment": (report or {}).get("environment"),
        },
        "projection_method": method,
        "pair": list(result.pair),
        "groups": list(table.groups),
        "bands": [int(b) for b in table.bands],
        "n_frames": int(len(table.frames)),
        "existing_reporting_threshold": {"value": threshold, "source": threshold_source},
        "grids": result.grids,
        "grid_note": (
            "every threshold is a sensitivity grid point, not a preferred value. The "
            "capture grid conditions counts and always includes 0 (every sample) and "
            "the existing reporting threshold; each row of mixing_sensitivity.csv "
            "states how many samples its capture condition set aside. Mixed means "
            "both pair fragments at or above the threshold (>=)"
        ),
        "definitions": {
            "raw_weight": "W_g = w_g * captured_projection, the fragment's share of the whole band",
            "uncaptured_weight": "1 - captured_projection, what normalization divides away",
            "unprojected_remainder": "1 - total_projection, band weight outside every ion's "
                                     "projection (equal to uncaptured for a complete partition)",
            "dominant_after_normalization": "argmax_g w_g",
            "dominant_raw_declared_only": (
                "argmax_g W_g. Identical to dominant_after_normalization by construction, "
                "because normalization divides every fragment by the same number; reported "
                "so the invariance is checked on the data"
            ),
            "dominant_before_normalization": (
                "argmax over the raw fragments AND the uncaptured weight. It differs from "
                "the normalized dominant exactly when more of the band is uncaptured than "
                "any one fragment holds"
            ),
            "pair_min_normalized": "min(w_a, w_b), the mixing metric Issue #4 counted with",
            "pair_min_worst_case": "min(W_a, W_b): none of the uncaptured weight on the pair",
            "pair_min_best_case": "largest min(f_a, f_b) any allocation of the uncaptured weight allows",
            "pair_balance": "min(W_a, W_b) / max(W_a, W_b); unchanged by normalization",
        },
        "bracket_assumptions": BRACKET_ASSUMPTIONS,
        "interpretation": INTERPRETATION_NOTE,
        "campaign": result.campaign,
        "per_band": result.per_band,
        "question": question_block(result, focus_bands) if focus_bands else None,
        "focus_reports": focus_reports,
        "windows": [
            {"name": w.name, "first": w.first, "last": w.last, "role": w.role} for w in windows
        ],
        "windows_source": (
            f"{args.profile}, configuration {args.configuration}" if args.profile
            else "command line" if windows else None
        ),
        "procar_resolution": resolution,
        "figures": figures,
    }
    write_json(out / "robustness_summary.json", payload)

    print(f"{len(table.frames)} frame(s) x {len(table.bands)} band(s), pair "
          f"{result.pair[0]}/{result.pair[1]}; projection method: {method['method']}"
          + (f" (LORBIT={method['LORBIT']})" if method["LORBIT"] is not None else ""))
    for band in focus_bands:
        record = result.band(band)
        reference = [e for e in record["mixing_by_threshold"] if abs(e["threshold"] - 0.10) < 1e-12]
        line = (f"  band {band}: capture median {record['capture']['median']:.3f}, "
                f"min {record['capture']['minimum']:.3f}")
        if reference:
            entry = reference[0]
            line += (f"; mixed at 0.10 normalized: {entry['n_mixed_normalized']}, of which "
                     f"{entry['n_normalized_mixed_that_survive_worst_case']} also clear 0.10 in "
                     "raw weights")
        print(line)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
