"""Top-level command dispatcher.

The main CLI remains in :mod:`namd_analysis.cli`.  The frame-dependent
character command is kept in its own module because it has a distinct, heavy
I/O surface (many PROCAR files) while preserving the public
``namd-analysis character-populations`` spelling.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from . import character_cli, cli, prepare_cli


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "character-populations":
        return character_cli.main(args[1:])
    if args and args[0] == "character-preflight":
        # Same code path; the alias just makes the cheap check discoverable.
        return character_cli.main([*args[1:], "--preflight"])
    if args and args[0] in ("character-prepare", "character-init"):
        return prepare_cli.main(args[1:])
    if args and args[0] == "character-sbatch":
        # Preparation with the batch script as the point of the exercise.
        return prepare_cli.main([*args[1:], "--write-sbatch"])
    if args in (["--help"], ["-h"]):
        parser = cli.build_parser()
        text = parser.format_help().rstrip()
        text += (
            "\n\nAdditional commands:\n"
            "  character-populations  projection-weight SHPROP populations with "
            "frame-dependent PROCAR subsystem character\n"
            "  character-preflight    check configuration, SHPROP provenance, frame "
            "alignment and PROCAR structure without parsing any projections\n"
            "  character-prepare      inspect a campaign and write state_map.json, "
            "atom_groups.json, projection_manifest.json and an audit of how every "
            "value was decided (alias: character-init)\n"
            "  character-sbatch       character-prepare with --write-sbatch\n"
            "\nRun 'namd-analysis character-populations --help' for their options.\n"
        )
        print(text)
        return 0
    return cli.main(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
