"""Top-level command dispatcher.

The main CLI remains in :mod:`namd_analysis.cli`.  The frame-dependent
character command is kept in its own module because it has a distinct, heavy
I/O surface (many PROCAR files) while preserving the public
``namd-analysis character-populations`` spelling.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from . import character_cli, cli


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "character-populations":
        return character_cli.main(args[1:])
    if args in (["--help"], ["-h"]):
        parser = cli.build_parser()
        text = parser.format_help().rstrip()
        text += (
            "\n\nAdditional command:\n"
            "  character-populations  projection-weight SHPROP populations with "
            "frame-dependent PROCAR subsystem character\n"
            "\nRun 'namd-analysis character-populations --help' for its options.\n"
        )
        print(text)
        return 0
    return cli.main(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
