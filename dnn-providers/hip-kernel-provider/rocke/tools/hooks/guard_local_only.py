#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# guard_local_only.py -- pre-commit guard that blocks local-only scratch files
# from ever being staged/committed inside the rocke tree. Recurring hygiene gap:
# CLAUDE.md / SANDBOX_NOTES.md / PROBLEM.md / .claude/ have repeatedly landed in
# repo worktrees during agent sessions. Receives staged file paths as argv.

import sys
from pathlib import Path

# Basenames and path fragments that must never be committed.
BLOCKED_NAMES = {"CLAUDE.md", "SANDBOX_NOTES.md", "PROBLEM.md"}
BLOCKED_DIRS = {".claude"}


def main(argv: list[str]) -> int:
    bad = []
    for f in argv:
        p = Path(f)
        if p.name in BLOCKED_NAMES:
            bad.append(f)
        elif any(part in BLOCKED_DIRS for part in p.parts):
            bad.append(f)
    if bad:
        sys.stderr.write(
            "rocke: refusing to commit local-only scratch files:\n"
            + "".join(f"  - {b}\n" for b in bad)
            + "These are local development aids (see rocke .gitignore / CONTRIBUTING.md);\n"
            "unstage them: git reset HEAD <file>\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
