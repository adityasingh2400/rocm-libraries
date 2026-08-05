#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# guard_artifacts.py -- pre-commit guard that blocks build artifacts from being
# staged inside the rocke tree (hard rule: no build*/ , *.a/*.o/*.so, egg-info,
# __pycache__ in git). Receives staged file paths as argv.

import sys
from pathlib import Path

BLOCKED_SUFFIXES = {".a", ".o", ".so", ".hsaco", ".pyc"}
BLOCKED_DIR_PARTS = {"__pycache__", "_deps"}
BLOCKED_DIR_SUFFIXES = (".egg-info",)


def _is_artifact(f: str) -> bool:
    p = Path(f)
    if p.suffix in BLOCKED_SUFFIXES:
        return True
    for part in p.parts:
        if part in BLOCKED_DIR_PARTS:
            return True
        if part.endswith(BLOCKED_DIR_SUFFIXES):
            return True
        if part == "build" or part.startswith("build-"):
            return True
    return False


def main(argv: list[str]) -> int:
    bad = [f for f in argv if _is_artifact(f)]
    if bad:
        sys.stderr.write(
            "rocke: refusing to commit build artifacts:\n"
            + "".join(f"  - {b}\n" for b in bad)
            + "Build out-of-tree (-B /tmp/rocke). Unstage: git reset HEAD <file>\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
