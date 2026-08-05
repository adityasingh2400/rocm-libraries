#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# remind_run_checks.py -- NON-BLOCKING reminder to run the rocke check runner
# before a PR merge. pre-commit/pre-push hooks are not interactive (no reliable
# TTY at commit/push time), so this is a printed reminder, never a prompt and
# never a gate: it ALWAYS exits 0 so it cannot block a commit or push.
#
# It only speaks up when the staged/pushed change actually touches the rocke tree
# (the hook's `files:` filter already scopes this, but the script re-checks its
# argv so it stays quiet on unrelated hunks). Receives changed file paths as argv.

import sys

ROCKE_PREFIX = "dnn-providers/hip-kernel-provider/rocke/"
BANNER = """\
+------------------------------------------------------------------+
|  rocke change detected -- run checks before you open/merge a PR:  |
|                                                                   |
|    python dnn-providers/hip-kernel-provider/rocke/\\               |
|           tools/run_checks.py                                     |
|                                                                   |
|  Scope while iterating:                                           |
|    --steps numeric         only the on-GPU correctness lane       |
|    --steps gate,parity     just the byte-identity checks          |
|    --op fmha_bwd           one operator across all stages         |
|                                                                   |
|  NOT comprehensive: the on-GPU lane only covers THIS machine's    |
|  GPU arch -- run it on each arch you can. (reminder only)         |
+------------------------------------------------------------------+"""


def main(argv: list[str]) -> int:
    if any(f.startswith(ROCKE_PREFIX) or ROCKE_PREFIX in f for f in argv):
        sys.stderr.write(BANNER + "\n")
    return 0  # never block


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
