#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# check_commit_msg.py -- Conventional-Commits checker for rocke commits, matching
# the style already used in history (feat(rocke): / test(rocke): / fix(rocke): /
# perf(attention): / feat(rocke/attention): ). Intended as a `commit-msg` hook:
#   pre-commit sample: stage: commit-msg, entry: python .../check_commit_msg.py
# Receives the path to the commit-message file as argv[1].
#
# Enforced format:
#   <type>(<scope>): <subject>
# where type ∈ the set below and scope is required for rocke work
# (rocke | hip-kernel-provider | rocke/<area> | attention | ...).
# Merge/revert/fixup commits are exempt.

import re
import sys

TYPES = {"feat", "fix", "perf", "refactor", "test", "docs", "chore", "ci", "build", "style"}
HEADER_RE = re.compile(r"^(?P<type>[a-z]+)(?:\((?P<scope>[a-z0-9/_-]+)\))?(?P<bang>!)?: .+")
EXEMPT_RE = re.compile(r"^(Merge |Revert |fixup!|squash!)")


def main(argv: list[str]) -> int:
    if not argv:
        sys.stderr.write("check_commit_msg: no commit-message file given\n")
        return 2
    msg = open(argv[0], encoding="utf-8").read()
    header = next((ln for ln in msg.splitlines() if ln.strip()), "")

    if EXEMPT_RE.match(header):
        return 0

    m = HEADER_RE.match(header)
    if not m:
        sys.stderr.write(
            "rocke commit message must be Conventional Commits:\n"
            "  <type>(<scope>): <subject>\n"
            f"got: {header!r}\n"
            f"types: {', '.join(sorted(TYPES))}\n"
            "example: feat(rocke): add gfx950 mask-phase-split 2D lever\n"
        )
        return 1
    if m.group("type") not in TYPES:
        sys.stderr.write(
            f"unknown commit type {m.group('type')!r}; use one of: "
            f"{', '.join(sorted(TYPES))}\n"
        )
        return 1
    if not m.group("scope"):
        sys.stderr.write(
            "rocke commits require a scope, e.g. (rocke), (hip-kernel-provider), "
            "or (rocke/attention).\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
