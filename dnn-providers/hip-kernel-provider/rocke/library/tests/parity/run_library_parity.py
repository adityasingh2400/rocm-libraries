#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_library_parity.py -- byte-identity runner for the library/attention
# .py/.c emitter pairs.
#
# WHY THIS EXISTS: the platform byte-identity gate (tools/check_byte_identity.py)
# and the platform run_parity.py cover ONLY the platform micro-kernels and the
# platform instance families. NEITHER covers the library/attention families
# (attention_unified, fmha_*, gfx9xx_attention_tiled_*, sage/sparse). Those
# .py/.c pairs shipped with no automated runner, so the SDPA/MHA product's
# dual-engine byte-identity was never gated in CI. This driver closes that gap.
#
# It mirrors platform/tests/instances/parity/run_parity.py: build (or reuse) the
# C++ engine archive librocke_core.a, compile each family's <family>_emit.c
# against it + the flat rocke/*.h ABI, then for each config index sha256-compare
# the C-emitted .ll against the Python reference emit (<family>_emit.py). It
# probes config indices from 0 upward until BOTH emitters reject an index (a
# nonzero exit / no stdout), so it needs no hardcoded per-family config counts.
#
# comgr-compile-only: no GPU required.
#
# Usage:
#   python run_library_parity.py [--flavor llvm20|llvm22] [--archive PATH]
#                                [--build-root DIR] [--only SUBSTR]

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # library/tests/parity
ROCKE = HERE.parents[2]                                 # parity -> tests -> library -> rocke
PLATFORM = ROCKE / "platform"
INCLUDE = PLATFORM / "cpp" / "include"                  # flat rocke/*.h ABI
PLATFORM_PYROOT = PLATFORM / "python"
# The library emitters do `from _emit_common import run_emit`; that shared driver
# ships only under the platform parity dir, so it must be on PYTHONPATH.
EMIT_COMMON_DIR = PLATFORM / "tests" / "instances" / "parity"
LIBROOT = ROCKE / "library"                             # exposes kernels/builders/dispatch


def _cxx() -> str:
    for c in ("amdclang++", "clang++", "c++", "g++"):
        p = shutil.which(c)
        if p:
            return p
    raise SystemExit("no C++ compiler (amdclang++/clang++/c++/g++) found on PATH")


def _ensure_archive(build_root: Path) -> Path:
    archive = build_root / "librocke_core.a"
    if archive.exists():
        return archive
    subprocess.run(
        ["cmake", "-S", str(PLATFORM), "-B", str(build_root),
         "-DCMAKE_BUILD_TYPE=Release"],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_root), "--target", "rocke_core",
         "-j", str(os.cpu_count() or 1)],
        check=True,
    )
    return archive


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _families(only: str | None) -> list[str]:
    fams = sorted(p.name[: -len("_emit.py")] for p in HERE.glob("*_emit.py"))
    # Keep only pairs that have BOTH a .py and a .c twin.
    fams = [f for f in fams if (HERE / f"{f}_emit.c").exists()]
    if only:
        fams = [f for f in fams if only in f]
    return fams


def _run_py(family: str, idx: int, env: dict) -> bytes:
    r = subprocess.run(
        [sys.executable, str(HERE / f"{family}_emit.py"), str(idx), "ll"],
        capture_output=True, env=env,
    )
    return r.stdout if r.returncode == 0 else b""


def _run_c(binexe: Path, idx: int) -> bytes:
    r = subprocess.run([str(binexe), str(idx), "ll"], capture_output=True)
    return r.stdout if r.returncode == 0 else b""


def main() -> int:
    ap = argparse.ArgumentParser(description="library/attention C-vs-Python .ll parity")
    ap.add_argument("--archive", default="")
    ap.add_argument("--build-root", default=str(Path(tempfile.gettempdir()) / "rocke_verify"))
    ap.add_argument("--flavor", default=os.environ.get("ROCKE_LLVM_FLAVOR", ""),
                    help="llvm20|llvm22; sets ROCKE_LLVM_FLAVOR for both engines")
    ap.add_argument("--only", default="", help="restrict to families containing SUBSTR")
    ap.add_argument("--max-idx", type=int, default=256, help="config-index probe ceiling")
    args = ap.parse_args()

    archive = Path(args.archive) if args.archive else _ensure_archive(Path(args.build_root))
    if not archive.exists():
        print(f"FATAL: engine archive not found: {archive}", file=sys.stderr)
        return 1

    out = Path(tempfile.gettempdir()) / "rocke_library_parity"
    out.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    # Both engines must lower under the same flavor; the .py side reads it from
    # the env, the .c side is compiled flavor-agnostic and honors it at runtime.
    if args.flavor:
        env["ROCKE_LLVM_FLAVOR"] = args.flavor
    # library emitters import kernels/builders/dispatch (library) AND rocke
    # (platform) AND _emit_common (platform parity dir).
    pyroots = os.pathsep.join([str(LIBROOT), str(PLATFORM_PYROOT), str(EMIT_COMMON_DIR)])
    env["PYTHONPATH"] = pyroots + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    families = _families(args.only)
    if not families:
        print("no library parity families found", file=sys.stderr)
        return 1

    total = fails = checked = 0
    for fam in families:
        binexe = out / (f"{fam}.exe" if os.name == "nt" else f"{fam}")
        cc = subprocess.run(
            [_cxx(), "-std=c++20", "-I", str(INCLUDE),
             str(HERE / f"{fam}_emit.c"), str(archive), "-lm", "-o", str(binexe)],
            capture_output=True, text=True,
        )
        if cc.returncode != 0:
            print(f"FAIL  {fam}: C emitter compile failed:\n{cc.stderr[-1500:]}", file=sys.stderr)
            fails += 1
            continue

        fam_checked = 0
        for idx in range(args.max_idx):
            c_ll = _run_c(binexe, idx)
            py_ll = _run_py(fam, idx, env)
            if not c_ll and not py_ll:
                break  # both engines reject this index -> end of this family's configs
            total += 1
            fam_checked += 1
            cs, ps = _sha(c_ll), _sha(py_ll)
            if cs == ps and c_ll:
                continue
            print(f"FAIL  {fam}[{idx}]  C={cs} PY={ps}")
            fails += 1
        checked += fam_checked
        print(f"{'ok  ' if fam_checked else 'SKIP'} {fam}: {fam_checked} configs")

    print(f">> library parity: {total} configs across {len(families)} families, {fails} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
