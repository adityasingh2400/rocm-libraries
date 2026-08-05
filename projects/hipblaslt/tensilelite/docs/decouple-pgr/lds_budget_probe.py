#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LDS-budget probe for the decoupled-PrefetchGlobalRead work.

Two modes:

``config`` mode
    Drive a real ``BenchmarkProblems`` config through TensileLite's solution
    derivation (CPU-only, no GPU) and dump the per-tensor LDS state, then the
    coupled / mixed / single-buffer budget comparison built from it.

        python3 lds_budget_probe.py configs/hero_f8f4_gfx1250.yaml gfx1250

    A config whose coupled variant overflows MaxLDS is dropped during
    derivation, so the probe reports ``0 solution(s)``.  Add
    ``- NoReject: [True]`` to ``ForkParameters``, or probe the ``1LDSBuffer: 1``
    twin, to get the per-tensor footprints out.

``--budget`` mode
    Same comparison from raw numbers, with no toolchain needed at all:

        python3 lds_budget_probe.py --budget 33280 131584 327680

Reuses the characterization suite's CPU-only config harness
(``Tensile/Tests/unit/characterization/_codegen/config_harness.py``) so the
numbers come from production code, not a reimplementation.
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# docs/decouple-pgr -> docs -> tensilelite
_TENSILELITE = os.path.dirname(os.path.dirname(_HERE))
_HARNESS = os.path.join(
    _TENSILELITE, "Tensile", "Tests", "unit", "characterization", "_codegen"
)

# Per-tensor and budget-relevant keys, in the order they are derived.
_STATE_KEYS = [
    "PrefetchGlobalRead", "1LDSBuffer", "DtlPlusLdsBuf",
    "DepthU", "_DepthUA", "_DepthUB", "MacroTile0", "MacroTile1",
    "NumWaves", "WavefrontSize",
    "LdsPadA", "LdsPadB", "LdsBlockSizePerPadA", "LdsBlockSizePerPadB",
    "UnrollMajorLDSA", "UnrollMajorLDSB",
    "ldsNumBytesA", "ldsNumBytesB",
    "LdsNumElementsAlignedA", "LdsNumElementsAlignedB",
    "LdsNumElementsAlignedMXSA", "LdsNumElementsAlignedMXSB",
    "LdsNumElementsAlignedMetadata",
    "LdsOffsetA", "LdsOffsetB", "LdsOffsetA_Blk", "LdsOffsetB_Blk",
    "NumLdsBlk", "StoreSwapAddr", "LdsNumBytes", "MaxLDS",
    "TDMInst", "enableTDMA", "enableTDMB", "LDSSegmentInterleave",
    "DirectToLdsA", "DirectToLdsB", "UseSubtileImpl", "ExpandPointerSwap",
    "LocalWriteUseSgprA", "LocalWriteUseSgprB", "UseCustomMainLoopSchedule",
]


def _kib(n):
    return "%.2f KiB" % (n / 1024.0)


def _pow2_ceil(n):
    return 1 << (n - 1).bit_length() if n > 0 else 0


def report_budget(a, b, max_lds, extra=0):
    """Print the coupled / mixed / single-buffer comparison for aligned A/B sizes.

    ``a``/``b`` are the *aligned* per-tensor footprints (LdsNumElementsAlignedA/B);
    ``extra`` covers MX-scale and metadata blocks that live inside a buffer.
    """
    per_blk = a + b + extra
    rows = [
        ("1LDSBuffer=1   (A:1, B:1)", per_blk),
        ("coupled PGR>=1 (A:2, B:2)", 2 * per_blk),
        ("decoupled      (A:1, B:2)", a + 2 * b + extra),
        ("decoupled      (A:2, B:1)", 2 * a + b + extra),
    ]
    print()
    print("LDS budget (aligned A=%d, B=%d, extra=%d, MaxLDS=%d / %s)"
          % (a, b, extra, max_lds, _kib(max_lds)))
    print("  %-26s %10s %12s   %s" % ("configuration", "bytes", "", "fits"))
    for name, total in rows:
        print("  %-26s %10d %12s   %s"
              % (name, total, "(%s)" % _kib(total), "yes" if total <= max_lds else "NO"))

    # The mixed layout is [A][B blk0][B blk1] with stride == aligned B, which is
    # not a power of two.  The inline-XOR swap needs a power-of-two stride, so
    # show why the mixed case has to use the SGPR/VGPR base-swap path instead.
    print()
    print("  mixed-layout swap form (A:1, B:2):")
    xor_stride = _pow2_ceil(b)
    base_swap = a + 2 * b + extra
    xor_total = a + xor_stride + b + extra
    print("    base swap (StoreSwapAddr), stride=%-8d total %7d (%s)  fits=%s"
          % (b, base_swap, _kib(base_swap), "yes" if base_swap <= max_lds else "NO"))
    print("    inline XOR, stride=pow2(%d)=%-8d total %7d (%s)  fits=%s"
          % (b, xor_stride, xor_total, _kib(xor_total),
             "yes" if xor_total <= max_lds else "NO"))
    if base_swap <= max_lds < xor_total:
        print("    => mixed buffering only fits with the base-swap path.")


def report_state(state):
    print("=" * 78)
    valid = state.get("Valid")
    print("Valid            : %s" % valid)
    if valid is False:
        print("RejectReason     : %s" % state.get("RejectReason"))
    for key in _STATE_KEYS:
        if key not in state:
            continue
        value = state[key]
        suffix = ""
        if isinstance(value, int) and not isinstance(value, bool) \
                and value >= 1024 and "Lds" in key:
            suffix = "   (%s)" % _kib(value)
        print("  %-30s = %s%s" % (key, value, suffix))

    a = state.get("LdsNumElementsAlignedA")
    b = state.get("LdsNumElementsAlignedB")
    if a is None or b is None:
        return
    extra = (state.get("LdsNumElementsAlignedMXSA", 0)
             + state.get("LdsNumElementsAlignedMXSB", 0)
             + state.get("LdsNumElementsAlignedMetadata", 0))
    report_budget(a, b, state.get("MaxLDS", 0), extra)


def probe_config(config, arch):
    sys.path.insert(0, _TENSILELITE)
    sys.path.insert(0, _HARNESS)
    import config_harness  # noqa: E402  (path set up above)

    solutions = config_harness.solutions_from_config(config, arch=arch)
    print("config=%s arch=%s -> %d solution(s)" % (config, arch, len(solutions)))
    if not solutions:
        print("No solution survived derivation. Rejection reasons are printed on "
              "stderr above; add '- NoReject: [True]' to ForkParameters to keep "
              "a rejected solution's state.")
    for solution in solutions:
        report_state(getattr(solution, "_state", solution))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", help="BenchmarkProblems config YAML")
    parser.add_argument("arch", nargs="?", default="gfx1250",
                        help="target gfx architecture (default: gfx1250)")
    parser.add_argument("--budget", nargs=3, type=int, metavar=("A", "B", "MAXLDS"),
                        help="skip derivation; report the budget for aligned A/B sizes")
    parser.add_argument("--extra", type=int, default=0,
                        help="bytes of MX-scale/metadata inside one buffer (--budget mode)")
    args = parser.parse_args()

    if args.budget:
        report_budget(args.budget[0], args.budget[1], args.budget[2], args.extra)
        return 0
    if not args.config:
        parser.error("give a config YAML, or --budget A B MAXLDS")
    probe_config(args.config, args.arch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
