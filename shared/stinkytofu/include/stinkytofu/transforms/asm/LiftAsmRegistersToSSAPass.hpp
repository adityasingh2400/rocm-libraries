/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/ir/asm/CanonicalSSA.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"

namespace stinkytofu {
class Function;
class Pass;
struct DominanceInfo;

struct LiftAsmRegistersToSSAOptions {
    /// Verify the constructed graph before handing it back.
    bool verify = true;

    /// Treat a read with no reaching definition as a function live-in.
    ///
    /// Physical input does not say which registers are genuine kernel inputs,
    /// so this conservative default preserves the meaning of the original
    /// program. Set false to require that every read is defined, which is the
    /// strict mode used once entry metadata is available.
    bool allowInferredLiveIns = true;
};

/// Build canonical SSA from the physical register operands of \p function.
///
/// Physical registers are treated as mutable variables: every reaching
/// definition of a register unit becomes its own SSA value, and each value
/// keeps its originating RegKey for legacy replay.
///
/// Values that merge at a control-flow join become canonical PHIs in the
/// sidecar, placed at iterated dominance frontiers and pruned by liveness so
/// no dead PHI is created. Reducible and irreducible CFGs are both supported.
///
/// Current scope is deliberately narrow. Operands must be full-DWORD VGPRs,
/// and every block must be reachable from the entry. Literals, special
/// registers such as EXEC or SCC, and pseudo registers are ignored rather than
/// lifted. Anything else - other register classes, unresolved template virtual
/// registers, True16 halves, calls, or leftover analysis PHIs - is reported as
/// an error instead of being silently mishandled.
///
/// Construction is atomic: on error nothing is returned, so a caller can never
/// attach a partially built graph.
STINKYTOFU_EXPORT Expected<CanonicalSSA> liftAsmRegistersToSSA(
    Function& function, const LiftAsmRegistersToSSAOptions& options = {});

/// As above, reusing dominance information the caller already computed.
STINKYTOFU_EXPORT Expected<CanonicalSSA> liftAsmRegistersToSSA(
    Function& function, const DominanceInfo& dominance,
    const LiftAsmRegistersToSSAOptions& options = {});

/// Creates a pass that lifts a function's physical registers to canonical SSA
/// and attaches the result to the function.
///
/// The pass is function-wide: PHI placement and renaming need every block, so
/// it refuses to run at all when basic-block filtering excludes any block.
///
/// It never mutates blocks, instructions, or register operands. Any previously
/// attached sidecar is detached first, so an unsupported function is left with
/// no canonical SSA rather than a graph describing an earlier state. Consumers
/// must therefore check Function::hasCanonicalSSA() and fall back when it is
/// absent; unsupported input is reported as a missed-optimization remark, not
/// a hard error.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createLiftAsmRegistersToSSAPass(
    const LiftAsmRegistersToSSAOptions& options = {});

}  // namespace stinkytofu
