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
#include <string>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/ir/asm/CanonicalSSAAllocation.hpp"

namespace stinkytofu {
class Function;
class Pass;

/// Everything that stopped SSA destruction, in deterministic order.
struct STINKYTOFU_EXPORT SSADestructionResult {
    std::vector<std::string> errors;

    bool ok() const {
        return errors.empty();
    }

    std::string toString() const;
};

/// Rewrite \p function's physical operands from \p allocation and detach the
/// canonical SSA sidecar.
///
/// This is the single lowering path shared by every allocation result, so
/// legacy replay and a real allocator differ only in the colouring they are
/// given, never in how it is applied.
///
/// The rewrite is atomic: every operand is validated before any is modified, so
/// a rejected function keeps its original registers and its sidecar.
///
/// A PHI whose inputs and result do not all land on the same register needs a
/// copy on the incoming edge. Copy insertion, parallel-copy sequencing, and
/// critical-edge splitting are not implemented, so that case is reported rather
/// than mis-lowered. Legacy replay never hits it: every version of a register
/// colours back to that same register.
STINKYTOFU_EXPORT SSADestructionResult destroyCanonicalSSA(Function& function,
                                                           const AllocationResult& allocation);

/// Rewrite \p function using its own origins, undoing the lift exactly.
STINKYTOFU_EXPORT SSADestructionResult replayLegacyColoring(Function& function);

/// Creates a pass that lowers the attached sidecar back to physical registers
/// using legacy colouring.
///
/// Lifting followed by this pass must be an identity transform on the physical
/// program. That round trip is the gate the design puts in front of enabling
/// any real allocation: until it holds, a difference in emitted code cannot be
/// attributed to allocation policy.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createReplayLegacyColoringPass();

}  // namespace stinkytofu
