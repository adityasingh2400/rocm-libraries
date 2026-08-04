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

// Teardown for physical-register analysis state. Nothing here computes an
// analysis; these functions only discard one.

#include <cstddef>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Function;

/// What clearTransientRegisterAnalyses() discarded.
struct TransientAnalysisCleanup {
    size_t removedPhis = 0;
    size_t clearedInstructions = 0;
};

/// Erase every `GFX::PHI` pseudo-instruction from \p function.
///
/// These are analysis artifacts: `insertPhiInstructions()` places them to carry
/// reaching definitions across joins, and they are keyed by physical register
/// rather than by value. They are never emitted.
///
/// This does not repair def-use edges pointing at the removed PHIs, matching
/// the existing rebuild-everything callers. Use
/// clearTransientRegisterAnalyses() when the chains are not being rebuilt.
STINKYTOFU_EXPORT size_t removeAnalysisPhis(Function& function);

/// Clear the def-use edges recorded on every instruction of \p function.
/// Returns the number of instructions visited.
STINKYTOFU_EXPORT size_t clearDefUseChains(Function& function);

/// Discard all transient physical-register analysis state, leaving only the
/// instruction stream and the CFG.
///
/// This is the cleanup done at the canonical SSA boundary: those analyses are
/// keyed by physical register, so they cannot describe SSA values, and leaving
/// them attached invites a consumer to trust stale data.
///
/// Chains are cleared before the PHIs are erased, so no instruction is left
/// pointing at freed memory.
STINKYTOFU_EXPORT TransientAnalysisCleanup clearTransientRegisterAnalyses(Function& function);

}  // namespace stinkytofu
