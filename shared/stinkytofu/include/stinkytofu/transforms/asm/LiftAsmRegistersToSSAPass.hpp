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

#include "stinkytofu/Export.hpp"
#include "stinkytofu/ir/asm/CanonicalSSA.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"

namespace stinkytofu {
class Function;

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
/// Current scope is deliberately narrow. Supported input is a single basic
/// block with no incoming edges and full-DWORD VGPR operands. Literals,
/// special registers such as EXEC or SCC, and pseudo registers are ignored
/// rather than lifted. Anything else - other register classes, unresolved
/// template virtual registers, True16 halves, calls, or leftover analysis
/// PHIs - is reported as an error instead of being silently mishandled.
///
/// Construction is atomic: on error nothing is returned, so a caller can never
/// attach a partially built graph.
STINKYTOFU_EXPORT Expected<CanonicalSSA> liftAsmRegistersToSSA(
    const Function& function, const LiftAsmRegistersToSSAOptions& options = {});

}  // namespace stinkytofu
