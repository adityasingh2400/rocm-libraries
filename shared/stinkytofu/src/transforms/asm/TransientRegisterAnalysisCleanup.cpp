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
#include "stinkytofu/transforms/asm/TransientRegisterAnalysisCleanup.hpp"

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/DefUseChainUpdater.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {

size_t removeAnalysisPhis(Function& function) {
    size_t removed = 0;
    for (BasicBlock& bb : function) {
        for (auto it = bb.begin(); it != bb.end();) {
            auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (inst != nullptr && inst->getUnifiedOpcode() == GFX::PHI) {
                it = bb.eraseIR(it);
                ++removed;
            } else {
                ++it;
            }
        }
    }
    return removed;
}

size_t clearDefUseChains(Function& function) {
    size_t cleared = 0;
    for (BasicBlock& bb : function) {
        for (IRBase& ir : bb) {
            if (auto* inst = dyn_cast<StinkyInstruction>(&ir)) {
                DefUseChainUpdater::clearChains(inst);
                ++cleared;
            }
        }
    }
    return cleared;
}

TransientAnalysisCleanup clearTransientRegisterAnalyses(Function& function) {
    TransientAnalysisCleanup cleanup;
    // Chains first: erasing a PHI while other instructions still list it as a
    // source would leave them pointing at freed memory.
    cleanup.clearedInstructions = clearDefUseChains(function);
    cleanup.removedPhis = removeAnalysisPhis(function);
    return cleanup;
}

}  // namespace stinkytofu
