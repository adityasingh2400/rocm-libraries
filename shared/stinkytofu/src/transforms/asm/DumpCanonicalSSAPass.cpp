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
#include "stinkytofu/transforms/asm/DumpCanonicalSSAPass.hpp"

#include <fstream>
#include <iostream>
#include <ostream>
#include <utility>

#include "stinkytofu/analysis/controlflow/DominanceAnalysis.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/CanonicalSSA.hpp"

#define DEBUG_TYPE "DumpCanonicalSSAPass"

namespace stinkytofu {
namespace {

class DumpCanonicalSSAPassImpl : public Pass {
   public:
    static char ID;

    explicit DumpCanonicalSSAPassImpl(DumpCanonicalSSAConfig config) : config_(std::move(config)) {}

    const char* getName() const override {
        return "Dump Canonical SSA";
    }

    PassID getPassID() const override {
        return &DumpCanonicalSSAPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& /*passCtx*/, AnalysisManager& AM) override {
        if (!func.hasCanonicalSSA() && config_.requireCanonicalSSA) {
            std::cerr << "DumpCanonicalSSAPass: @" << func.getName()
                      << " has no canonical SSA attached; run LiftAsmRegistersToSSAPass first\n";
            return PreservedAnalyses::all();
        }

        std::ofstream file;
        if (!config_.outputPath.empty()) {
            file.open(config_.outputPath, std::ios::out | std::ios::trunc);
            if (!file) {
                std::cerr << "DumpCanonicalSSAPass: cannot open '" << config_.outputPath << "'\n";
                return PreservedAnalyses::all();
            }
        }
        std::ostream& out = config_.outputPath.empty() ? std::cout : file;

        if (func.hasCanonicalSSA()) {
            // Verifying first keeps a dump from being mistaken for evidence that
            // the graph is well formed.
            const DominanceInfo& dominance = AM.getResult<DominanceAnalysis>(func);
            const CanonicalSSAVerificationResult verification =
                verifyCanonicalSSA(func, func.getCanonicalSSA(), dominance);
            if (!verification.ok()) {
                out << "// canonical SSA verification failed:\n";
                for (const std::string& error : verification.errors)
                    out << "//   " << error << "\n";
            }
        }

        CanonicalSSAPrinter(out, config_.printerOptions).print(func);
        return PreservedAnalyses::all();
    }

   private:
    DumpCanonicalSSAConfig config_;
};

char DumpCanonicalSSAPassImpl::ID = 0;

}  // namespace

std::unique_ptr<Pass> createDumpCanonicalSSAPass(DumpCanonicalSSAConfig config) {
    return std::make_unique<DumpCanonicalSSAPassImpl>(std::move(config));
}

}  // namespace stinkytofu
