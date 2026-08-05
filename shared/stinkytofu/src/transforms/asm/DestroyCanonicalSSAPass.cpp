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
#include "stinkytofu/transforms/asm/DestroyCanonicalSSAPass.hpp"

#include <memory>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/CanonicalSSA.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/support/OptimizationRemark.hpp"

#define DEBUG_TYPE "DestroyCanonicalSSAPass"

namespace stinkytofu {
namespace {

constexpr const char* kPassName = "ReplayLegacyColoring";

/// One operand rewrite, staged so nothing is applied until all are valid.
struct OperandRewrite {
    StinkyInstruction* instruction = nullptr;
    bool isDestination = false;
    size_t operand = 0;
    RegType type = RegType::UNKNOWN;
    unsigned baseIndex = 0;
};

class Destroyer {
   public:
    Destroyer(Function& function, const AllocationResult& allocation)
        : function_(function), ssa_(function.getCanonicalSSA()), allocation_(allocation) {}

    SSADestructionResult run() {
        indexInstructions();
        planOperandRewrites();
        checkPhis();
        // Applying only after every check keeps a rejected function exactly as
        // it was, sidecar included.
        if (!result_.ok()) return std::move(result_);

        for (const OperandRewrite& rewrite : rewrites_) apply(rewrite);
        function_.clearCanonicalSSA();
        return std::move(result_);
    }

   private:
    void indexInstructions() {
        uint32_t index = 0;
        for (BasicBlock& bb : function_) {
            for (IRBase& ir : bb) {
                if (const auto* inst = dyn_cast<StinkyInstruction>(&ir))
                    instructionIndex_.emplace(inst, index++);
            }
        }
    }

    std::string locate(const StinkyInstruction* instruction, bool isDestination,
                       size_t operand) const {
        auto it = instructionIndex_.find(instruction);
        const std::string where = it == instructionIndex_.end() ? "<foreign-instruction>"
                                                                : "#" + std::to_string(it->second);
        return "@" + function_.getName() + " " + where + (isDestination ? " dst" : " src") +
               std::to_string(operand);
    }

    void error(const std::string& message) {
        result_.errors.push_back(message);
    }

    /// Validates one operand's units and stages its new base register.
    void planBinding(StinkyInstruction& instruction, bool isDestination, size_t operand,
                     const SSAOperandBinding& binding) {
        if (binding.units.empty()) return;

        const std::string where = locate(&instruction, isDestination, operand);
        RegKey first{RegType::UNKNOWN, 0, RegHalf::NONE};

        for (size_t unit = 0; unit < binding.units.size(); ++unit) {
            const SSAValueID id = binding.units[unit];
            if (!allocation_.isAssigned(id)) {
                error(where + " unit " + std::to_string(unit) + ": %" + std::to_string(id) +
                      " has no physical register");
                return;
            }

            const RegKey physical = allocation_.assignmentOf(id);
            if (physical.half != RegHalf::NONE) {
                error(where + " unit " + std::to_string(unit) + ": %" + std::to_string(id) +
                      " is assigned the sub-DWORD register " + regKeyToString(physical) +
                      ", which cannot be written back to a full-DWORD operand");
                return;
            }

            if (unit == 0) {
                first = physical;
                continue;
            }
            // A range operand must stay one consecutive run in operand order.
            if (physical.type != first.type || physical.idx != first.idx + unit) {
                error(where + ": unit 0 is " + regKeyToString(first) + " but unit " +
                      std::to_string(unit) + " is " + regKeyToString(physical) +
                      "; a range operand must be consecutive in operand order");
                return;
            }
        }

        rewrites_.push_back(
            OperandRewrite{&instruction, isDestination, operand, first.type, first.idx});
    }

    void planOperandRewrites() {
        for (BasicBlock& bb : function_) {
            for (IRBase& ir : bb) {
                auto* instruction = dyn_cast<StinkyInstruction>(&ir);
                if (instruction == nullptr) continue;

                const SSAInstructionInfo* info = ssa_.findInstructionInfo(*instruction);
                if (info == nullptr) continue;

                for (size_t operand = 0; operand < info->sources.size(); ++operand)
                    planBinding(*instruction, /*isDestination=*/false, operand,
                                info->sources[operand]);
                for (size_t operand = 0; operand < info->destinations.size(); ++operand)
                    planBinding(*instruction, /*isDestination=*/true, operand,
                                info->destinations[operand]);
            }
        }
    }

    void checkPhis() {
        for (const SSAPhi& phi : ssa_.phis()) {
            if (!allocation_.isAssigned(phi.result)) {
                error("@" + function_.getName() + " phi#" + std::to_string(phi.id) + ": result %" +
                      std::to_string(phi.result) + " has no physical register");
                continue;
            }

            const RegKey resultPhysical = allocation_.assignmentOf(phi.result);
            for (size_t edge = 0; edge < phi.incoming.size(); ++edge) {
                const SSAValueID incoming = phi.incoming[edge].value;
                if (!allocation_.isAssigned(incoming)) {
                    error("@" + function_.getName() + " phi#" + std::to_string(phi.id) + " edge " +
                          std::to_string(edge) + ": %" + std::to_string(incoming) +
                          " has no physical register");
                    continue;
                }
                const RegKey incomingPhysical = allocation_.assignmentOf(incoming);
                if (incomingPhysical == resultPhysical) continue;

                error("@" + function_.getName() + " phi#" + std::to_string(phi.id) + " edge " +
                      std::to_string(edge) + ": %" + std::to_string(incoming) + " is " +
                      regKeyToString(incomingPhysical) + " but the result is " +
                      regKeyToString(resultPhysical) +
                      "; lowering that needs a copy on the incoming edge, which is not "
                      "implemented yet");
            }
        }
    }

    void apply(const OperandRewrite& rewrite) {
        const std::vector<StinkyRegister>& operands = rewrite.isDestination
                                                          ? rewrite.instruction->getDestRegs()
                                                          : rewrite.instruction->getSrcRegs();
        // Rewriting only the class and base index keeps width, modifiers, and
        // symbolic names exactly as the producer wrote them.
        StinkyRegister updated = operands[rewrite.operand];
        updated.reg.type = rewrite.type;
        updated.reg.idx = rewrite.baseIndex;

        if (rewrite.isDestination)
            rewrite.instruction->setDestReg(rewrite.operand, updated);
        else
            rewrite.instruction->setSrcReg(rewrite.operand, updated);
    }

    Function& function_;
    const CanonicalSSA& ssa_;
    const AllocationResult& allocation_;

    std::unordered_map<const StinkyInstruction*, uint32_t> instructionIndex_;
    std::vector<OperandRewrite> rewrites_;
    SSADestructionResult result_;
};

class ReplayLegacyColoringPassImpl : public Pass {
   public:
    static char ID;

    const char* getName() const override {
        return "Replay Legacy Coloring";
    }

    PassID getPassID() const override {
        return &ReplayLegacyColoringPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        if (!func.hasCanonicalSSA()) {
            emitRemark(passCtx,
                       {OptimizationRemark::Kind::Missed, kPassName, "NoCanonicalSSA",
                        "@" + func.getName() + ": no canonical SSA attached; nothing to lower"});
            return preserveCFGAnalyses();
        }

        const SSADestructionResult result = replayLegacyColoring(func);
        if (!result.ok()) {
            PASS_DEBUG(std::cerr << "ReplayLegacyColoring: " << result.toString() << "\n");
            emitRemark(passCtx, {OptimizationRemark::Kind::Missed, kPassName, "NotLowered",
                                 result.toString()});
            return preserveCFGAnalyses();
        }

        emitRemark(passCtx, {OptimizationRemark::Kind::Passed, kPassName, "ReplayedLegacyColoring",
                             "@" + func.getName() +
                                 ": lowered canonical SSA back to its original registers"});
        return preserveCFGAnalyses();
    }
};

char ReplayLegacyColoringPassImpl::ID = 0;

}  // namespace

std::string SSADestructionResult::toString() const {
    std::ostringstream out;
    for (size_t i = 0; i < errors.size(); ++i) {
        if (i > 0) out << "\n";
        out << errors[i];
    }
    return out.str();
}

SSADestructionResult destroyCanonicalSSA(Function& function, const AllocationResult& allocation) {
    SSADestructionResult result;
    if (!function.hasCanonicalSSA()) {
        result.errors.push_back("@" + function.getName() +
                                ": no canonical SSA is attached, so there is nothing to lower");
        return result;
    }
    return Destroyer(function, allocation).run();
}

SSADestructionResult replayLegacyColoring(Function& function) {
    SSADestructionResult result;
    if (!function.hasCanonicalSSA()) {
        result.errors.push_back("@" + function.getName() +
                                ": no canonical SSA is attached, so there is nothing to lower");
        return result;
    }
    const AllocationResult legacy = createLegacyColoring(function.getCanonicalSSA());
    return destroyCanonicalSSA(function, legacy);
}

std::unique_ptr<Pass> createReplayLegacyColoringPass() {
    return std::make_unique<ReplayLegacyColoringPassImpl>();
}

}  // namespace stinkytofu
