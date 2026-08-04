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
#include "stinkytofu/transforms/asm/LiftAsmRegistersToSSAPass.hpp"

#include <string>
#include <utility>
#include <vector>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {
namespace {

/// How one physical operand participates in allocator SSA.
enum class OperandKind {
    /// Not allocatable: literal, special register, or pseudo register.
    Ignored,
    /// Allocatable full-DWORD VGPR range.
    VgprRange,
    /// Recognised but out of scope; `reason` explains why.
    Unsupported,
};

struct OperandClass {
    OperandKind kind = OperandKind::Ignored;
    size_t units = 0;
    std::string reason;
};

OperandClass classifyOperand(const StinkyRegister& reg) {
    if (!reg.isRegister()) return {OperandKind::Ignored, 0, {}};
    if (reg.isVirtualReg())
        return {OperandKind::Unsupported, 0,
                "unresolved template virtual register; resolve it before lifting"};
    if (isPseudoReg(reg)) return {OperandKind::Ignored, 0, {}};
    if (!isAllocatableReg(reg.reg.type)) return {OperandKind::Ignored, 0, {}};
    if (reg.reg.type != RegType::V)
        return {OperandKind::Unsupported, 0,
                "register class '" + regTypeToString(reg.reg.type) +
                    "' is not lifted yet; only VGPRs are supported"};
    return {OperandKind::VgprRange, reg.reg.num, {}};
}

/// True when any operand selects a True16 half, which needs sub-DWORD units.
bool usesTrue16Halves(const StinkyInstruction& instruction) {
    const auto* modifier = instruction.getModifier<True16Modifiers>();
    if (modifier == nullptr) return false;
    if (modifier->getDst0() != HighBitSel::NONE) return true;
    if (modifier->getDst1() != HighBitSel::NONE) return true;
    for (size_t src = 0; src < modifier->getSrcCount(); ++src) {
        if (modifier->getSrc(src) != HighBitSel::NONE) return true;
    }
    return false;
}

class Lifter {
   public:
    Lifter(const Function& function, const LiftAsmRegistersToSSAOptions& options)
        : function_(function), options_(options) {}

    Expected<CanonicalSSA> run();

   private:
    using Result = Expected<CanonicalSSA>;

    /// Diagnostics are located as "@function[ #instruction[ operand]]: message".
    Result fail(const std::string& location, const std::string& message) const {
        return Result::Error("@" + function_.getName() + location + ": " + message);
    }

    Result fail(const std::string& message) const {
        return fail("", message);
    }

    Result failAt(uint32_t instructionIndex, const std::string& message) const {
        return fail(" #" + std::to_string(instructionIndex), message);
    }

    Result failAtOperand(uint32_t instructionIndex, bool isDestination, size_t operand,
                         const std::string& message) const {
        const std::string role = isDestination ? " dst" : " src";
        return fail(" #" + std::to_string(instructionIndex) + role + std::to_string(operand),
                    message);
    }

    const Function& function_;
    const LiftAsmRegistersToSSAOptions& options_;
};

Expected<CanonicalSSA> Lifter::run() {
    if (function_.empty()) return CanonicalSSA{};
    if (function_.size() != 1) {
        return fail("lifting currently requires a single basic block, but the function has " +
                    std::to_string(function_.size()) + "; multi-block input needs PHI placement");
    }

    const BasicBlock& entry = *function_.begin();
    if (!entry.getPredecessors().empty()) {
        return fail("the entry block has incoming edges; loop-carried values need PHI placement");
    }

    CanonicalSSABuilder builder;

    // Reaching definition of every register unit seen so far.
    RegKeyMap<SSAValueID> reaching;

    uint32_t instructionIndex = 0;
    for (const IRBase& ir : entry) {
        const auto* instruction = dyn_cast<StinkyInstruction>(&ir);
        if (instruction == nullptr) continue;

        const uint32_t index = instructionIndex++;
        if (instruction->getHwInstDesc() == nullptr)
            return failAt(index, "instruction has no hardware descriptor");

        const uint16_t opcode = instruction->getUnifiedOpcode();
        // Labels are block boundaries, not dataflow.
        if (opcode == GFX::LABEL) continue;
        if (opcode == GFX::PHI) {
            return failAt(index,
                          "transient analysis PHIs must be removed before lifting; canonical "
                          "PHIs live in the sidecar, not the instruction stream");
        }
        if (isCall(*instruction)) {
            return failAt(index,
                          "call sites need a calling convention to describe argument, result, "
                          "and clobbered registers");
        }
        if (usesTrue16Halves(*instruction))
            return failAt(index, "True16 half operands need sub-DWORD SSA units");

        SSAInstructionInfo info;

        // Sources are bound before destinations are created, so a
        // read-modify-write operand reads the previous value.
        const std::vector<StinkyRegister>& srcRegs = instruction->getSrcRegs();
        info.sources.resize(srcRegs.size());
        for (size_t operand = 0; operand < srcRegs.size(); ++operand) {
            const OperandClass operandClass = classifyOperand(srcRegs[operand]);
            if (operandClass.kind == OperandKind::Unsupported)
                return failAtOperand(index, /*isDestination=*/false, operand, operandClass.reason);

            for (size_t unit = 0; unit < operandClass.units; ++unit) {
                const RegKey key = toRegKey(srcRegs[operand], static_cast<unsigned>(unit));

                SSAValueID id = kInvalidSSAValueID;
                auto reachingIt = reaching.find(key);
                if (reachingIt != reaching.end()) {
                    id = reachingIt->second;
                } else {
                    if (!options_.allowInferredLiveIns) {
                        return failAtOperand(
                            index, /*isDestination=*/false, operand,
                            "reads " + regKeyToString(key) + " with no reaching definition");
                    }
                    SSAValue liveIn;
                    liveIn.kind = SSAValueKind::LiveIn;
                    liveIn.origin = key;
                    id = builder.addValue(std::move(liveIn));
                    reaching.emplace(key, id);
                }

                info.sources[operand].units.push_back(id);

                SSAUse use;
                use.instruction = instruction;
                use.operand = static_cast<uint32_t>(operand);
                use.unit = static_cast<uint32_t>(unit);
                builder.value(id).uses.push_back(use);
            }
        }

        const std::vector<StinkyRegister>& destRegs = instruction->getDestRegs();
        info.destinations.resize(destRegs.size());
        RegKeySet definedHere;
        for (size_t operand = 0; operand < destRegs.size(); ++operand) {
            const OperandClass operandClass = classifyOperand(destRegs[operand]);
            if (operandClass.kind == OperandKind::Unsupported)
                return failAtOperand(index, /*isDestination=*/true, operand, operandClass.reason);

            for (size_t unit = 0; unit < operandClass.units; ++unit) {
                const RegKey key = toRegKey(destRegs[operand], static_cast<unsigned>(unit));
                if (!definedHere.insert(key).second) {
                    return failAtOperand(
                        index, /*isDestination=*/true, operand,
                        "defines " + regKeyToString(key) + " more than once in one instruction");
                }

                SSAValue defined;
                defined.kind = SSAValueKind::InstructionDef;
                defined.origin = key;
                defined.definingInstruction = instruction;
                defined.definingOperand = static_cast<uint32_t>(operand);
                defined.definingUnit = static_cast<uint32_t>(unit);
                const SSAValueID id = builder.addValue(std::move(defined));

                info.destinations[operand].units.push_back(id);
                reaching[key] = id;
            }
        }

        builder.setInstructionInfo(*instruction, std::move(info));
    }

    CanonicalSSA ssa = builder.take();
    if (options_.verify) {
        const CanonicalSSAVerificationResult verification = verifyCanonicalSSA(function_, ssa);
        if (!verification.ok())
            return fail("canonical SSA verification failed:\n" + verification.toString());
    }
    return ssa;
}

}  // namespace

Expected<CanonicalSSA> liftAsmRegistersToSSA(const Function& function,
                                             const LiftAsmRegistersToSSAOptions& options) {
    return Lifter(function, options).run();
}

}  // namespace stinkytofu
