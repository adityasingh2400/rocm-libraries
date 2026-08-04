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
#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/CanonicalSSA.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/serialization/asm/CanonicalSSAPrinter.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu/transforms/asm/LiftAsmRegistersToSSAPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

constexpr GfxArchID kArch = GfxArchID::Gfx1250;

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

class LiftAsmRegistersToSSATest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kArch);
        entry = func->createBasicBlock("entry");
    }

    /// Lifts and requires success, returning the graph.
    CanonicalSSA lift(const LiftAsmRegistersToSSAOptions& options = {}) {
        Expected<CanonicalSSA> result = liftAsmRegistersToSSA(*func, options);
        EXPECT_TRUE(result.hasValue()) << (result.hasValue() ? "" : result.getError());
        if (!result.hasValue()) return CanonicalSSA{};
        return std::move(*result);
    }

    /// Lifts and requires failure, returning the diagnostic.
    std::string liftError(const LiftAsmRegistersToSSAOptions& options = {}) {
        Expected<CanonicalSSA> result = liftAsmRegistersToSSA(*func, options);
        EXPECT_TRUE(result.hasError());
        return result.hasError() ? result.getError() : std::string{};
    }

    std::unique_ptr<Function> func;
    BasicBlock* entry = nullptr;
};

}  // namespace

TEST_F(LiftAsmRegistersToSSATest, EmptyFunctionYieldsEmptyGraph) {
    Function empty("empty");
    Expected<CanonicalSSA> result = liftAsmRegistersToSSA(empty);
    ASSERT_TRUE(result.hasValue()) << result.getError();
    EXPECT_TRUE(result->empty());
}

TEST_F(LiftAsmRegistersToSSATest, EmptyBlockYieldsEmptyGraph) {
    EXPECT_TRUE(lift().empty());
}

TEST_F(LiftAsmRegistersToSSATest, StraightLineProducesVerifiedSSA) {
    createVAddInBlock(entry, kArch, /*dest=*/2, /*src0=*/0, /*src1=*/1);

    const CanonicalSSA ssa = lift();
    EXPECT_EQ(ssa.valueCount(), 3u);
    EXPECT_EQ(ssa.value(1).kind, SSAValueKind::LiveIn);
    EXPECT_EQ(ssa.value(1).origin.idx, 0u);
    EXPECT_EQ(ssa.value(2).kind, SSAValueKind::LiveIn);
    EXPECT_EQ(ssa.value(2).origin.idx, 1u);
    EXPECT_EQ(ssa.value(3).kind, SSAValueKind::InstructionDef);
    EXPECT_EQ(ssa.value(3).origin.idx, 2u);
    EXPECT_TRUE(verifyCanonicalSSA(*func, ssa).ok());
}

TEST_F(LiftAsmRegistersToSSATest, LiftedDumpIsExact) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    createVAddInBlock(entry, kArch, 3, 2, 0);

    const CanonicalSSA ssa = lift();
    EXPECT_EQ(canonicalSSAToString(*func, ssa),
              "ssa.func @kernel {\n"
              "  initial_values:\n"
              "    %1:v = livein { origin = v0 }\n"
              "    %2:v = livein { origin = v1 }\n"
              "  ^entry:\n"
              "    %3:v = \"st.v_add_f32\"(src0 = [%1:v], src1 = [%2:v]) "
              "{ inst = #0, origin = [v2] }\n"
              "      // physical: v2 = \"st.v_add_f32\"(v0, v1)\n"
              "    %4:v = \"st.v_add_f32\"(src0 = [%3:v], src1 = [%1:v]) "
              "{ inst = #1, origin = [v3] }\n"
              "      // physical: v3 = \"st.v_add_f32\"(v2, v0)\n"
              "}\n");
}

TEST_F(LiftAsmRegistersToSSATest, RepeatedDefinitionsOfOneRegisterBecomeDistinctValues) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    createVAddInBlock(entry, kArch, 2, 0, 1);
    createVAddInBlock(entry, kArch, 3, 2, 2);

    const CanonicalSSA ssa = lift();
    // v0, v1 live-ins, then three definitions; the last reads the second v2.
    ASSERT_EQ(ssa.valueCount(), 5u);
    EXPECT_EQ(ssa.value(3).origin.idx, 2u);
    EXPECT_EQ(ssa.value(4).origin.idx, 2u);
    EXPECT_NE(ssa.value(3).id, ssa.value(4).id);

    const std::string text = canonicalSSAToString(*func, ssa);
    EXPECT_TRUE(contains(text, "%5:v = \"st.v_add_f32\"(src0 = [%4:v], src1 = [%4:v])")) << text;
}

TEST_F(LiftAsmRegistersToSSATest, OneValueUsedTwiceRecordsTwoUses) {
    createVAddInBlock(entry, kArch, 2, 0, 0);

    const CanonicalSSA ssa = lift();
    ASSERT_EQ(ssa.valueCount(), 2u);
    const SSAValue& liveIn = ssa.value(1);
    ASSERT_EQ(liveIn.uses.size(), 2u);
    EXPECT_EQ(liveIn.uses[0].operand, 0u);
    EXPECT_EQ(liveIn.uses[1].operand, 1u);
}

TEST_F(LiftAsmRegistersToSSATest, OneLiveInIsSharedByEveryReadOfThatUnit) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    createVAddInBlock(entry, kArch, 3, 0, 1);

    const CanonicalSSA ssa = lift();
    EXPECT_EQ(ssa.value(1).uses.size(), 2u);
    EXPECT_EQ(ssa.value(2).uses.size(), 2u);
    // Two live-ins plus two definitions, not four live-ins.
    EXPECT_EQ(ssa.valueCount(), 4u);
}

TEST_F(LiftAsmRegistersToSSATest, ReadModifyWriteReadsTheOldValueAndDefinesANewOne) {
    // v2 = v_add_f32 v2, v1 reads the incoming v2 and defines a new one.
    createVAddInBlock(entry, kArch, /*dest=*/2, /*src0=*/2, /*src1=*/1);

    const CanonicalSSA ssa = lift();
    ASSERT_EQ(ssa.valueCount(), 3u);
    const SSAValueID incoming = 1;
    EXPECT_EQ(ssa.value(incoming).kind, SSAValueKind::LiveIn);
    EXPECT_EQ(ssa.value(incoming).origin.idx, 2u);

    const SSAValueID defined = 3;
    EXPECT_EQ(ssa.value(defined).kind, SSAValueKind::InstructionDef);
    EXPECT_EQ(ssa.value(defined).origin.idx, 2u);
    EXPECT_TRUE(ssa.value(defined).uses.empty());
    EXPECT_TRUE(verifyCanonicalSSA(*func, ssa).ok());
}

TEST_F(LiftAsmRegistersToSSATest, MultiDwordRangeBindsOneValuePerDword) {
    createDsReadB128InBlock(entry, kArch, /*dest=*/4, /*addr=*/0);

    const CanonicalSSA ssa = lift();
    ASSERT_EQ(ssa.valueCount(), 5u);
    for (unsigned unit = 0; unit < 4; ++unit) {
        const SSAValue& value = ssa.value(2 + unit);
        EXPECT_EQ(value.kind, SSAValueKind::InstructionDef);
        EXPECT_EQ(value.origin.idx, 4u + unit);
        EXPECT_EQ(value.definingUnit, unit);
    }
}

TEST_F(LiftAsmRegistersToSSATest, PartialUseOfAWideDefinitionKeepsUnitIdentity) {
    createDsReadB128InBlock(entry, kArch, /*dest=*/4, /*addr=*/0);
    createDSWriteInBlock(entry, kArch, /*addr=*/0, /*data=*/4);

    const CanonicalSSA ssa = lift();
    const std::string text = canonicalSSAToString(*func, ssa);
    EXPECT_TRUE(contains(text, "%2:v, %3:v, %4:v, %5:v = \"st.ds_load_b128\"")) << text;
    EXPECT_TRUE(contains(text, "\"st.ds_store_b64\"(src0 = [%1:v], src1 = [%2:v, %3:v])")) << text;
    // v6 and v7 are defined but never read, so they gain no uses.
    EXPECT_TRUE(ssa.value(4).uses.empty());
    EXPECT_TRUE(ssa.value(5).uses.empty());
}

TEST_F(LiftAsmRegistersToSSATest, LiteralOperandsBindNothing) {
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kArch));
    mov->addDestReg(StinkyRegister("v", 2, 1));
    mov->addSrcReg(StinkyRegister(7));

    const CanonicalSSA ssa = lift();
    ASSERT_EQ(ssa.valueCount(), 1u);
    const SSAInstructionInfo* info = ssa.findInstructionInfo(*mov);
    ASSERT_NE(info, nullptr);
    ASSERT_EQ(info->sources.size(), 1u);
    EXPECT_TRUE(info->sources[0].units.empty());
    EXPECT_TRUE(verifyCanonicalSSA(*func, ssa).ok());
}

TEST_F(LiftAsmRegistersToSSATest, SpecialRegistersAreNotLifted) {
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* cmp = builder.create(getMCIDByUOp(GFX::v_cmp_eq_u32, kArch));
    cmp->addDestReg(StinkyRegister::getSCCRegister());
    cmp->addSrcReg(StinkyRegister("v", 0, 1));
    cmp->addSrcReg(StinkyRegister("v", 1, 1));

    const CanonicalSSA ssa = lift();
    const SSAInstructionInfo* info = ssa.findInstructionInfo(*cmp);
    ASSERT_NE(info, nullptr);
    ASSERT_EQ(info->destinations.size(), 1u);
    EXPECT_TRUE(info->destinations[0].units.empty());
    EXPECT_EQ(ssa.valueCount(), 2u);
}

TEST_F(LiftAsmRegistersToSSATest, InstructionsWithoutAllocatableOperandsAreStillBound) {
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* nop = builder.create(getMCIDByUOp(GFX::s_nop, kArch));

    const CanonicalSSA ssa = lift();
    EXPECT_NE(ssa.findInstructionInfo(*nop), nullptr);
    EXPECT_FALSE(contains(canonicalSSAToString(*func, ssa), "unmapped"));
}

TEST_F(LiftAsmRegistersToSSATest, LabelsAreSkippedButStillConsumeAnIndex) {
    AsmIRBuilder builder(*entry, kArch);
    builder.createLabel("top");
    createVAddInBlock(entry, kArch, 2, 0, 1);

    const CanonicalSSA ssa = lift();
    EXPECT_TRUE(contains(canonicalSSAToString(*func, ssa), "{ inst = #1, origin = [v2] }"));
}

TEST_F(LiftAsmRegistersToSSATest, StrictModeRejectsInferredLiveIns) {
    createVAddInBlock(entry, kArch, 2, 0, 1);

    LiftAsmRegistersToSSAOptions options;
    options.allowInferredLiveIns = false;
    const std::string error = liftError(options);
    EXPECT_TRUE(contains(error, "@kernel #0 src0: reads v0 with no reaching definition")) << error;
}

TEST_F(LiftAsmRegistersToSSATest, StrictModeAcceptsFullyDefinedCode) {
    // Every read is defined earlier in the block, so no live-in is needed.
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kArch));
    mov->addDestReg(StinkyRegister("v", 0, 1));
    mov->addSrcReg(StinkyRegister(1));
    createVAddInBlock(entry, kArch, /*dest=*/2, /*src0=*/0, /*src1=*/0);

    LiftAsmRegistersToSSAOptions options;
    options.allowInferredLiveIns = false;
    Expected<CanonicalSSA> result = liftAsmRegistersToSSA(*func, options);
    ASSERT_TRUE(result.hasValue()) << result.getError();

    EXPECT_EQ(result->valueCount(), 2u);
    for (const SSAValue& value : result->values())
        EXPECT_EQ(value.kind, SSAValueKind::InstructionDef);
}

TEST_F(LiftAsmRegistersToSSATest, RejectsMultipleBasicBlocks) {
    func->createBasicBlock("second");
    const std::string error = liftError();
    EXPECT_TRUE(contains(error, "requires a single basic block")) << error;
}

TEST_F(LiftAsmRegistersToSSATest, RejectsBlockWithIncomingEdges) {
    func->addEdge(entry, entry);
    const std::string error = liftError();
    EXPECT_TRUE(contains(error, "incoming edges")) << error;
}

TEST_F(LiftAsmRegistersToSSATest, RejectsTemplateVirtualRegisters) {
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kArch));
    mov->addDestReg(StinkyRegister::Virtual(0));
    mov->addSrcReg(StinkyRegister("v", 1, 1));

    const std::string error = liftError();
    EXPECT_TRUE(contains(error, "unresolved template virtual register")) << error;
}

TEST_F(LiftAsmRegistersToSSATest, RejectsSgprOperands) {
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kArch));
    mov->addDestReg(StinkyRegister("v", 0, 1));
    mov->addSrcReg(StinkyRegister("s", 4, 1));

    const std::string error = liftError();
    EXPECT_TRUE(contains(error, "register class 's' is not lifted yet")) << error;
}

TEST_F(LiftAsmRegistersToSSATest, RejectsAnalysisPhis) {
    AsmIRBuilder builder(*entry, kArch);
    builder.createPhi(RegType::V, 2);

    const std::string error = liftError();
    EXPECT_TRUE(contains(error, "transient analysis PHIs must be removed")) << error;
}

TEST_F(LiftAsmRegistersToSSATest, RejectsTrue16HalfOperands) {
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kArch));
    mov->addDestReg(StinkyRegister("v", 0, 1));
    mov->addSrcReg(StinkyRegister("v", 1, 1));
    mov->addModifier<True16Modifiers>(
        True16Modifiers(HighBitSel::HIGH, HighBitSel::NONE, {HighBitSel::NONE}));

    const std::string error = liftError();
    EXPECT_TRUE(contains(error, "True16 half operands")) << error;
}

TEST_F(LiftAsmRegistersToSSATest, RejectsOneInstructionDefiningAUnitTwice) {
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* instruction = builder.create(getMCIDByUOp(GFX::v_add_f32, kArch));
    instruction->addDestReg(StinkyRegister("v", 2, 1));
    instruction->addDestReg(StinkyRegister("v", 2, 1));
    instruction->addSrcReg(StinkyRegister("v", 0, 1));

    const std::string error = liftError();
    EXPECT_TRUE(contains(error, "defines v2 more than once")) << error;
}

TEST_F(LiftAsmRegistersToSSATest, DiagnosticsNameTheFunctionAndInstruction) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kArch));
    mov->addDestReg(StinkyRegister("v", 3, 1));
    mov->addSrcReg(StinkyRegister("s", 4, 1));

    const std::string error = liftError();
    EXPECT_EQ(error,
              "@kernel #1 src0: register class 's' is not lifted yet; only VGPRs are supported");
}

TEST_F(LiftAsmRegistersToSSATest, RepeatedLiftsProduceIdenticalGraphs) {
    createDsReadB128InBlock(entry, kArch, 4, 0);
    createDSWriteInBlock(entry, kArch, 0, 4);
    createVAddInBlock(entry, kArch, 8, 4, 5);

    CanonicalSSAPrinterOptions options;
    options.printUses = true;

    const CanonicalSSA first = lift();
    const CanonicalSSA second = lift();
    EXPECT_EQ(canonicalSSAToString(*func, first, options),
              canonicalSSAToString(*func, second, options));
}

TEST_F(LiftAsmRegistersToSSATest, ResultCanBeAttachedToTheFunction) {
    createVAddInBlock(entry, kArch, 2, 0, 1);

    Expected<CanonicalSSA> result = liftAsmRegistersToSSA(*func);
    ASSERT_TRUE(result.hasValue()) << result.getError();
    func->setCanonicalSSA(std::make_unique<CanonicalSSA>(std::move(*result)));

    ASSERT_TRUE(func->hasCanonicalSSA());
    EXPECT_TRUE(verifyCanonicalSSA(*func).ok());
}

TEST_F(LiftAsmRegistersToSSATest, FailureLeavesNoGraphBehind) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    func->createBasicBlock("second");

    Expected<CanonicalSSA> result = liftAsmRegistersToSSA(*func);
    ASSERT_TRUE(result.hasError());
    EXPECT_FALSE(func->hasCanonicalSSA());
}

// ---------------------------------------------------------------------------
// Pass wrapper
// ---------------------------------------------------------------------------

namespace {

class LiftAsmRegistersToSSAPassTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kArch);
        entry = func->createBasicBlock("entry");
        registerAllAnalyses(am);
    }

    void runPass(const LiftAsmRegistersToSSAOptions& options = {}) {
        auto pass = createLiftAsmRegistersToSSAPass(options);
        pass->run(*func, passCtx, am);
    }

    /// Physical instruction stream, used to prove the pass rewrites nothing.
    std::string physicalIR() const {
        std::ostringstream out;
        AsmPrinter printer(out);
        printer.print(*func);
        return out.str();
    }

    std::unique_ptr<Function> func;
    BasicBlock* entry = nullptr;
    PassContext passCtx;
    AnalysisManager am;
};

}  // namespace

TEST_F(LiftAsmRegistersToSSAPassTest, HasNameAndStableID) {
    auto first = createLiftAsmRegistersToSSAPass();
    auto second = createLiftAsmRegistersToSSAPass();

    ASSERT_NE(first, nullptr);
    EXPECT_STREQ(first->getName(), "Lift Asm Registers to SSA");
    EXPECT_EQ(first->getPassID(), second->getPassID());
}

TEST_F(LiftAsmRegistersToSSAPassTest, AttachesVerifiedSSAOnSuccess) {
    createVAddInBlock(entry, kArch, 2, 0, 1);

    runPass();

    ASSERT_TRUE(func->hasCanonicalSSA());
    EXPECT_EQ(func->getCanonicalSSA().valueCount(), 3u);
    EXPECT_TRUE(verifyCanonicalSSA(*func).ok());
}

TEST_F(LiftAsmRegistersToSSAPassTest, RunsThroughThePassManager) {
    createVAddInBlock(entry, kArch, 2, 0, 1);

    PassManager pm;
    pm.addPass(createLiftAsmRegistersToSSAPass());
    pm.run(*func);

    ASSERT_TRUE(func->hasCanonicalSSA());
    EXPECT_TRUE(verifyCanonicalSSA(*func).ok());
}

TEST_F(LiftAsmRegistersToSSAPassTest, DoesNotRewritePhysicalOperands) {
    createDsReadB128InBlock(entry, kArch, 4, 0);
    createVAddInBlock(entry, kArch, 8, 4, 5);
    const std::string before = physicalIR();

    runPass();

    ASSERT_TRUE(func->hasCanonicalSSA());
    EXPECT_EQ(physicalIR(), before);
}

TEST_F(LiftAsmRegistersToSSAPassTest, UnsupportedFunctionIsLeftWithoutSSA) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    func->createBasicBlock("second");

    runPass();

    EXPECT_FALSE(func->hasCanonicalSSA());
}

TEST_F(LiftAsmRegistersToSSAPassTest, FailureDetachesAStaleGraph) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    runPass();
    ASSERT_TRUE(func->hasCanonicalSSA());

    // Make the function unsupported, then run again: keeping the old graph
    // would leave SSA that no longer describes the function.
    func->createBasicBlock("second");
    runPass();

    EXPECT_FALSE(func->hasCanonicalSSA());
}

TEST_F(LiftAsmRegistersToSSAPassTest, RerunRebuildsAnEquivalentGraph) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    createVAddInBlock(entry, kArch, 3, 2, 0);

    runPass();
    ASSERT_TRUE(func->hasCanonicalSSA());
    const std::string first = canonicalSSAToString(*func);

    runPass();
    ASSERT_TRUE(func->hasCanonicalSSA());
    EXPECT_EQ(canonicalSSAToString(*func), first);
}

TEST_F(LiftAsmRegistersToSSAPassTest, RefusesToRunWhenBlockFilteringExcludesABlock) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    passCtx.setBasicBlockFilter(BasicBlockFilterBuilder::byLabels({"somewhere_else"}));

    runPass();

    EXPECT_FALSE(func->hasCanonicalSSA());
}

TEST_F(LiftAsmRegistersToSSAPassTest, RunsWhenBlockFilteringIncludesEveryBlock) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    passCtx.setBasicBlockFilter(BasicBlockFilterBuilder::byLabels({"entry"}));

    runPass();

    EXPECT_TRUE(func->hasCanonicalSSA());
}

TEST_F(LiftAsmRegistersToSSAPassTest, ForwardsOptionsToTheLifter) {
    createVAddInBlock(entry, kArch, 2, 0, 1);

    LiftAsmRegistersToSSAOptions options;
    options.allowInferredLiveIns = false;
    runPass(options);

    EXPECT_FALSE(func->hasCanonicalSSA());
}

TEST_F(LiftAsmRegistersToSSAPassTest, ReportsWhyAFunctionWasNotLifted) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    func->createBasicBlock("second");
    passCtx.setRemarksEnabled(true);

    std::ostringstream captured;
    std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
    runPass();
    std::cerr.rdbuf(previous);

    const std::string text = captured.str();
    EXPECT_TRUE(contains(text, "missed: LiftAsmRegistersToSSA")) << text;
    EXPECT_TRUE(contains(text, "requires a single basic block")) << text;
}

TEST_F(LiftAsmRegistersToSSAPassTest, ReportsWhatWasLifted) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    passCtx.setRemarksEnabled(true);

    std::ostringstream captured;
    std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
    runPass();
    std::cerr.rdbuf(previous);

    const std::string text = captured.str();
    EXPECT_TRUE(contains(text, "remark: LiftAsmRegistersToSSA")) << text;
    EXPECT_TRUE(contains(text, "@kernel: lifted 3 SSA value(s) and 0 phi(s)")) << text;
}

TEST_F(LiftAsmRegistersToSSAPassTest, StaysQuietWhenRemarksAreDisabled) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    func->createBasicBlock("second");

    std::ostringstream captured;
    std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
    runPass();
    std::cerr.rdbuf(previous);

    EXPECT_TRUE(captured.str().empty()) << captured.str();
}

TEST_F(LiftAsmRegistersToSSAPassTest, PreservesCFGAnalyses) {
    createVAddInBlock(entry, kArch, 2, 0, 1);

    auto pass = createLiftAsmRegistersToSSAPass();
    const PreservedAnalyses preserved = pass->run(*func, passCtx, am);

    EXPECT_TRUE(preserved.isPreserved<DominanceAnalysis>());
    EXPECT_TRUE(preserved.isPreserved<BBIndexAnalysis>());
    EXPECT_TRUE(preserved.isPreserved<LoopAnalysis>());
}
