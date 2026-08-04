/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

/**
 * Unit tests to verify that when a Function is destructed, all owned
 * BasicBlocks and IRs are deleted correctly without memory leaks.
 *
 * Uses a test-only IR type with a static alive counter to assert that
 * every created IR is destroyed when the Function goes out of scope.
 * For full process-level leak checking, run tests with LeakSanitizer
 * (e.g. LSAN_OPTIONS=detect_leaks=1).
 */

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/CanonicalSSA.hpp"

using namespace stinkytofu;

namespace {
// Test-only IR type to verify that IR instances are deleted when the owning
// Function (and its BasicBlocks' IR lists) are destroyed.
class CountedTestIR : public IRBase {
   public:
    static int s_alive;

    CountedTestIR() : IRBase(IRType::LogicalIR) {
        ++s_alive;
    }

    ~CountedTestIR() override {
        --s_alive;
    }

    void dump(std::ostream& out) const override {
        out << "CountedTestIR";
    }
};

int CountedTestIR::s_alive = 0;

/// Sidecar holding one live-in value whose origin is a full-DWORD VGPR.
std::unique_ptr<CanonicalSSA> makeSidecar(unsigned regIdx) {
    CanonicalSSABuilder builder;
    SSAValue liveIn;
    liveIn.kind = SSAValueKind::LiveIn;
    liveIn.origin = {RegType::V, regIdx, RegHalf::NONE};
    builder.addValue(std::move(liveIn));
    return std::make_unique<CanonicalSSA>(builder.take());
}
}  // namespace

class FunctionDestructionTest : public ::testing::Test {
   protected:
    void SetUp() override {
        CountedTestIR::s_alive = 0;
    }

    void TearDown() override {
        EXPECT_EQ(CountedTestIR::s_alive, 0)
            << "All CountedTestIR instances should be destroyed after each test";
    }
};

TEST_F(FunctionDestructionTest, DestructingFunctionDeletesAllIRs) {
    EXPECT_EQ(CountedTestIR::s_alive, 0);

    {
        Function func("kernel");
        BasicBlock* bb = func.createBasicBlock("entry");
        ASSERT_NE(bb, nullptr);

        IRBase* ir1 = IRBase::createIR<CountedTestIR>();
        IRBase* ir2 = IRBase::createIR<CountedTestIR>();
        bb->appendIR(ir1);
        bb->appendIR(ir2);

        EXPECT_EQ(CountedTestIR::s_alive, 2);
    }

    EXPECT_EQ(CountedTestIR::s_alive, 0);
}

TEST_F(FunctionDestructionTest, DestructingFunctionDeletesAllBasicBlocksAndIRs) {
    EXPECT_EQ(CountedTestIR::s_alive, 0);

    {
        Function func("kernel");
        BasicBlock* bb1 = func.createBasicBlock("bb1");
        BasicBlock* bb2 = func.createBasicBlock("bb2");
        BasicBlock* bb3 = func.createBasicBlock("bb3");
        ASSERT_NE(bb1, nullptr);
        ASSERT_NE(bb2, nullptr);
        ASSERT_NE(bb3, nullptr);

        bb1->appendIR(IRBase::createIR<CountedTestIR>());
        bb1->appendIR(IRBase::createIR<CountedTestIR>());
        bb2->appendIR(IRBase::createIR<CountedTestIR>());
        bb3->appendIR(IRBase::createIR<CountedTestIR>());
        bb3->appendIR(IRBase::createIR<CountedTestIR>());

        EXPECT_EQ(CountedTestIR::s_alive, 5);
        EXPECT_EQ(func.size(), 3u);
    }

    EXPECT_EQ(CountedTestIR::s_alive, 0);
}

TEST_F(FunctionDestructionTest, ClearThenDestructLeavesNoIRs) {
    EXPECT_EQ(CountedTestIR::s_alive, 0);

    {
        Function func("kernel");
        BasicBlock* bb = func.createBasicBlock("entry");
        bb->appendIR(IRBase::createIR<CountedTestIR>());
        bb->appendIR(IRBase::createIR<CountedTestIR>());
        EXPECT_EQ(CountedTestIR::s_alive, 2);

        func.clear();
        EXPECT_EQ(CountedTestIR::s_alive, 0);
        EXPECT_TRUE(func.empty());
    }

    EXPECT_EQ(CountedTestIR::s_alive, 0);
}

TEST_F(FunctionDestructionTest, NoCanonicalSSAAttachedByDefault) {
    Function func("kernel");
    EXPECT_FALSE(func.hasCanonicalSSA());
}

TEST_F(FunctionDestructionTest, AttachAndDetachCanonicalSSA) {
    Function func("kernel");

    func.setCanonicalSSA(makeSidecar(8));
    ASSERT_TRUE(func.hasCanonicalSSA());
    EXPECT_EQ(func.getCanonicalSSA().valueCount(), 1u);
    EXPECT_EQ(func.getCanonicalSSA().value(1).origin.idx, 8u);

    func.clearCanonicalSSA();
    EXPECT_FALSE(func.hasCanonicalSSA());

    // Detaching twice is safe.
    func.clearCanonicalSSA();
    EXPECT_FALSE(func.hasCanonicalSSA());
}

TEST_F(FunctionDestructionTest, ReplacingCanonicalSSAReleasesThePreviousGraph) {
    Function func("kernel");

    func.setCanonicalSSA(makeSidecar(8));
    func.setCanonicalSSA(makeSidecar(20));
    ASSERT_TRUE(func.hasCanonicalSSA());
    EXPECT_EQ(func.getCanonicalSSA().value(1).origin.idx, 20u);

    func.setCanonicalSSA(nullptr);
    EXPECT_FALSE(func.hasCanonicalSSA());
}

TEST_F(FunctionDestructionTest, ClearDetachesCanonicalSSABeforeDeletingIRs) {
    EXPECT_EQ(CountedTestIR::s_alive, 0);

    {
        Function func("kernel");
        BasicBlock* bb = func.createBasicBlock("entry");
        bb->appendIR(IRBase::createIR<CountedTestIR>());
        func.setCanonicalSSA(makeSidecar(8));
        ASSERT_TRUE(func.hasCanonicalSSA());

        func.clear();

        EXPECT_FALSE(func.hasCanonicalSSA());
        EXPECT_TRUE(func.empty());
        EXPECT_EQ(CountedTestIR::s_alive, 0);
    }

    EXPECT_EQ(CountedTestIR::s_alive, 0);
}

TEST_F(FunctionDestructionTest, DestructingFunctionReleasesCanonicalSSA) {
    EXPECT_EQ(CountedTestIR::s_alive, 0);

    {
        Function func("kernel");
        BasicBlock* bb = func.createBasicBlock("entry");
        bb->appendIR(IRBase::createIR<CountedTestIR>());
        func.setCanonicalSSA(makeSidecar(8));
        EXPECT_EQ(CountedTestIR::s_alive, 1);
    }

    EXPECT_EQ(CountedTestIR::s_alive, 0);
}

TEST_F(FunctionDestructionTest, RepeatedCreateAndDestroyNoLeak) {
    for (int iter = 0; iter < 3; ++iter) {
        Function func("repeated");
        for (int b = 0; b < 5; ++b) {
            BasicBlock* bb = func.createBasicBlock("bb" + std::to_string(b));
            for (int i = 0; i < 3; ++i) {
                bb->appendIR(IRBase::createIR<CountedTestIR>());
            }
        }
        EXPECT_EQ(CountedTestIR::s_alive, 15);
    }
    EXPECT_EQ(CountedTestIR::s_alive, 0);
}
