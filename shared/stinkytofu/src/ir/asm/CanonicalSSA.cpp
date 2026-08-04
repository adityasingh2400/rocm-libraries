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
#include "stinkytofu/ir/asm/CanonicalSSA.hpp"

#include <cassert>
#include <limits>
#include <utility>

namespace stinkytofu {
namespace {

const std::vector<SSAPhiID> kEmptyPhiList;

}  // namespace

bool CanonicalSSA::empty() const {
    return values_.empty() && phis_.empty() && instructions_.empty() && blockPhis_.empty();
}

size_t CanonicalSSA::valueCount() const {
    return values_.size();
}

size_t CanonicalSSA::phiCount() const {
    return phis_.size();
}

bool CanonicalSSA::containsValue(SSAValueID id) const {
    return id != kInvalidSSAValueID && id <= values_.size();
}

bool CanonicalSSA::containsPhi(SSAPhiID id) const {
    return id != kInvalidSSAPhiID && id <= phis_.size();
}

const SSAValue& CanonicalSSA::value(SSAValueID id) const {
    assert(containsValue(id) && "invalid SSA value ID");
    return values_.at(id - 1);
}

const SSAPhi& CanonicalSSA::phi(SSAPhiID id) const {
    assert(containsPhi(id) && "invalid SSA PHI ID");
    return phis_.at(id - 1);
}

const std::vector<SSAValue>& CanonicalSSA::values() const {
    return values_;
}

const std::vector<SSAPhi>& CanonicalSSA::phis() const {
    return phis_;
}

const SSAInstructionInfo* CanonicalSSA::findInstructionInfo(
    const StinkyInstruction& instruction) const {
    auto it = instructions_.find(&instruction);
    return it == instructions_.end() ? nullptr : &it->second;
}

const std::vector<SSAPhiID>& CanonicalSSA::phisForBlock(const BasicBlock& block) const {
    auto it = blockPhis_.find(&block);
    return it == blockPhis_.end() ? kEmptyPhiList : it->second;
}

SSAValueID CanonicalSSABuilder::addValue(SSAValue value) {
    assert(ssa_.values_.size() < std::numeric_limits<SSAValueID>::max() && "SSA value ID overflow");
    value.id = static_cast<SSAValueID>(ssa_.values_.size() + 1);
    ssa_.values_.push_back(std::move(value));
    return ssa_.values_.back().id;
}

SSAPhiID CanonicalSSABuilder::addPhi(SSAPhi phi) {
    assert(ssa_.phis_.size() < std::numeric_limits<SSAPhiID>::max() && "SSA PHI ID overflow");
    phi.id = static_cast<SSAPhiID>(ssa_.phis_.size() + 1);
    ssa_.phis_.push_back(std::move(phi));
    return ssa_.phis_.back().id;
}

SSAValue& CanonicalSSABuilder::value(SSAValueID id) {
    assert(ssa_.containsValue(id) && "invalid SSA value ID");
    return ssa_.values_.at(id - 1);
}

SSAPhi& CanonicalSSABuilder::phi(SSAPhiID id) {
    assert(ssa_.containsPhi(id) && "invalid SSA PHI ID");
    return ssa_.phis_.at(id - 1);
}

void CanonicalSSABuilder::setInstructionInfo(const StinkyInstruction& instruction,
                                             SSAInstructionInfo info) {
    ssa_.instructions_.insert_or_assign(&instruction, std::move(info));
}

void CanonicalSSABuilder::addPhiToBlock(const BasicBlock& block, SSAPhiID phiID) {
    assert(ssa_.containsPhi(phiID) && "invalid SSA PHI ID");
    ssa_.blockPhis_[&block].push_back(phiID);
}

CanonicalSSA CanonicalSSABuilder::take() {
    CanonicalSSA result = std::move(ssa_);
    ssa_ = CanonicalSSA{};
    return result;
}

}  // namespace stinkytofu
