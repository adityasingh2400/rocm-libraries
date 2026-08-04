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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "stinkytofu/core/Function.hpp"

#include <cassert>
#include <iostream>
#include <ostream>
#include <utility>

#include "stinkytofu/ir/asm/CanonicalSSA.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"

namespace stinkytofu {
Function::Function(const std::string& name) : name(name), basicBlocks(this) {}

Function::~Function() = default;

CanonicalSSA& Function::getCanonicalSSA() {
    assert(canonicalSSA && "no canonical SSA attached to this function");
    return *canonicalSSA;
}

const CanonicalSSA& Function::getCanonicalSSA() const {
    assert(canonicalSSA && "no canonical SSA attached to this function");
    return *canonicalSSA;
}

void Function::setCanonicalSSA(std::unique_ptr<CanonicalSSA> ssa) {
    canonicalSSA = std::move(ssa);
}

void Function::clearCanonicalSSA() {
    canonicalSSA.reset();
}

void Function::clear() {
    clearCanonicalSSA();
    basicBlocks.clear();
}

void Function::dump(std::ostream& out) const {
    AsmPrinter printer(out, AsmPrinterOptions());
    printer.print(*this);
}

void Function::dump() const {
    dump(std::cerr);
}
}  // namespace stinkytofu
