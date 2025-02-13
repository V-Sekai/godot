/**************************************************************************/
/*  testable_instruction.hpp                                              */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef TESTABLE_INSTRUCTION_HPP
#define TESTABLE_INSTRUCTION_HPP
#include <cstdio>
#include <libriscv/machine.hpp>
#include <libriscv/rv32i_instr.hpp>

namespace riscv {
template <int W>
struct testable_insn {
	const char *name; // test name
	address_type<W> bits; // the instruction bits
	const int reg; // which register this insn affects
	const int index; // test loop index
	address_type<W> initial_value; // start value of register
};

template <int W>
static bool
validate(Machine<W> &machine, const testable_insn<W> &insn,
		std::function<bool(CPU<W> &, const testable_insn<W> &)> callback) {
	static const address_type<W> MEMBASE = 0x1000;

	const std::array<uint32_t, 1> instr_page = {
		insn.bits
	};

	DecodedExecuteSegment<W> &des = machine.cpu.init_execute_area(&instr_page[0], MEMBASE, sizeof(instr_page));
	// jump to page containing instruction
	machine.cpu.jump(MEMBASE);
	// execute instruction
	machine.cpu.reg(insn.reg) = insn.initial_value;
	machine.cpu.step_one();
	// There is a max number of execute segments. Evict the latest to avoid the max limit check
	machine.cpu.memory().evict_execute_segment(des);
	// call instruction validation callback
	if (callback(machine.cpu, insn))
		return true;
	fprintf(stderr, "Failed test: %s on iteration %d\n", insn.name, insn.index);
	fprintf(stderr, "Register value: 0x%X\n", machine.cpu.reg(insn.reg));
	return false;
}
} //namespace riscv

#endif // TESTABLE_INSTRUCTION_HPP
