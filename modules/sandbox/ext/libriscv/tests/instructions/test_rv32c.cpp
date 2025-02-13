/**************************************************************************/
/*  test_rv32c.cpp                                                        */
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

#include "testable_instruction.hpp"
#include <cassert>
#include <libriscv/rvc.hpp>
using namespace riscv;

void test_rv32c() {
	riscv::Machine<RISCV32> machine{ std::string_view{}, { .memory_max = 65536 } };

	// C.SRLI imm = [0, 31] CI_CODE(0b100, 0b01):
	for (int i = 0; i < 32; i++) {
		rv32c_instruction ci;
		ci.CA.opcode = 0b01; // Quadrant 1
		ci.CA.funct6 = 0b100000; // ALU OP: SRLI
		ci.CAB.srd = 0x2; // A0
		ci.CAB.imm04 = i;
		ci.CAB.imm5 = 0;

		const testable_insn<RISCV32> insn{
			.name = "C.SRLI",
			.bits = ci.whole,
			.reg = REG_ARG0,
			.index = i,
			.initial_value = 0xFFFFFFFF
		};
		bool b = validate<RISCV32>(machine, insn,
				[](auto &cpu, const auto &insn) -> bool {
					return cpu.reg(insn.reg) == (insn.initial_value >> insn.index);
				});
		assert(b);
	}

	// C.SRAI imm = [0, 31] CI_CODE(0b100, 0b01):
	for (int i = 0; i < 32; i++) {
		rv32c_instruction ci;
		ci.CA.opcode = 0b01; // Quadrant 1
		ci.CA.funct6 = 0b100001; // ALU OP: SRAI
		ci.CAB.srd = 0x2; // A0
		ci.CAB.imm04 = i;
		ci.CAB.imm5 = 0;

		const testable_insn<RISCV32> insn{
			.name = "C.SRAI",
			.bits = ci.whole,
			.reg = REG_ARG0,
			.index = i,
			.initial_value = 0xFFFFFFFF
		};
		bool b = validate<RISCV32>(machine, insn,
				[](auto &cpu, const auto &insn) -> bool {
					return cpu.reg(insn.reg) == insn.initial_value;
				});
		assert(b);
	}

	// C.ANDI imm = [-32, 31] CI_CODE(0b100, 0b01):
	for (int i = 0; i < 64; i++) {
		rv32c_instruction ci;
		ci.CA.opcode = 0b01; // Quadrant 1
		ci.CA.funct6 = 0b100010; // ALU OP: ANDI
		ci.CAB.srd = 0x2; // A0
		ci.CAB.imm04 = i & 31;
		ci.CAB.imm5 = i >> 5;

		const testable_insn<RISCV32> insn{
			.name = "C.ANDI",
			.bits = ci.whole,
			.reg = REG_ARG0,
			.index = i,
			.initial_value = 0xFFFFFFFF
		};
		bool b = validate<RISCV32>(machine, insn,
				[](auto &cpu, const auto &insn) -> bool {
					if (insn.index < 32) {
						return cpu.reg(insn.reg) == (insn.initial_value & insn.index);
					}
					return cpu.reg(insn.reg) == (insn.initial_value & (insn.index - 64));
				});
		assert(b);
	}

	// C.SLLI imm = [0, 31] CI_CODE(0b011, 0b10):
	for (int i = 0; i < 32; i++) {
		rv32c_instruction ci;
		ci.CI.opcode = 0b10; // Quadrant 1
		ci.CI.funct3 = 0x0; // OP: SLLI
		ci.CI.rd = 0xA; // A0
		ci.CI.imm1 = i;
		ci.CI.imm2 = 0;

		const testable_insn<RISCV32> insn{
			.name = "C.SLLI",
			.bits = ci.whole,
			.reg = REG_ARG0,
			.index = i,
			.initial_value = 0xA
		};
		bool b = validate<RISCV32>(machine, insn,
				[](auto &cpu, const auto &insn) -> bool {
					return cpu.reg(insn.reg) == (insn.initial_value << insn.index);
				});
		assert(b);
	}

	printf("%lu instructions passed.\n", machine.instruction_counter());
}
