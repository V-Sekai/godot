/**************************************************************************/
/*  rvfd.hpp                                                              */
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

#ifndef RVFD_HPP
#define RVFD_HPP
#include "rv32i_instr.hpp"

namespace riscv {
union rv32f_instruction {
	struct {
		uint32_t opcode : 7;
		uint32_t rd : 5;
		uint32_t funct3 : 3;
		uint32_t rs1 : 5;
		uint32_t rs2 : 5;
		uint32_t funct7 : 7;
	} Rtype;
	struct {
		uint32_t opcode : 7;
		uint32_t rd : 5;
		uint32_t funct3 : 3;
		uint32_t rs1 : 5;
		uint32_t rs2 : 5;
		uint32_t funct2 : 2;
		uint32_t rs3 : 5;
	} R4type;
	struct {
		uint32_t opcode : 7;
		uint32_t rd : 5;
		uint32_t funct3 : 3;
		uint32_t rs1 : 5;
		uint32_t imm : 12;

		bool sign() const noexcept {
			return imm & 0x800;
		}
		int32_t signed_imm() const noexcept {
			return int32_t(imm << 20) >> 20;
		}
	} Itype;
	struct {
		uint32_t opcode : 7;
		uint32_t imm04 : 5;
		uint32_t funct3 : 3;
		uint32_t rs1 : 5;
		uint32_t rs2 : 5;
		uint32_t imm510 : 6;
		uint32_t imm11 : 1;

		bool sign() const noexcept {
			return imm11;
		}
		int32_t signed_imm() const noexcept {
			const int32_t imm = imm04 | (imm510 << 5) | (imm11 << 11);
			return (imm << 20) >> 20;
		}
	} Stype;

	uint16_t half[2];
	uint32_t whole;

	rv32f_instruction(rv32i_instruction i) :
			whole(i.whole) {}

	uint32_t opcode() const noexcept {
		return Rtype.opcode;
	}
};
static_assert(sizeof(rv32f_instruction) == 4, "Must be 4 bytes");

enum fflags {
	FFLAG_NX = 0x1,
	FFLAG_UF = 0x2,
	FFLAG_OF = 0x4,
	FFLAG_DZ = 0x8,
	FFLAG_NV = 0x10
};
} //namespace riscv

#endif // RVFD_HPP
