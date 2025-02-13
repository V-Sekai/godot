/**************************************************************************/
/*  safe_instr_loader.hpp                                                 */
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

#ifndef SAFE_INSTR_LOADER_HPP
#define SAFE_INSTR_LOADER_HPP
#include "rv32i_instr.hpp"

namespace riscv {
struct UnalignedLoad32 {
	uint16_t data[2];
	operator uint32_t() const {
		return data[0] | uint32_t(data[1]) << 16;
	}
};
struct AlignedLoad16 {
	uint16_t data;
	operator uint32_t() { return data; }
	unsigned length() const {
		return (data & 3) != 3 ? 2 : 4;
	}
	unsigned opcode() const {
		return data & 0x7F;
	}
	uint16_t half() const {
		return data;
	}
};
inline rv32i_instruction read_instruction(
		const uint8_t *exec_segment, uint64_t pc, uint64_t end_pc) {
	if (pc + 4 <= end_pc)
		return rv32i_instruction{ *(UnalignedLoad32 *)&exec_segment[pc] };
	else if (pc + 2 <= end_pc)
		return rv32i_instruction{ *(AlignedLoad16 *)&exec_segment[pc] };
	else
		return rv32i_instruction{ 0 };
}
} //namespace riscv

#endif // SAFE_INSTR_LOADER_HPP
