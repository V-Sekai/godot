/**************************************************************************/
/*  rva128_instr.cpp                                                      */
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

#include "instr_helpers.hpp"
#include <inttypes.h>
#include <cstdint>
#define AMOSIZE_W 0x2
#define AMOSIZE_D 0x3
#define AMOSIZE_Q 0x4

namespace riscv {
ATOMIC_INSTR(AMOADD_Q, [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR { cpu.template amo<__int128_t>(instr,
																						 [](auto &cpu, auto &value, auto rs2) {
																							 auto old_value = value;
																							 value += cpu.reg(rs2);
																							 return old_value;
																						 }); }, DECODED_ATOMIC(AMOADD_W).printer);

ATOMIC_INSTR(AMOXOR_Q, [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR { cpu.template amo<__int128_t>(instr,
																						 [](auto &cpu, auto &value, auto rs2) {
																							 auto old_value = value;
																							 value ^= cpu.reg(rs2);
																							 return old_value;
																						 }); }, DECODED_ATOMIC(AMOADD_W).printer);

ATOMIC_INSTR(AMOOR_Q, [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR { cpu.template amo<__int128_t>(instr,
																						[](auto &cpu, auto &value, auto rs2) {
																							auto old_value = value;
																							value |= cpu.reg(rs2);
																							return old_value;
																						}); }, DECODED_ATOMIC(AMOADD_W).printer);

ATOMIC_INSTR(AMOAND_Q, [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR { cpu.template amo<__int128_t>(instr,
																						 [](auto &cpu, auto &value, auto rs2) {
																							 auto old_value = value;
																							 value &= cpu.reg(rs2);
																							 return old_value;
																						 }); }, DECODED_ATOMIC(AMOADD_W).printer);

ATOMIC_INSTR(AMOSWAP_Q, [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR { cpu.template amo<__int128_t>(instr,
																						  [](auto &cpu, auto &value, auto rs2) {
																							  auto old_value = value;
																							  value = cpu.reg(rs2);
																							  return old_value;
																						  }); }, DECODED_ATOMIC(AMOSWAP_W).printer);

} //namespace riscv
