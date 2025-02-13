/**************************************************************************/
/*  tr_types.hpp                                                          */
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

#ifndef TR_TYPES_HPP
#define TR_TYPES_HPP
#include "rv32i_instr.hpp"
#include "types.hpp"
#include <unordered_set>
#include <vector>

namespace riscv {
template <int W>
struct TransInstr;

template <int W>
struct TransOutput {
	std::unordered_map<std::string, std::string> defines;
	timespec t0;
	std::shared_ptr<std::string> code;
	std::string footer;
	std::vector<TransMapping<W>> mappings;
};

template <int W>
struct TransInfo {
	const std::vector<rv32i_instruction> instr;
	address_type<W> basepc;
	address_type<W> endpc;
	address_type<W> segment_basepc;
	address_type<W> segment_endpc;
	address_type<W> gp;
	bool trace_instructions;
	bool ignore_instruction_limit;
	bool use_shared_execute_segments;
	bool use_register_caching;
	bool use_automatic_nbit_address_space;
	std::unordered_set<address_type<W>> jump_locations;
	std::unordered_map<address_type<W>, address_type<W>> single_return_locations;
	// Pointer to all the other blocks (including current)
	std::vector<TransInfo<W>> *blocks = nullptr;
	// Pointer to list of ebreak-locations
	const std::unordered_set<address_type<W>> *ebreak_locations = nullptr;

	std::unordered_set<address_type<W>> &global_jump_locations;

	uintptr_t arena_ptr;
	address_type<W> arena_roend;
	address_type<W> arena_size;
};
} //namespace riscv

#endif // TR_TYPES_HPP
