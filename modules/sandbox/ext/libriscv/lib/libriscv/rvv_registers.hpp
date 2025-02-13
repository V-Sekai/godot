/**************************************************************************/
/*  rvv_registers.hpp                                                     */
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

#ifndef RVV_REGISTERS_HPP
#define RVV_REGISTERS_HPP
#include "types.hpp"
#include <array>

namespace riscv {
union alignas(RISCV_EXT_VECTOR) VectorLane {
	static constexpr unsigned VSIZE = RISCV_EXT_VECTOR;
	static constexpr unsigned size() noexcept { return VSIZE; }

	std::array<uint8_t, VSIZE / 1> u8 = {};
	std::array<uint16_t, VSIZE / 2> u16;
	std::array<uint32_t, VSIZE / 4> u32;
	std::array<uint64_t, VSIZE / 8> u64;

	std::array<float, VSIZE / 4> f32;
	std::array<double, VSIZE / 8> f64;
};
static_assert(sizeof(VectorLane) == RISCV_EXT_VECTOR, "Vectors are 32 bytes");
static_assert(alignof(VectorLane) == RISCV_EXT_VECTOR, "Vectors are 32-byte aligned");

template <int W>
struct alignas(RISCV_EXT_VECTOR) VectorRegisters {
	using address_t = address_type<W>; // one unsigned memory address
	using register_t = register_type<W>; // integer register

	auto &get(unsigned idx) noexcept { return m_vec[idx]; }
	auto &f32(unsigned idx) { return m_vec[idx].f32; }
	auto &u32(unsigned idx) { return m_vec[idx].u32; }

	register_t vtype() const noexcept {
		return 0u;
	}

private:
	std::array<VectorLane, 32> m_vec{};
};
} //namespace riscv
#endif // RVV_REGISTERS_HPP
