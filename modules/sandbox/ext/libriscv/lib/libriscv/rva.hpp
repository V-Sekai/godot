/**************************************************************************/
/*  rva.hpp                                                               */
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

#ifndef RVA_HPP
#define RVA_HPP
#include "types.hpp"
#include <cstdint>

namespace riscv {
template <int W>
struct AtomicMemory {
	using address_t = address_type<W>;

	bool load_reserve(int size, address_t addr) RISCV_INTERNAL {
		if (!check_alignment(size, addr))
			return false;

		m_reservation = addr;
		return true;
	}

	// Volume I: RISC-V Unprivileged ISA V20190608 p.49:
	// An SC can only pair with the most recent LR in program order.
	bool store_conditional(int size, address_t addr) RISCV_INTERNAL {
		if (!check_alignment(size, addr))
			return false;

		bool result = m_reservation == addr;
		// Regardless of success or failure, executing an SC.W
		// instruction invalidates any reservation held by this hart.
		m_reservation = 0x0;
		return result;
	}

private:
	inline bool check_alignment(int size, address_t addr) RISCV_INTERNAL {
		return (addr & (size - 1)) == 0;
	}

	address_t m_reservation = 0x0;
};
} //namespace riscv

#endif // RVA_HPP
