/**************************************************************************/
/*  memory_mmap.cpp                                                       */
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

#include "internal_common.hpp"
#include "machine.hpp"

namespace riscv {
template <int W>
address_type<W> Memory<W>::mmap_allocate(address_t bytes) {
	// Bytes rounded up to nearest PageSize.
	const address_t result = this->m_mmap_address;
	this->m_mmap_address += (bytes + PageMask) & ~address_t{ PageMask };
	return result;
}

template <int W>
bool Memory<W>::mmap_relax(address_t addr, address_t size, address_t new_size) {
	// Undo or relax the last mmap allocation. Returns true if successful.
	if (this->m_mmap_address == addr + size && new_size <= size) {
		this->m_mmap_address = (addr + new_size + PageMask) & ~address_t{ PageMask };
		return true;
	}
	return false;
}

template <int W>
bool Memory<W>::mmap_unmap(address_t addr, address_t size) {
	size = (size + PageMask) & ~address_t{ PageMask };
	const bool relaxed = this->mmap_relax(addr, size, 0u);
	if (relaxed) {
		// If relaxation happened, invalidate intersecting cache entries.
		this->mmap_cache().invalidate(addr, size);
	} else if (addr >= this->mmap_start()) {
		// If relaxation didn't happen, put in the cache for later.
		this->mmap_cache().insert(addr, size);
	}
	return relaxed;
}

INSTANTIATE_32_IF_ENABLED(Memory);
INSTANTIATE_64_IF_ENABLED(Memory);
INSTANTIATE_128_IF_ENABLED(Memory);
} //namespace riscv
