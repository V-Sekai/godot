/**************************************************************************/
/*  mmap_cache.hpp                                                        */
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

#ifndef MMAP_CACHE_HPP
#define MMAP_CACHE_HPP
#include "types.hpp"
#include <array>
#include <cstdint>
#include <vector>

namespace riscv {
template <int W>
struct MMapCache {
	using address_t = address_type<W>;

	struct Range {
		address_t addr = 0x0;
		address_t size = 0u;

		constexpr bool empty() const noexcept { return size == 0u; }
		// Invalidate if one of the ranges is in the other (both ways!)
		constexpr bool within(address_t mem, address_t memsize) const noexcept {
			return ((this->addr >= mem) && (this->addr + this->size <= mem + memsize)) || ((mem >= this->addr) && (mem + memsize <= this->addr + this->size));
		}
		constexpr bool equals(address_t mem, address_t memsize) const noexcept {
			return (this->addr == mem) && (this->addr + this->size == mem + memsize);
		}
	};

	Range find(address_t size) {
		auto it = m_lines.begin();
		while (it != m_lines.end()) {
			auto &r = *it;
			if (!r.empty()) {
				if (r.size >= size) {
					const Range result{ r.addr, size };
					if (r.size > size) {
						r.addr += size;
						r.size -= size;
					} else {
						m_lines.erase(it);
					}
					return result;
				}
			}
			++it;
		}
		return Range{};
	}

	void invalidate(address_t addr, address_t size) {
		auto it = m_lines.begin();
		while (it != m_lines.end()) {
			const auto r = *it;
			if (r.within(addr, size)) {
				bool equals = r.equals(addr, size);
				it = m_lines.erase(it);
				if (equals)
					return;
			} else
				++it;
		}
	}

	void insert(address_t addr, address_t size) {
		/* Extend the back range? */
		if (!m_lines.empty()) {
			if (m_lines.back().addr + m_lines.back().size == addr) {
				m_lines.back().size += size;
				return;
			}
		}

		m_lines.push_back({ addr, size });
	}

private:
	std::vector<Range> m_lines{};
};

} //namespace riscv

#endif // MMAP_CACHE_HPP
