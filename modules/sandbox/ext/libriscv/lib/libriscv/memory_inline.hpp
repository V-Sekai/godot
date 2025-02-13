/**************************************************************************/
/*  memory_inline.hpp                                                     */
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

#ifndef MEMORY_INLINE_HPP
#define MEMORY_INLINE_HPP
// Force-align memory operations to their native alignments
template <typename T>
constexpr inline size_t memory_align_mask() {
	if constexpr (force_align_memory)
		return size_t(Page::size() - 1) & ~size_t(sizeof(T) - 1);
	else
		return size_t(Page::size() - 1);
}

template <int W>
template <typename T>
inline T Memory<W>::read(address_t address) {
	if constexpr (encompassing_Nbit_arena) {
		if constexpr (encompassing_Nbit_arena == 32)
			return *(T *)&((const char *)m_arena.data)[uint32_t(address)];
		else // It's a power-of-two encompassing arena
			return *(T *)&((const char *)m_arena.data)[address & encompassing_arena_mask];
	} else {
		const auto offset = address & memory_align_mask<T>();
		if constexpr (unaligned_memory_slowpaths) {
			if (UNLIKELY(offset + sizeof(T) > Page::size())) {
				T value;
				memcpy_out(&value, address, sizeof(T));
				return value;
			}
		} else if constexpr (flat_readwrite_arena) {
			if (LIKELY(address - RWREAD_BEGIN < memory_arena_read_boundary())) {
#ifdef RISCV_EXT_VECTOR
				if constexpr (sizeof(T) >= 32) {
					// Reads and writes using vectors might have alignment requirements
					auto *arena = (VectorLane *)m_arena.data;
					return arena[RISCV_SPECSAFE(address / sizeof(VectorLane))];
				}
#endif
				return *(T *)&((const char *)m_arena.data)[RISCV_SPECSAFE(address)];
			}
			[[unlikely]];
		}

		const auto &pagedata = cached_readable_page(address, sizeof(T));
		return pagedata.template aligned_read<T>(offset);

	} // encompassing_Nbit_arena
}

template <int W>
template <typename T>
inline T &Memory<W>::writable_read(address_t address) {
	if constexpr (encompassing_Nbit_arena) {
		if constexpr (encompassing_Nbit_arena == 32)
			return *(T *)&((char *)m_arena.data)[uint32_t(address)];
		else // It's a power-of-two encompassing arena
			return *(T *)&((char *)m_arena.data)[address & encompassing_arena_mask];
	} else {
		if constexpr (flat_readwrite_arena) {
			if (LIKELY(address - initial_rodata_end() < memory_arena_write_boundary())) {
				return *(T *)&((char *)m_arena.data)[RISCV_SPECSAFE(address)];
			}
			[[unlikely]];
		}

		auto &pagedata = cached_writable_page(address);
		return pagedata.template aligned_read<T>(address & memory_align_mask<T>());

	} // encompassing_Nbit_arena
}

template <int W>
template <typename T>
inline void Memory<W>::write(address_t address, T value) {
	if constexpr (encompassing_Nbit_arena) {
		if constexpr (encompassing_Nbit_arena == 32)
			*(T *)&((char *)m_arena.data)[uint32_t(address)] = value;
		else // It's a power-of-two encompassing arena
			*(T *)&((char *)m_arena.data)[address & encompassing_arena_mask] = value;
		return;
	} else {
		const auto offset = address & memory_align_mask<T>();
		if constexpr (unaligned_memory_slowpaths) {
			if (UNLIKELY(offset + sizeof(T) > Page::size())) {
				memcpy(address, &value, sizeof(T));
				return;
			}
		} else if constexpr (flat_readwrite_arena) {
			if (LIKELY(address - initial_rodata_end() < memory_arena_write_boundary())) {
#ifdef RISCV_EXT_VECTOR
				if constexpr (sizeof(T) >= 32) {
					// Reads and writes using vectors might have alignment requirements
					auto *arena = (VectorLane *)m_arena.data;
					arena[RISCV_SPECSAFE(address / sizeof(VectorLane))] = value;
				} else
#endif
					*(T *)&((char *)m_arena.data)[RISCV_SPECSAFE(address)] = value;
				return;
			}
		}

		const auto pageno = page_number(address);
		auto &entry = m_wr_cache;
		if (entry.pageno == pageno) {
			entry.page->template aligned_write<T>(offset, value);
			return;
		}

		auto &page = create_writable_pageno(pageno);
		if (LIKELY(page.attr.is_cacheable())) {
			entry = { pageno, &page.page() };
		} else if constexpr (memory_traps_enabled && sizeof(T) <= 16) {
			if (UNLIKELY(page.has_trap())) {
				page.trap(offset, sizeof(T) | TRAP_WRITE, value);
				return;
			}
		}
		page.page().template aligned_write<T>(offset, value);

	} // encompassing_Nbit_arena
}

template <int W>
inline address_type<W> Memory<W>::resolve_address(std::string_view name) const {
	auto *sym = resolve_symbol(name);
	return (sym) ? sym->st_value : 0x0;
}

template <int W>
inline address_type<W> Memory<W>::resolve_section(const char *name) const {
	auto *shdr = this->section_by_name(name);
	if (shdr)
		return shdr->sh_addr;
	return 0x0;
}

template <int W>
inline address_type<W> Memory<W>::exit_address() const noexcept {
	return this->m_exit_address;
}

template <int W>
inline void Memory<W>::set_exit_address(address_t addr) {
	this->m_exit_address = addr;
}

template <int W>
inline std::shared_ptr<DecodedExecuteSegment<W>> &Memory<W>::exec_segment_for(address_t vaddr) {
	for (auto &segment : m_exec) {
		if (segment && segment->is_within(vaddr))
			return segment;
	}
	return CPU<W>::empty_execute_segment();
}

#endif // MEMORY_INLINE_HPP
