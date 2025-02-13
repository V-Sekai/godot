/**************************************************************************/
/*  memory_inline_pages.hpp                                               */
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

#ifndef MEMORY_INLINE_PAGES_HPP
#define MEMORY_INLINE_PAGES_HPP
template <int W>
inline const PageData &Memory<W>::cached_readable_page(address_t address, size_t len) const {
	const auto pageno = page_number(address);
	auto &entry = m_rd_cache;
	if (entry.pageno == pageno)
		return *entry.page;

	auto &page = get_readable_pageno(pageno);
	if (LIKELY(page.attr.is_cacheable())) {
		entry = { pageno, &page.page() };
	} else if constexpr (memory_traps_enabled) {
		if (UNLIKELY(page.has_trap())) {
			page.trap(address & (Page::size() - 1), len | TRAP_READ, 0);
		}
	}
	return page.page();
}

template <int W>
inline PageData &Memory<W>::cached_writable_page(address_t address) {
	const auto pageno = page_number(address);
	auto &entry = m_wr_cache;
	if (entry.pageno == pageno)
		return *entry.page;
	auto &page = create_writable_pageno(pageno);
	if (LIKELY(page.attr.is_cacheable()))
		entry = { pageno, &page.page() };
	return page.page();
}

template <int W>
inline const Page &Memory<W>::get_page(const address_t address) const {
	const auto page = page_number(address);
	return get_pageno(page);
}

template <int W>
inline const Page &Memory<W>::get_exec_pageno(const address_t pageno) const {
	auto it = m_pages.find(pageno);
	if (LIKELY(it != m_pages.end())) {
		return it->second;
	}
	CPU<W>::trigger_exception(EXECUTION_SPACE_PROTECTION_FAULT, pageno * Page::size());
}

template <int W>
inline const Page &Memory<W>::get_pageno(const address_t pageno) const {
	auto it = m_pages.find(pageno);
	if (LIKELY(it != m_pages.end())) {
		return it->second;
	}

	return m_page_readf_handler(*this, pageno);
}

template <int W>
inline void
Memory<W>::invalidate_cache(address_t pageno, Page *page) const noexcept {
	// NOTE: It is only possible to keep the write page as long as
	// the page tables are node-based. In that case, we only have
	// to invalidate the read page when it matches.
	if (m_rd_cache.pageno == pageno) {
		m_rd_cache.pageno = (address_t)-1;
	}
	(void)page;
}
template <int W>
inline void
Memory<W>::invalidate_reset_cache() const noexcept {
	m_rd_cache.pageno = (address_t)-1;
	m_wr_cache.pageno = (address_t)-1;
}

template <int W>
template <typename... Args>
inline Page &Memory<W>::allocate_page(const address_t page, Args &&...args) {
	const auto it = m_pages.try_emplace(
			page,
			std::forward<Args>(args)...);
	// Invalidate only this page
	this->invalidate_cache(page, &it.first->second);
	// Return new default-writable page
	return it.first->second;
}

template <int W>
inline size_t Memory<W>::owned_pages_active() const noexcept {
	size_t count = 0;
	for (const auto &it : m_pages) {
		if (!it.second.attr.non_owning)
			count++;
	}
	return count;
}

template <int W>
inline void Memory<W>::trap(address_t page_addr, mmio_cb_t callback) {
	// This can probably be improved, but this will force-create
	// a page if it doesn't exist. At least this way the trap will
	// always work. Less surprises this way.
	auto &page = create_writable_pageno(page_number(page_addr));
	// Disabling caching will force the slow-path for the page,
	// and enables page traps when RISCV_DEBUG is enabled.
	page.attr.cacheable = false;
	page.set_trap(callback);
}

#endif // MEMORY_INLINE_PAGES_HPP
