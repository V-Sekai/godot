/**************************************************************************/
/*  multiprocessing.hpp                                                   */
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

#ifndef MULTIPROCESSING_HPP
#define MULTIPROCESSING_HPP
#include "common.hpp"

#ifdef RISCV_MULTIPROCESS
#include "util/threadpool.h"
#else
#include <cstddef>
#include <cstdint>
#endif

namespace riscv {

template <int W>
struct Multiprocessing {
	using failure_bits_t = uint32_t;

	Multiprocessing(size_t);
#ifdef RISCV_MULTIPROCESS
	void async_work(std::vector<std::function<void()>> &&wrk);
	failure_bits_t wait();
	bool is_multiprocessing() const noexcept { return this->processing; }
	size_t workers() const noexcept { return m_threadpool.get_pool_size(); }

	ThreadPool m_threadpool;
	std::mutex m_lock;
	bool processing = false;
	failure_bits_t failures = 0; // Bitmap of failed vCPU tasks
	static constexpr bool shared_page_faults = true;
	static constexpr bool shared_read_faults = true;
#else
	bool is_multiprocessing() const noexcept { return false; }
	size_t workers() const noexcept { return 0u; }
#endif
};

} //namespace riscv

#endif // MULTIPROCESSING_HPP
