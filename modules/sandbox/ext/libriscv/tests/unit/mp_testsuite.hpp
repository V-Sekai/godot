/**************************************************************************/
/*  mp_testsuite.hpp                                                      */
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

#ifndef MP_TESTSUITE_HPP
#define MP_TESTSUITE_HPP
#include <array>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
static constexpr size_t MP_WORKERS = 4;
static constexpr size_t MP_STACK_SIZE = 512 * 1024u;
extern "C" long sys_write(const char *);

static void print(const char *format, ...) {
	char buffer[256];
	va_list args;
	va_start(args, format);
	vsnprintf(buffer, 256, format, args);
	sys_write(buffer);
	va_end(args);
}

template <size_t SIZE>
struct MultiprocessWork {
	unsigned workers = 1;
	std::array<float, SIZE> data_a;
	std::array<float, SIZE> data_b;
	float result[16] = { 0 };
	int counter = 0;

	inline size_t work_size() const noexcept {
		return SIZE / workers;
	}
	float final_sum() const noexcept {
		float sum = 0.0f;
		for (size_t i = 0; i < this->workers; i++)
			sum += this->result[i];
		return sum;
	}
};
static constexpr size_t WORK_SIZE = 16384;
static MultiprocessWork<WORK_SIZE> mp_work;

template <size_t SIZE>
static void initialize_work(MultiprocessWork<SIZE> &work) {
	for (size_t i = 0; i < SIZE; i++) {
		work.data_a[i] = 1.0;
		work.data_b[i] = 1.0;
	}
}

template <size_t SIZE>
static void multiprocessing_function(int cpu, void *vdata) {
	auto &work = *(MultiprocessWork<SIZE> *)vdata;
	const size_t start = (cpu + 0) * work.work_size();
	const size_t end = (cpu + 1) * work.work_size();

	float sum = 0.0f;
	for (size_t i = start; i < end; i++) {
		sum += work.data_a[i] * work.data_b[i];
	}

	work.result[cpu] = sum;
	__sync_fetch_and_add(&work.counter, 1);
}

extern "C" unsigned multiprocess(unsigned int);
extern "C" long sys_multiprocess_wait();

#if 0
inline unsigned multiprocess(unsigned cpus)
{
	register unsigned a0 asm("a0") = cpus;
	register int     sid asm("a7") = 1; // multiprocess_fork

	asm volatile ("ecall" : "+r"(a0) : "r"(sid) : "memory");
	return a0;
}
#else
asm(".global multiprocess\n"
	"multiprocess:\n"
	"	li a1, 0\n"
	"	li a2, 0\n"
	"	li a7, 1\n"
	"   ecall\n"
	"	ret\n");
#endif

inline long multiprocess_wait() {
	register unsigned a0 asm("a0");
	register int sid asm("a7") = 2;

	asm volatile("ecall" : "=r"(a0) : "r"(sid) : "memory");
	return a0;
}
inline long sys_write(const char *buf) {
	register const char *a0_in asm("a0") = buf;
	register unsigned a0_out asm("a0");
	register int sid asm("a7") = 10;

	asm volatile("ecall" : "=r"(a0_out) : "r"(a0_in), "r"(sid) : "memory");
	return a0_out;
}

#endif // MP_TESTSUITE_HPP
