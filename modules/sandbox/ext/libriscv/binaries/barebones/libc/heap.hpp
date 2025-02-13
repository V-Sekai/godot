/**************************************************************************/
/*  heap.hpp                                                              */
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

#ifndef HEAP_HPP
#define HEAP_HPP
/**
 * Accelerated heap using syscalls
 *
 **/
#include <cstddef>
#include <include/syscall.hpp>

struct MemInfo {
	size_t bytes_free;
	size_t bytes_used;
	size_t chunks_used;
};

inline void *sys_malloc(std::size_t size) {
	register void *ret asm("a0");
	register size_t a0 asm("a0") = size;
	register long syscall_id asm("a7") = SYSCALL_MALLOC;

	asm volatile("ecall"
			: "=m"(*(char(*)[size])ret), "=r"(ret)
			: "r"(a0), "r"(syscall_id));
	return ret;
}
inline void *sys_calloc(size_t, size_t);
inline void *sys_realloc(void *, size_t);
inline void sys_free(void *ptr) {
	register void *a0 asm("a0") = ptr;
	register long syscall_id asm("a7") = SYSCALL_FREE;

	asm volatile("ecall"
			:
			: "r"(a0), "r"(syscall_id));
}

inline int sys_meminfo(void *ptr, size_t len) {
	return psyscall(SYSCALL_MEMINFO, ptr, len);
}

#endif // HEAP_HPP
