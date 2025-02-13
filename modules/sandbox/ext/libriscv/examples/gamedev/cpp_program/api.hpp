/**************************************************************************/
/*  api.hpp                                                               */
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

#ifndef API_HPP
#define API_HPP
#include <cstddef>
#include <new>
#define PUBLIC(x) extern "C" __attribute__((used, retain)) x

#define NATIVE_MEM_FUNCATTR /* */
#define NATIVE_SYSCALLS_BASE 490 /* libc starts at 490 */

#define SYSCALL_MALLOC (NATIVE_SYSCALLS_BASE + 0)
#define SYSCALL_CALLOC (NATIVE_SYSCALLS_BASE + 1)
#define SYSCALL_REALLOC (NATIVE_SYSCALLS_BASE + 2)
#define SYSCALL_FREE (NATIVE_SYSCALLS_BASE + 3)
#define SYSCALL_MEMINFO (NATIVE_SYSCALLS_BASE + 4)

#define SYSCALL_MEMCPY (NATIVE_SYSCALLS_BASE + 5)
#define SYSCALL_MEMSET (NATIVE_SYSCALLS_BASE + 6)
#define SYSCALL_MEMMOVE (NATIVE_SYSCALLS_BASE + 7)
#define SYSCALL_MEMCMP (NATIVE_SYSCALLS_BASE + 8)

#define SYSCALL_STRLEN (NATIVE_SYSCALLS_BASE + 10)
#define SYSCALL_STRCMP (NATIVE_SYSCALLS_BASE + 11)

#define SYSCALL_BACKTRACE (NATIVE_SYSCALLS_BASE + 19)

extern "C" __attribute__((noreturn)) void fast_exit(int);

inline void *sys_malloc(std::size_t size) {
	register void *ret asm("a0");
	register size_t a0 asm("a0") = size;
	register long syscall_id asm("a7") = SYSCALL_MALLOC;

	asm volatile("ecall"
			: "=m"(*(char(*)[size])ret), "=r"(ret)
			: "r"(a0), "r"(syscall_id));
	return ret;
}
inline void sys_free(void *ptr) {
	register void *a0 asm("a0") = ptr;
	register long syscall_id asm("a7") = SYSCALL_FREE;

	asm volatile("ecall"
			:
			: "r"(a0), "r"(syscall_id));
}

inline void *operator new(std::size_t size) {
	return sys_malloc(size);
}
inline void *operator new[](std::size_t size) {
	return sys_malloc(size);
}
inline void operator delete(void *ptr) {
	sys_free(ptr);
}
inline void operator delete[](void *ptr) {
	sys_free(ptr);
}
// C++14 sized deallocation
inline void operator delete(void *ptr, std::size_t) {
	sys_free(ptr);
}
inline void operator delete[](void *ptr, std::size_t) {
	sys_free(ptr);
}

#endif // API_HPP
