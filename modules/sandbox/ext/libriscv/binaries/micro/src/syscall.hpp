/**************************************************************************/
/*  syscall.hpp                                                           */
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

#ifndef SYSCALL_HPP
#define SYSCALL_HPP

#ifndef SYSCALL_WRITE
#define SYSCALL_WRITE 64
#endif
#ifndef SYSCALL_EXIT
#define SYSCALL_EXIT 93
#endif

extern "C" {
__attribute__((noreturn)) void _exit(int);
long write(int fd, const void *, unsigned long);
}

inline long syscall(long n) {
	register long a0 asm("a0");
	register long syscall_id asm("a7") = n;

	asm volatile("scall" : "=r"(a0) : "r"(syscall_id));

	return a0;
}

inline long syscall(long n, long arg0) {
	register long a0 asm("a0") = arg0;
	register long syscall_id asm("a7") = n;

	asm volatile("scall" : "+r"(a0) : "r"(syscall_id));

	return a0;
}

inline long syscall(long n, long arg0, long arg1) {
	register long a0 asm("a0") = arg0;
	register long a1 asm("a1") = arg1;
	register long syscall_id asm("a7") = n;

	asm volatile("scall" : "+r"(a0) : "r"(a1), "r"(syscall_id));

	return a0;
}

inline long syscall(long n, long arg0, long arg1, long arg2) {
	register long a0 asm("a0") = arg0;
	register long a1 asm("a1") = arg1;
	register long a2 asm("a2") = arg2;
	register long syscall_id asm("a7") = n;

	asm volatile("scall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(syscall_id));

	return a0;
}

#endif // SYSCALL_HPP
