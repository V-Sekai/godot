/**************************************************************************/
/*  linux_va_fib.c                                                        */
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

#define uintptr_t __UINTPTR_TYPE__
typedef long (*fib_func)(long, long, long);

static long syscall(long n, long arg0);
static long syscall3(long n, long arg0, long arg1, long arg2);

static void copy(uintptr_t dst, const void *src, unsigned len) {
	for (unsigned i = 0; i < len; i++)
		((char *)dst)[i] = ((const char *)src)[i];
}

static long fib(long n, long acc, long prev) {
	if (n == 0)
		return acc;
	else
		return fib(n - 1, prev + acc, acc);
}
static void fib_end() {}

int main() {
	const uintptr_t DST = 0xF0000000;
	copy(DST, &fib, (char *)&fib_end - (char *)&fib);
	// mprotect +execute
	syscall3(226, DST, 0x1000, 0x4);

	const volatile long n = 256000000;
	fib_func other_fib = (fib_func)DST;
	// exit(...)
	syscall(93, other_fib(n, 0, 1));
}

long syscall(long n, long arg0) {
	register long a0 asm("a0") = arg0;
	register long syscall_id asm("a7") = n;

	__asm__ volatile("scall" : "+r"(a0) : "r"(syscall_id));

	return a0;
}
long syscall3(long n, long arg0, long arg1, long arg2) {
	register long a0 asm("a0") = arg0;
	register long a1 asm("a1") = arg1;
	register long a2 asm("a2") = arg2;
	register long syscall_id asm("a7") = n;

	__asm__ volatile("scall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(syscall_id));

	return a0;
}
