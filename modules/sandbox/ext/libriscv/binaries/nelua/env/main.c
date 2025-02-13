/**************************************************************************/
/*  main.c                                                                */
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

#include <stddef.h>

#define ECALL_WRITE 64
#define ECALL_EXIT 93

int my_write(int fd, const char *buffer, size_t size) {
	register int a0 __asm__("a0") = fd;
	register const char *a1 __asm__("a1") = buffer;
	register size_t a2 __asm__("a2") = size;
	register long syscall_id __asm__("a7") = ECALL_WRITE;

	__asm__ volatile("ecall"
			: "+r"(a0)
			: "m"(*(const char(*)[size])a1), "r"(a2),
			"r"(syscall_id));
	return a0;
}

int my_exit(int status) {
	register int a0 __asm__("a0") = status;
	register long syscall_id __asm__("a7") = ECALL_EXIT;

	__asm__ volatile("ecall" : : "r"(a0), "r"(syscall_id));
	__builtin_unreachable();
}
