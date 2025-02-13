/**************************************************************************/
/*  assert.cpp                                                            */
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

#include <cstdarg>
#include <cstdint>
#include <include/printf.hpp>
#include <include/syscall.hpp>

uint64_t __stack_chk_guard = 0x123456780C0A00FF;

extern "C"
		__attribute__((noreturn)) void
		panic(const char *reason) {
	tfp_printf("\n\n!!! PANIC !!!\n%s\n", reason);

	// the end
	syscall(SYSCALL_EXIT, -1);
	__builtin_unreachable();
}

extern "C" void abort() {
	panic("Abort called");
}

extern "C" void abort_message(const char *fmt, ...) {
	char buffer[2048];
	va_list arg;
	va_start(arg, fmt);
	int bytes = tfp_vsnprintf(buffer, sizeof(buffer), fmt, arg);
	(void)bytes;
	va_end(arg);
	panic(buffer);
}

extern "C" void __assert_func(
		const char *file,
		int line,
		const char *func,
		const char *failedexpr) {
	tfp_printf(
			"assertion \"%s\" failed: file \"%s\", line %d%s%s\n",
			failedexpr, file, line,
			func ? ", function: " : "", func ? func : "");
	abort();
}

extern "C" void __stack_chk_fail() {
	panic("Stack protector failed check");
}
