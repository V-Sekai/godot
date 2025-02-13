/**************************************************************************/
/*  libcxx.cpp                                                            */
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

#include <cstddef>
extern "C"
		__attribute__((noreturn)) void
		abort_message(const char *fmt, ...);

#ifndef USE_NEWLIB
#ifndef __EXCEPTIONS
// exception stubs for various C++ containers
namespace std {
void __throw_bad_alloc() {
	abort_message("exception: bad_alloc thrown\n");
}
void __throw_length_error(char const *) {
	abort_message("C++ length error exception");
}
void __throw_bad_array_new_length() {
	abort_message("C++ bad array new length exception");
}
void __throw_logic_error(char const *) {
	abort_message("C++ length error exception");
}
void __throw_out_of_range_fmt(char const *, ...) {
	abort_message("C++ out-of-range exception");
}
void __throw_bad_function_call() {
	abort_message("Bad std::function call!");
}
} //namespace std
#endif

extern "C" int __cxa_atexit(void (*func)(void *), void * /*arg*/, void * /*dso_handle*/) {
	(void)func;
	return 0;
}
#endif
