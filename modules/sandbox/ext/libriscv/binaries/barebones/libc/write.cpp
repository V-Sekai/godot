/**************************************************************************/
/*  write.cpp                                                             */
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

#include <cassert>
#include <cstdio>
#include <cstring>
#include <include/syscall.hpp>

extern "C" long write(int fd, const void *data, size_t len) {
	return syscall(SYSCALL_WRITE, fd, (long)data, len);
}

extern "C" long _write_r(_reent *, int fd, const void *data, size_t len) {
	return syscall(SYSCALL_WRITE, fd, (long)data, len);
}

extern "C" int puts(const char *string) {
	const long len = __builtin_strlen(string);
	return write(0, string, len);
}

// buffered serial output
static char buffer[256];
static unsigned cnt = 0;

extern "C" int fflush(FILE *) {
	long ret = write(0, buffer, cnt);
	cnt = 0;
	return ret;
}

extern "C" void __print_putchr(void *, char c) {
	buffer[cnt++] = c;
	if (c == '\n' || cnt == sizeof(buffer)) {
		fflush(0);
	}
}
