/**************************************************************************/
/*  machine_defaults.cpp                                                  */
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

/**
 * Some default implementations of OS-specific I/O routines
 * stdout: Used by write/writev system calls
 * stdin:  Used by read/readv system calls
 * rdtime: Used by the RDTIME/RDTIMEH instructions
 **/
#include "internal_common.hpp"
#include "machine.hpp"

#include <chrono>
extern "C" {
#ifdef _WIN32
int write(int fd, const void *buf, unsigned count);
#else
ssize_t write(int fd, const void *buf, size_t count);
#endif
}

namespace riscv {
// Default: Stdout allowed
template <int W>
void Machine<W>::default_printer(const Machine<W> &, const char *buffer, size_t len) {
	std::ignore = ::write(1, buffer, len);
}
// Default: Stdin *NOT* allowed
template <int W>
long Machine<W>::default_stdin(const Machine<W> &, char * /*buffer*/, size_t /*len*/) {
	return 0;
}

// Default: RDTIME produces monotonic time with *microsecond*-granularity
template <int W>
uint64_t Machine<W>::default_rdtime(const Machine<W> &machine) {
#ifdef __wasm__
	return 0;
#else
	auto now = std::chrono::steady_clock::now();
	auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
	if (!(machine.has_file_descriptors() && machine.fds().proxy_mode))
		micros &= ANTI_FINGERPRINTING_MASK_MICROS();
	return micros;
#endif
}

#ifndef __GNUG__ /* Workaround for GCC bug */
INSTANTIATE_32_IF_ENABLED(Machine);
INSTANTIATE_64_IF_ENABLED(Machine);
INSTANTIATE_128_IF_ENABLED(Machine);
#endif
} //namespace riscv
