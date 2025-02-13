/**************************************************************************/
/*  tr_tcc.cpp                                                            */
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

#include "common.hpp"

#include <libtcc.h>
#include <libtcc1.h> // libtcc1_c and libtcc1_c_len
#include <cstring>

namespace riscv {
void *libtcc_compile(const std::string &code,
		int, const std::unordered_map<std::string, std::string> &cflags, const std::string &libtcc1) {
	(void)libtcc1;

	TCCState *state = tcc_new();
	if (!state)
		return nullptr;

	tcc_set_output_type(state, TCC_OUTPUT_MEMORY);

	for (const auto &pair : cflags) {
		tcc_define_symbol(state, pair.first.c_str(), pair.second.c_str());
	}

	tcc_define_symbol(state, "ARCH", "HOST_UNKNOWN");
	tcc_set_options(state, "-std=c99 -O2 -nostdlib");

#if defined(_WIN32)
	// Look for some headers in the win32 directory
	tcc_add_include_path(state, "win32");
#elif defined(__linux__) && defined(__x86_64__)
	tcc_add_include_path(state, "/usr/include/x86_64-linux-gnu");
#elif defined(__linux__) && defined(__i386__)
	tcc_add_include_path(state, "/usr/include/i386-linux-gnu");
#elif defined(__linux__) && defined(__aarch64__)
	tcc_add_include_path(state, "/usr/include/aarch64-linux-gnu");
#elif defined(__linux__) && defined(__riscv)
	tcc_add_include_path(state, "/usr/include/riscv64-linux-gnu");
#endif

	tcc_add_symbol(state, "memset", (void *)memset);
	tcc_add_symbol(state, "memcpy", (void *)memcpy);
	tcc_add_symbol(state, "memcmp", (void *)memcmp);
	tcc_add_symbol(state, "memmove", (void *)memmove);

	std::string code1 = std::string((const char *)lib_libtcc1_c, lib_libtcc1_c_len) + code;

	if (tcc_compile_string(state, code1.c_str()) < 0) {
		tcc_delete(state);
		return nullptr;
	}

#if defined(TCC_RELOCATE_AUTO)
	if (tcc_relocate(state, TCC_RELOCATE_AUTO) < 0)
#else
	if (tcc_relocate(state) < 0)
#endif
	{
		tcc_delete(state);
		return nullptr;
	}

	return state;
}

void *tcc_lookup(void *state, const char *symbol) {
	return tcc_get_symbol((TCCState *)state, symbol);
}

void tcc_close(void *state) {
	tcc_delete((TCCState *)state);
}
} //namespace riscv
