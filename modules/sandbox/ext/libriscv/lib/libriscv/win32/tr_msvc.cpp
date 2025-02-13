/**************************************************************************/
/*  tr_msvc.cpp                                                           */
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

#include "../common.hpp"

#include "dlfcn.h"
#include <cstring>

namespace riscv {
void *compile(const std::string &code, int arch, const std::string &cflags, const std::string &outfile) {
	(void)arch;
	(void)cflags;
	(void)outfile;
	(void)code;

	return nullptr;
}

bool mingw_compile(const std::string &code, int arch, const std::string &cflags,
		const std::string &outfile, const MachineTranslationCrossOptions &cross_options) {
	(void)arch;
	(void)cflags;
	(void)outfile;
	(void)code;
	(void)cross_options;

	return false;
}

void *dylib_lookup(void *dylib, const char *symbol, bool) {
	return dlsym(dylib, symbol);
}

void dylib_close(void *dylib, bool) {
	dlclose(dylib);
}
} //namespace riscv
