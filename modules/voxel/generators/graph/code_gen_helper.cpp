/**************************************************************************/
/*  code_gen_helper.cpp                                                   */
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

#include "code_gen_helper.h"
#include "../../util/containers/fixed_array.h"
#include "../../util/errors.h"
#include "../../util/string/format.h"

#include <cstring>
#include <sstream>

namespace zylann {

CodeGenHelper::CodeGenHelper(StdStringStream &main_ss, StdStringStream &lib_ss) : _main_ss(main_ss), _lib_ss(lib_ss) {}

void CodeGenHelper::indent() {
	++_indent_level;
}

void CodeGenHelper::dedent() {
	ZN_ASSERT(_indent_level > 0);
	--_indent_level;
}

void CodeGenHelper::add(const char *s, unsigned int len) {
	for (unsigned int i = 0; i < len; ++i) {
		const char c = s[i];
		if (_newline) {
			for (unsigned int j = 0; j < _indent_level; ++j) {
				_main_ss << "    ";
			}
			_newline = false;
		}
		_main_ss << c;
		if (c == '\n') {
			_newline = true;
		}
	}
}

void CodeGenHelper::add(const char *s) {
	add(s, strlen(s));
}

void CodeGenHelper::add(FwdConstStdString s) {
	add(s.s.c_str(), s.s.size());
}

void CodeGenHelper::add(float x) {
	FixedArray<char, 32> buffer;
	// Godot shaders want float constants to be explicit
	const unsigned int decimals = float(int(x)) == x ? 1 : 10;
	const unsigned int len = snprintf(buffer.data(), buffer.size(), "%.*f", decimals, x);
	add(buffer.data(), len);
}

void CodeGenHelper::add(double x) {
	FixedArray<char, 32> buffer;
	// Godot shaders want float constants to be explicit
	const unsigned int decimals = double(int(x)) == x ? 1 : 16;
	const unsigned int len = snprintf(buffer.data(), buffer.size(), "%.*lf", decimals, x);
	add(buffer.data(), len);
}

void CodeGenHelper::add(int x) {
	FixedArray<char, 32> buffer;
	const unsigned int len = snprintf(buffer.data(), buffer.size(), "%i", x);
	add(buffer.data(), len);
}

void CodeGenHelper::require_lib_code(const char *lib_name, const char *code) {
	auto p = _included_libs.insert(lib_name);
	if (p.second) {
		_lib_ss << "\n\n";
		_lib_ss << code;
		_lib_ss << "\n\n";
	}
}

// Some code can be too big to fit in a single literal depending on the compiler,
// so an option is to provide it as a zero-terminated array of string literals
void CodeGenHelper::require_lib_code(const char *lib_name, const char **code) {
	auto p = _included_libs.insert(lib_name);
	if (p.second) {
		_lib_ss << "\n\n";
		while (*code != 0) {
			_lib_ss << *code;
			++code;
		}
		_lib_ss << "\n\n";
	}
}

void CodeGenHelper::generate_var_name(FwdMutableStdString out_var_name) {
	const StdString s = format("v{}", _next_var_name_id);
	++_next_var_name_id;
	out_var_name.s = s;
}

void CodeGenHelper::print(FwdMutableStdString output) {
	output.s = _lib_ss.str() + _main_ss.str();
}

} // namespace zylann
