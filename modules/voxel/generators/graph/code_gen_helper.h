/**************************************************************************/
/*  code_gen_helper.h                                                     */
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

#pragma once

#include "../../util/containers/std_unordered_set.h"
#include "../../util/string/fwd_std_string.h"
#include "../../util/string/std_stringstream.h"

namespace zylann {

class CodeGenHelper {
public:
	CodeGenHelper(StdStringStream &main_ss, StdStringStream &lib_ss);

	void indent();
	void dedent();

	void add(const char *s, unsigned int len);
	void add(const char *s);
	void add(FwdConstStdString s);
	void add(double x);
	void add(float x);
	void add(int x);

	template <typename T>
	void add_format(const char *fmt, const T &a0) {
		fmt = _add_format(fmt, a0);
		if (*fmt != '\0') {
			add(fmt);
		}
	}

	template <typename T0, typename... TN>
	void add_format(const char *fmt, const T0 &a0, const TN &...an) {
		fmt = _add_format(fmt, a0);
		add_format(fmt, an...);
	}

	void require_lib_code(const char *lib_name, const char *code);
	void require_lib_code(const char *lib_name, const char **code);
	void generate_var_name(FwdMutableStdString out_var_name);

	void print(FwdMutableStdString output);

private:
	template <typename T>
	const char *_add_format(const char *fmt, const T &a0) {
		if (*fmt == '\0') {
			return fmt;
		}
		const char *c = fmt;
		while (*c != '\0') {
			if (*c == '{') {
				++c;
				if (*c == '}') {
					add(fmt, c - fmt - 1);
					add(a0);
					return c + 1;
				}
			}
			++c;
		}
		add(fmt, c - fmt);
		return c;
	}

	StdStringStream &_main_ss;
	StdStringStream &_lib_ss;
	unsigned int _indent_level = 0;
	unsigned int _next_var_name_id = 0;
	StdUnorderedSet<const char *> _included_libs;
	bool _newline = true;
};

} // namespace zylann
