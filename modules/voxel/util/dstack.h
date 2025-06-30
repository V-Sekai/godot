/**************************************************************************/
/*  dstack.h                                                              */
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

#include "containers/std_vector.h"
#include "string/fwd_std_string.h"

#ifdef DEBUG_ENABLED
#define ZN_DSTACK_ENABLED
#endif

#ifdef ZN_DSTACK_ENABLED
// Put this macro on top of each function you want to track in debug stack traces.
#define ZN_DSTACK() zylann::dstack::Scope dstack_scope_##__LINE__(__FILE__, __LINE__, __FUNCTION__)
#else
#define ZN_DSTACK()
#endif

namespace zylann {
namespace dstack {

void push(const char *file, unsigned int line, const char *fname);
void pop();

struct Scope {
	Scope(const char *file, unsigned int line, const char *function) {
		push(file, line, function);
	}
	~Scope() {
		pop();
	}
};

struct Frame {
	const char *file = nullptr;
	const char *function = nullptr;
	unsigned int line = 0;
};

struct Info {
public:
	// Constructs a copy of the current stack gathered so far from ZN_DSTACK() calls
	Info();
	void to_string(FwdMutableStdString s) const;

private:
	StdVector<Frame> _frames;
};

} // namespace dstack
} // namespace zylann
