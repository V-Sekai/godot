/**************************************************************************/
/*  dstack.cpp                                                            */
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

#include "dstack.h"
#include "containers/fixed_array.h"
#include "string/format.h"

#include <string>

namespace zylann {
namespace dstack {

struct Stack {
	FixedArray<Frame, 64> frames;
	unsigned int count = 0;
};

Stack &get_tls_stack() {
	thread_local Stack tls_stack;
	return tls_stack;
}

void push(const char *file, unsigned int line, const char *function) {
	Stack &stack = get_tls_stack();
	stack.frames[stack.count] = { file, function, line };
	++stack.count;
}

void pop() {
	Stack &stack = get_tls_stack();
	--stack.count;
}

Info::Info() {
	const Stack &stack = get_tls_stack();
	_frames.resize(stack.count);
	for (unsigned int i = 0; i < stack.count; ++i) {
		_frames[i] = stack.frames[i];
	}
}

void Info::to_string(FwdMutableStdString s) const {
	for (unsigned int i = 0; i < _frames.size(); ++i) {
		const Frame &frame = _frames[i];
		s.s += format("{} ({}:{})\n", frame.function, frame.file, frame.line);
	}
}

} // namespace dstack
} // namespace zylann
