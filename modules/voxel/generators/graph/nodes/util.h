/**************************************************************************/
/*  util.h                                                                */
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

#include "../voxel_graph_runtime.h"

namespace zylann::voxel::pg {

template <typename F>
inline void do_monop(pg::Runtime::ProcessBufferContext &ctx, F f) {
	const Runtime::Buffer &a = ctx.get_input(0);
	Runtime::Buffer &out = ctx.get_output(0);
	if (a.is_constant) {
		// Normally this case should have been optimized out at compile-time
		const float v = f(a.constant_value);
		for (uint32_t i = 0; i < a.size; ++i) {
			out.data[i] = v;
		}
	} else {
		for (uint32_t i = 0; i < a.size; ++i) {
			out.data[i] = f(a.data[i]);
		}
	}
}

template <typename F>
inline void do_binop(pg::Runtime::ProcessBufferContext &ctx, F f) {
	const Runtime::Buffer &a = ctx.get_input(0);
	const Runtime::Buffer &b = ctx.get_input(1);
	Runtime::Buffer &out = ctx.get_output(0);
	const uint32_t buffer_size = out.size;

	if (a.is_constant || b.is_constant) {
		if (!b.is_constant) {
			const float c = a.constant_value;
			const float *v = b.data;
			for (uint32_t i = 0; i < buffer_size; ++i) {
				out.data[i] = f(c, v[i]);
			}

		} else if (!a.is_constant) {
			const float c = b.constant_value;
			const float *v = a.data;
			for (uint32_t i = 0; i < buffer_size; ++i) {
				out.data[i] = f(v[i], c);
			}

		} else {
			// Normally this case should have been optimized out at compile-time
			const float c = f(a.constant_value, b.constant_value);
			for (uint32_t i = 0; i < buffer_size; ++i) {
				out.data[i] = c;
			}
		}

	} else {
		for (uint32_t i = 0; i < buffer_size; ++i) {
			out.data[i] = f(a.data[i], b.data[i]);
		}
	}
}

} // namespace zylann::voxel::pg
