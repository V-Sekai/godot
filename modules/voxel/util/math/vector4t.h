/**************************************************************************/
/*  vector4t.h                                                            */
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

#include "../errors.h"
#include "funcs.h"

namespace zylann {

// Template 4-dimensional vector. Only fields and standard operators.
// Math functions are separate to allow more unified overloading, and similarity with other math libraries such as
// shaders.
template <typename T>
struct Vector4T {
	static const unsigned int AXIS_COUNT = 4;

	union {
		struct {
			T x;
			T y;
			T z;
			T w;
		};
		T coords[4];
	};

	Vector4T() : x(0), y(0), z(0), w(0) {}

	// It is recommended to use `explicit` because otherwise it would open the door to plenty of implicit conversions
	// which would make many cases ambiguous.
	explicit Vector4T(T p_v) : x(p_v), y(p_v), z(p_v), w(p_v) {}

	Vector4T(T p_x, T p_y, T p_z, T p_w) : x(p_x), y(p_y), z(p_z), w(p_w) {}

	inline const T &operator[](const unsigned int p_axis) const {
#ifdef DEBUG_ENABLED
		ZN_ASSERT(p_axis < AXIS_COUNT);
#endif
		return coords[p_axis];
	}

	inline T &operator[](const unsigned int p_axis) {
#ifdef DEBUG_ENABLED
		ZN_ASSERT(p_axis < AXIS_COUNT);
#endif
		return coords[p_axis];
	}

	inline Vector4T operator+(const Vector4T &p_v) const {
		return Vector4T( //
				x + p_v.x, //
				y + p_v.y, //
				z + p_v.z, //
				w + p_v.w //
		);
	}

	inline Vector4T operator*(const Vector4T &p_v) const {
		return Vector4T( //
				x * p_v.x, //
				y * p_v.y, //
				z * p_v.z, //
				w * p_v.w //
		);
	}

	inline Vector4T operator*(const T p_scalar) const {
		return Vector4T( //
				x * p_scalar, //
				y * p_scalar, //
				z * p_scalar, //
				w * p_scalar //
		);
	}
};

} // namespace zylann
