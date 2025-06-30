/**************************************************************************/
/*  quaternionf.h                                                         */
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

#include "funcs.h"

namespace zylann {

// 32-bit float Quaternion.
struct Quaternionf {
	union {
		struct {
			float x;
			float y;
			float z;
			float w;
		};
		float components[4] = { 0, 0, 0, 1.0 };
	};

	inline Quaternionf() {}

	inline Quaternionf(float p_x, float p_y, float p_z, float p_w) : x(p_x), y(p_y), z(p_z), w(p_w) {}

	inline Quaternionf operator/(const Quaternionf &q) const {
		return Quaternionf(x / q.x, y / q.y, z / q.z, w / q.w);
	}

	inline Quaternionf operator/(float d) const {
		return Quaternionf(x / d, y / d, z / d, w / d);
	}
};

namespace math {

inline float length_squared(const Quaternionf &q) {
	return q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
}

inline float length(const Quaternionf &q) {
	return Math::sqrt(length_squared(q));
}

inline Quaternionf normalized(const Quaternionf &q) {
	return q / length(q);
}

} // namespace math
} // namespace zylann
