/**************************************************************************/
/*  vector3f.h                                                            */
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
#include "../string/std_stringstream.h"
#include "vector3t.h"

namespace zylann {

// 32-bit float precision 3D vector.
// Because Godot's `Vector3` uses `real_t`, so when `real_t` is `double` it forces some things to use double-precision
// vectors while they dont need that amount of precision. This is also a problem for some third-party libraries
// that do not support `double` as a result.
typedef Vector3T<float> Vector3f;

namespace math {

inline Vector3f floor(const Vector3f a) {
	return Vector3f(Math::floor(a.x), Math::floor(a.y), Math::floor(a.z));
}

inline Vector3f ceil(const Vector3f a) {
	return Vector3f(Math::ceil(a.x), Math::ceil(a.y), Math::ceil(a.z));
}

inline Vector3f lerp(const Vector3f a, const Vector3f b, const float t) {
	return Vector3f(Math::lerp(a.x, b.x, t), Math::lerp(a.y, b.y, t), Math::lerp(a.z, b.z, t));
}

inline bool has_nan(const Vector3f &v) {
	return is_nan(v.x) || is_nan(v.y) || is_nan(v.z);
}

inline Vector3f normalized(const Vector3f &v) {
	const float lengthsq = length_squared(v);
	if (lengthsq == 0) {
		return Vector3f();
	} else {
		const float length = Math::sqrt(lengthsq);
		return v / length;
	}
}

inline Vector3f normalized(Vector3f v, float &out_length) {
	const float lengthsq = length_squared(v);
	if (lengthsq == 0) {
		out_length = 0.f;
		return Vector3f();
	} else {
		const float length = Math::sqrt(lengthsq);
		out_length = length;
		return v / length;
	}
}

inline bool is_normalized(const Vector3f &v) {
	// use length_squared() instead of length() to avoid sqrt(), makes it more stringent.
	return Math::is_equal_approx(length_squared(v), 1, float(UNIT_EPSILON));
}

inline bool is_equal_approx(const Vector3f a, const Vector3f b) {
	return Math::is_equal_approx(a.x, b.x) && Math::is_equal_approx(a.y, b.y) && Math::is_equal_approx(a.z, b.z);
}

} // namespace math

StdStringStream &operator<<(StdStringStream &ss, const Vector3f &v);

} // namespace zylann
