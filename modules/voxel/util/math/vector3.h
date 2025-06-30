/**************************************************************************/
/*  vector3.h                                                             */
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

#include "../godot/core/vector3.h"
#include "../string/std_stringstream.h"
#include "funcs.h"

// 3-dimensional vector which components are either 32-bit float or 64-bit float depending on how Godot was compiled.
// This is the type to use for interoperating with Godot.

namespace zylann::godot::Vector3Utility {

inline Vector3 splat(const real_t v) {
	return Vector3(v, v, v);
}

} // namespace zylann::godot::Vector3Utility

namespace zylann::math {

inline Vector3 fract(const Vector3 &p) {
	return Vector3(fract(p.x), fract(p.y), fract(p.z));
}

inline Vector3 floor(const Vector3 &p) {
	return p.floor();
}

inline Vector3 ceil(const Vector3 &p) {
	return p.ceil();
}

inline Vector3 round(const Vector3 &p) {
	return p.round();
}

inline bool is_valid_size(const Vector3 &s) {
	return s.x >= 0 && s.y >= 0 && s.z >= 0;
}

inline bool has_nan(const Vector3 &v) {
	return is_nan(v.x) || is_nan(v.y) || is_nan(v.z);
}

inline bool is_normalized(const Vector3 &v) {
	return v.is_normalized();
}

inline Vector3 lerp(const Vector3 a, const Vector3 b, const Vector3 alpha) {
	return Vector3(Math::lerp(a.x, b.x, alpha.x), Math::lerp(a.y, b.y, alpha.y), Math::lerp(a.z, b.z, alpha.z));
}

inline Vector3 wrapf(const Vector3 v, real_t d) {
	return Math::is_zero_approx(d) ? Vector3() : (v - (d * floor(v / d)));
}

inline Vector3 min(const Vector3 a, const Vector3 b) {
	return Vector3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

inline Vector3 max(const Vector3 a, const Vector3 b) {
	return Vector3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

inline Vector3 clamp(const Vector3 a, const Vector3 minv, const Vector3 maxv) {
	return Vector3(clamp(a.x, minv.x, maxv.x), clamp(a.y, minv.y, maxv.y), clamp(a.z, minv.z, maxv.z));
}

inline Vector3 abs(const Vector3 &v) {
	return Vector3(Math::abs(v.x), Math::abs(v.y), Math::abs(v.z));
}

inline real_t dot(const Vector3 &a, const Vector3 &b) {
	return a.dot(b);
}

inline real_t length_squared(const Vector3 &a) {
	return a.length_squared();
}

inline real_t get_largest_coord(Vector3 v) {
	return math::max(math::max(v.x, v.y), v.z);
}

} // namespace zylann::math

namespace zylann {
StdStringStream &operator<<(StdStringStream &ss, const Vector3 &v);
}
