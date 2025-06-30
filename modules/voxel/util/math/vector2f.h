/**************************************************************************/
/*  vector2f.h                                                            */
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
#include "vector2t.h"

namespace zylann {

// 32-bit float precision 2D vector.
// Because Godot's `Vector2` uses `real_t`, so when `real_t` is `double` it forces some things to use double-precision
// vectors while they dont need that amount of precision.
typedef Vector2T<float> Vector2f;

namespace math {

inline Vector2f floor(const Vector2f a) {
	return Vector2f(Math::floor(a.x), Math::floor(a.y));
}

inline Vector2f lerp(const Vector2f a, const Vector2f b, const float t) {
	return Vector2f(Math::lerp(a.x, b.x, t), Math::lerp(a.y, b.y, t));
}

inline bool is_equal_approx(const Vector2f a, const Vector2f b) {
	return Math::is_equal_approx(a.x, b.x) && Math::is_equal_approx(a.y, b.y);
}

} // namespace math

StdStringStream &operator<<(StdStringStream &ss, const Vector2f &v);

} // namespace zylann
