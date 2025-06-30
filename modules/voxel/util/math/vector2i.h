/**************************************************************************/
/*  vector2i.h                                                            */
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
#include "../godot/core/vector2i.h"
#include "../godot/macros.h"
#include "../hash_funcs.h"
#include "../macros.h"
#include "../string/std_stringstream.h"
#include "funcs.h"
#include <functional> // For std::hash

ZN_GODOT_NAMESPACE_BEGIN

inline Vector2i operator&(const Vector2i &a, int b) {
	return Vector2i(a.x & b, a.y & b);
}

ZN_GODOT_NAMESPACE_END

namespace zylann {

namespace Vector2iUtil {

inline Vector2i create(int xy) {
	return Vector2i(xy, xy);
}

inline int64_t get_area(const Vector2i v) {
#ifdef DEBUG_ENABLED
	ZN_ASSERT_RETURN_V(v.x >= 0 && v.y >= 0, 0);
#endif
	return v.x * v.y;
}

inline unsigned int get_yx_index(const Vector2i v, const Vector2i area_size) {
	return v.x + v.y * area_size.x;
}

} // namespace Vector2iUtil

namespace math {

inline Vector2i floordiv(const Vector2i v, const Vector2i d) {
	return Vector2i(math::floordiv(v.x, d.x), math::floordiv(v.y, d.y));
}

inline Vector2i floordiv(const Vector2i v, const int d) {
	return Vector2i(math::floordiv(v.x, d), math::floordiv(v.y, d));
}

inline Vector2i ceildiv(const Vector2i v, const int d) {
	return Vector2i(math::ceildiv(v.x, d), math::ceildiv(v.y, d));
}

inline Vector2i ceildiv(const Vector2i v, const Vector2i d) {
	return Vector2i(math::ceildiv(v.x, d.x), math::ceildiv(v.y, d.y));
}

inline int chebyshev_distance(const Vector2i &a, const Vector2i &b) {
	// In Chebyshev metric, points on the sides of a square are all equidistant to its center
	return math::max(Math::abs(a.x - b.x), Math::abs(a.y - b.y));
}

inline Vector2i min(const Vector2i a, const Vector2i b) {
	return Vector2i(min(a.x, b.x), min(a.y, b.y));
}

} // namespace math

StdStringStream &operator<<(StdStringStream &ss, const Vector2i &v);

} // namespace zylann

// For STL
namespace std {
template <>
struct hash<Vector2i> {
	size_t operator()(const Vector2i &v) const {
		// TODO This is 32-bit, would it be better if it was 64?
		uint32_t h = zylann::hash_murmur3_one_32(v.x);
		h = zylann::hash_murmur3_one_32(v.y, h);
		return zylann::hash_fmix32(h);
	}
};
} // namespace std
