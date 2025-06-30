/**************************************************************************/
/*  box3f.h                                                               */
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

#include "vector3t.h"
#include <type_traits>

namespace zylann {

// Axis-aligned 3D box using floating point coordinates.
template <typename T>
class Box3fT {
public:
	static_assert(std::is_floating_point<T>::value);

	Vector3T<T> min;
	Vector3T<T> max;

	static Box3fT<T> from_min_max(const Vector3T<T> p_min, const Vector3T<T> p_max) {
		return { p_min, p_max };
	}

	static Box3fT<T> from_min_size(const Vector3T<T> p_min, const Vector3T<T> size) {
		return { p_min, p_min + size };
	}

	static Box3fT<T> from_center_half_size(const Vector3T<T> center, const Vector3T<T> hs) {
		return { center - hs, center + hs };
	}

	bool contains(const Vector3T<T> &p_point) const {
		if (p_point.x < min.x) {
			return false;
		}
		if (p_point.y < min.y) {
			return false;
		}
		if (p_point.z < min.z) {
			return false;
		}
		if (p_point.x > max.x) {
			return false;
		}
		if (p_point.y > max.y) {
			return false;
		}
		if (p_point.z > max.z) {
			return false;
		}

		return true;
	}

	T distance_squared(const Vector3T<T> p) const {
		const Vector3T<T> d = math::max(min - p, p - max, T(0.0));
		return d.length_squared();
	}
};

using Box3f = Box3fT<float>;

} // namespace zylann
