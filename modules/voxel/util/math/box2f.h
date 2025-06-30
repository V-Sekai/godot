/**************************************************************************/
/*  box2f.h                                                               */
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

#include "../containers/small_vector.h"
#include "../string/std_stringstream.h"
#include "funcs.h"
#include "vector2f.h"

namespace zylann {

// Axis-aligned 2D box using float coordinates
class Box2f {
public:
	Vector2f min;
	Vector2f max;

	static Box2f from_min_size(const Vector2f p_min, const Vector2f p_size) {
		return { p_min, p_min + p_size };
	}

	static Box2f from_min_max(const Vector2f p_min, const Vector2f p_max) {
		return { p_min, p_max };
	}

	bool intersects(const Box2f &other) const {
		if (min.x >= other.max.x) {
			return false;
		}
		if (min.y >= other.max.y) {
			return false;
		}
		if (other.min.x >= max.x) {
			return false;
		}
		if (other.min.y >= max.y) {
			return false;
		}
		return true;
	}

	void clip(const Box2f lim) {
		min.x = math::clamp(min.x, lim.min.x, lim.max.x);
		min.y = math::clamp(min.y, lim.min.y, lim.max.y);
		max.x = math::clamp(max.x, lim.min.x, lim.max.x);
		max.y = math::clamp(max.y, lim.min.y, lim.max.y);
	}

	// Subtracts another box from the current box,
	// then execute a function on a set of boxes representing the remaining area.
	//
	// For example, seen from 2D, a possible result would be:
	//
	// o-----------o                 o-----o-----o
	// | A         |                 | C1  | C2  |
	// |     o-----+---o             |     o-----o
	// |     |     |   |   A - B =>  |     |
	// o-----+-----o   |             o-----o
	//       | B       |
	//       o---------o
	//
	template <typename A>
	void difference(const Box2f &b, A action) const {
		if (!intersects(b)) {
			action(*this);
			return;
		}

		Box2f a = *this;

		if (a.min.x < b.min.x) {
			action(Box2f::from_min_max(a.min, Vector2f(b.min.x, a.max.y)));
			a.min.x = b.min.x;
		}
		if (a.min.y < b.min.y) {
			action(Box2f::from_min_max(a.min, Vector2f(a.max.x, b.min.y)));
			a.min.y = b.min.y;
		}

		if (a.max.x > b.max.x) {
			action(Box2f::from_min_max(Vector2f(b.max.x, a.min.y), a.max));
			a.max.x = b.max.x;
		}
		if (a.max.y > b.max.y) {
			action(Box2f::from_min_max(Vector2f(a.min.x, b.max.y), a.max));
		}
	}

	inline void difference_to_vec(const Box2f &b, SmallVector<Box2f, 6> &output) const {
		difference(b, [&output](const Box2f &sub_box) { output.push_back(sub_box); });
	}
};

StdStringStream &operator<<(StdStringStream &ss, const Box2f &box);

} // namespace zylann
