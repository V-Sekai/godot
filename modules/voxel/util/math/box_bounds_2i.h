/**************************************************************************/
/*  box_bounds_2i.h                                                       */
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

#include "../string/std_stringstream.h"
#include "box2i.h"

namespace zylann {

// Alternative implementation of an integer axis-aligned box, storing min and max positions for faster intersection
// checks.
struct BoxBounds2i {
	Vector2i min_pos;
	Vector2i max_pos; // Exclusive

	BoxBounds2i() {}

	BoxBounds2i(Vector2i p_min, Vector2i p_max) : min_pos(p_min), max_pos(p_max) {}

	BoxBounds2i(Box2i box) : min_pos(box.position), max_pos(box.position + box.size) {}

	static BoxBounds2i from_position_size(Vector2i pos, Vector2i size) {
		return BoxBounds2i(pos, pos + size);
	}

	static BoxBounds2i from_min_max_included(Vector2i minp, Vector2i maxp) {
		return BoxBounds2i(minp, maxp + Vector2i(1, 1));
	}

	static BoxBounds2i from_position(Vector2i pos) {
		return BoxBounds2i(pos, pos + Vector2i(1, 1));
	}

	static BoxBounds2i from_everywhere() {
		return BoxBounds2i(
				Vector2iUtil::create(std::numeric_limits<int>::min()),
				Vector2iUtil::create(std::numeric_limits<int>::max())
		);
	}

	inline bool intersects(const BoxBounds2i &other) const {
		return !( //
				max_pos.x < other.min_pos.x || //
				max_pos.y < other.min_pos.y || //
				min_pos.x > other.max_pos.x || //
				min_pos.y > other.max_pos.y
		);
	}

	inline bool operator==(const BoxBounds2i &other) const {
		return min_pos == other.min_pos && max_pos == other.max_pos;
	}

	inline bool is_empty() const {
		return min_pos.x >= max_pos.x || min_pos.y >= max_pos.y;
	}

	inline Vector2i get_size() const {
		return max_pos - min_pos;
	}
};

StdStringStream &operator<<(StdStringStream &ss, const BoxBounds2i &box);

} // namespace zylann
