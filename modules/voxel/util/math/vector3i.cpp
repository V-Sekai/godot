/**************************************************************************/
/*  vector3i.cpp                                                          */
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

#include "vector3i.h"
#include <sstream>

namespace zylann {

StdStringStream &operator<<(StdStringStream &ss, const Vector3i &v) {
	ss << "(" << v.x << ", " << v.y << ", " << v.z << ")";
	return ss;
}

namespace math {

Vector3i rotate_90(Vector3i v, Axis axis, bool clockwise) {
	if (axis == AXIS_X) {
		if (clockwise) {
			return math::rotate_x_90_cw(v);
		} else {
			return math::rotate_x_90_ccw(v);
		}
	} else if (axis == AXIS_Y) {
		if (clockwise) {
			return math::rotate_y_90_cw(v);
		} else {
			return math::rotate_y_90_ccw(v);
		}
	} else if (axis == AXIS_Z) {
		if (clockwise) {
			return math::rotate_z_90_cw(v);
		} else {
			return math::rotate_z_90_ccw(v);
		}
	} else {
		ZN_PRINT_ERROR("Invalid axis");
		return v;
	}
}

void rotate_90(Span<Vector3i> vecs, const Axis axis, const bool clockwise) {
	if (axis == AXIS_X) {
		if (clockwise) {
			for (Vector3i &v : vecs) {
				v = math::rotate_x_90_cw(v);
			}
		} else {
			for (Vector3i &v : vecs) {
				v = math::rotate_x_90_ccw(v);
			}
		}
	} else if (axis == AXIS_Y) {
		if (clockwise) {
			for (Vector3i &v : vecs) {
				v = math::rotate_y_90_cw(v);
			}
		} else {
			for (Vector3i &v : vecs) {
				v = math::rotate_y_90_ccw(v);
			}
		}
	} else if (axis == AXIS_Z) {
		if (clockwise) {
			for (Vector3i &v : vecs) {
				v = math::rotate_z_90_cw(v);
			}
		} else {
			for (Vector3i &v : vecs) {
				v = math::rotate_z_90_ccw(v);
			}
		}
	} else {
		ZN_PRINT_ERROR("Invalid axis");
	}
}

} // namespace math
} // namespace zylann
