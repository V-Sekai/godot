/**************************************************************************/
/*  sdf.h                                                                 */
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

#ifndef VOXEL_MATH_SDF_H
#define VOXEL_MATH_SDF_H

#include "interval.h"
#include <core/math/vector2.h>

// Signed-distance-field functions.
// For more, see
// https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm

inline float sdf_box(const Vector3 pos, const Vector3 extents) {
	Vector3 d = pos.abs() - extents;
	return MIN(MAX(d.x, MAX(d.y, d.z)), 0.f) +
			Vector3(MAX(d.x, 0.f), MAX(d.y, 0.f), MAX(d.z, 0.f)).length();
}

inline Interval sdf_box(const Interval &x, const Interval &y, const Interval &z,
		const Interval &sx, const Interval &sy,
		const Interval &sz) {
	Interval dx = abs(x) - sx;
	Interval dy = abs(y) - sy;
	Interval dz = abs(z) - sz;
	return min_interval(max_interval(dx, max_interval(dy, dz)), 0.f) +
			get_length(max_interval(dx, 0.f), max_interval(dy, 0.f),
					max_interval(dz, 0.f));
}

inline float sdf_torus(float x, float y, float z, float r0, float r1) {
	Vector2 q = Vector2(Vector2(x, z).length() - r0, y);
	return q.length() - r1;
}

inline Interval sdf_torus(const Interval &x, const Interval &y,
		const Interval &z, const Interval r0,
		const Interval r1) {
	Interval qx = get_length(x, z) - r0;
	return get_length(qx, y) - r1;
}

inline float sdf_union(float a, float b) {
	return MIN(a, b);
}

// Subtracts SDF b from SDF a
inline float sdf_subtract(float a, float b) {
	return MAX(a, -b);
}

inline float sdf_smooth_union(float a, float b, float s) {
	float h = clamp(0.5f + 0.5f * (b - a) / s, 0.0f, 1.0f);
	return Math::lerp(b, a, h) - s * h * (1.0f - h);
}

// Inverted a and b because it subtracts SDF a from SDF b
inline float sdf_smooth_subtract(float b, float a, float s) {
	float h = clamp(0.5f - 0.5f * (b + a) / s, 0.0f, 1.0f);
	return Math::lerp(b, -a, h) + s * h * (1.0f - h);
}

#endif // VOXEL_MATH_SDF_H
