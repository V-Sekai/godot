/**************************************************************************/
/*  geometry_2d.cpp                                                       */
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

#include "geometry_2d.h"
#include "../../math/conv.h"
#include "../../profiling.h"

namespace zylann::godot {

void geometry_2d_make_atlas(Span<const Vector2i> p_sizes, StdVector<Vector2i> &r_result, Vector2i &r_size) {
	ZN_PROFILE_SCOPE();

#if defined(ZN_GODOT)

	Vector<Vector2i> sizes;
	sizes.resize(p_sizes.size());
	Vector2i *sizes_data = sizes.ptrw();
	ZN_ASSERT_RETURN(sizes_data != nullptr);
	memcpy(sizes_data, p_sizes.data(), p_sizes.size() * sizeof(Vector2i));

	Vector<Vector2i> result;
	Geometry2D::make_atlas(sizes, result, r_size);

	r_result.resize(result.size());
	memcpy(r_result.data(), result.ptr(), result.size() * sizeof(Vector2i));

#elif defined(ZN_GODOT_EXTENSION)

	// PackedVector2iArray doesn't exist, so have to convert to float. Internally Godot allocates another array and
	// converts back to ints. Then allocates another float array and converts results to floats, returns it, and then
	// we finally convert back to ints... so much for not having added `PackedVector2iArray`.

	PackedVector2Array sizes;
	sizes.resize(p_sizes.size());
	Vector2 *sizes_data = sizes.ptrw();
	ZN_ASSERT_RETURN(sizes_data != nullptr);
	for (unsigned int i = 0; i < p_sizes.size(); ++i) {
		sizes_data[i] = p_sizes[i];
	}

	Dictionary result = Geometry2D::get_singleton()->make_atlas(sizes);
	PackedVector2Array positions = result["points"];
	r_size = result["size"];

	r_result.resize(positions.size());
	const Vector2 *positions_data = positions.ptr();
	ZN_ASSERT_RETURN(positions_data != nullptr);
	for (unsigned int i = 0; i < r_result.size(); ++i) {
		r_result[i] = to_vec2i(positions_data[i]);
	}

#endif
}

void geometry_2d_clip_polygons( //
		const PackedVector2Array &polygon_a, //
		const PackedVector2Array &polygon_b, //
		StdVector<PackedVector2Array> &output //
) {
#if defined(ZN_GODOT)
	Vector<Vector<Vector2>> result = Geometry2D::clip_polygons(polygon_a, polygon_b);
	output.resize(result.size());
	for (unsigned int i = 0; i < output.size(); ++i) {
		output[i] = result[i];
	}

#elif defined(ZN_GODOT_EXTENSION)
	TypedArray<PackedVector2Array> result = Geometry2D::get_singleton()->clip_polygons(polygon_a, polygon_b);
	output.resize(result.size());
	for (unsigned int i = 0; i < output.size(); ++i) {
		output[i] = result[i];
	}

#endif
}

} // namespace zylann::godot
