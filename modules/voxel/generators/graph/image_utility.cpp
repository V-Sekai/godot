/**************************************************************************/
/*  image_utility.cpp                                                     */
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

#include "image_utility.h"
#include "../../util/godot/classes/image.h"
#include "../../util/math/vector2i.h"
#include "../../util/string/format.h"

namespace zylann {

using namespace math;

Interval get_heightmap_range(const Image &im) {
	return get_heightmap_range(im, Rect2i(0, 0, im.get_width(), im.get_height()));
}

Interval get_heightmap_range(const Image &im, Rect2i rect) {
#ifdef DEBUG_ENABLED
	ZN_ASSERT_RETURN_V_MSG(!im.is_compressed(), Interval(), format("Image format not supported: {}", im.get_format()));
	ZN_ASSERT_RETURN_V_MSG(
			Rect2i(0, 0, im.get_width(), im.get_height()).encloses(rect),
			Interval(0, 0),
			format("Rectangle out of range: image size is {}, rectangle is {}", im.get_size(), rect)
	);
#endif

	Interval r;

	r.min = im.get_pixel(rect.position.x, rect.position.y).r;
	r.max = r.min;

	const int max_x = rect.position.x + rect.size.x;
	const int max_y = rect.position.y + rect.size.y;

	for (int y = rect.position.y; y < max_y; ++y) {
		for (int x = rect.position.x; x < max_x; ++x) {
			r.add_point(im.get_pixel(x, y).r);
		}
	}

	return r;
}

} // namespace zylann
