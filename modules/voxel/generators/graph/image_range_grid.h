/**************************************************************************/
/*  image_range_grid.h                                                    */
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

#include "../../util/containers/fixed_array.h"
#include "../../util/containers/std_vector.h"
#include "../../util/godot/macros.h"
#include "../../util/math/interval.h"

ZN_GODOT_FORWARD_DECLARE(class Image)

namespace zylann {

// Stores minimum and maximum values over a 2D image at multiple levels of detail
class ImageRangeGrid {
public:
	~ImageRangeGrid();

	void clear();
	void generate(const Image &im);
	inline math::Interval get_range() const {
		return _total_range;
	}

	// Gets a quick upper bound of the range of values within an area of the image. If the area goes out of bounds of
	// the image, evaluation will be done as if the image repeats infinitely.
	math::Interval get_range_repeat(math::Interval xr, math::Interval yr) const;

private:
	static const int MAX_LODS = 16;

	struct Lod {
		// Grid of chunks containing the min and max of all pixels covered by each chunk
		StdVector<math::Interval> data;
		// In chunks
		int size_x = 0;
		int size_y = 0;
	};

	// Original size
	int _pixels_x = 0;
	int _pixels_y = 0;
	bool _pixels_x_is_power_of_2 = true;
	bool _pixels_y_is_power_of_2 = true;

	int _lod_base = 0;
	int _lod_count = 0;

	math::Interval _total_range;

	FixedArray<Lod, MAX_LODS> _lods;
};

} // namespace zylann
