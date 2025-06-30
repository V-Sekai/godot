/**************************************************************************/
/*  interval.cpp                                                          */
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

#include "interval.h"
#include "../string/format.h"
#include <sstream>

namespace zylann {

namespace math {
namespace interval_impl {

template <typename T>
inline void check_range_once_t(T min, T max) {
	static bool once = false;
	if (min > max && once == false) {
		once = true;
		ZN_PRINT_ERROR(format("Interval constructed with invalid range: min={}, max={}", min, max));
	}
}

void check_range_once(float min, float max) {
	check_range_once_t(min, max);
}

void check_range_once(double min, double max) {
	check_range_once_t(min, max);
}

} // namespace interval_impl
} // namespace math

StdStringStream &operator<<(StdStringStream &ss, const math::Interval &v) {
	ss << "[" << v.min << ", " << v.max << "]";
	return ss;
}

} // namespace zylann
