/**************************************************************************/
/*  test_noise.cpp                                                        */
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

#include "test_noise.h"
#include "../../util/noise/fast_noise_lite/fast_noise_lite.h"
#include "../../util/noise/fast_noise_lite/fast_noise_lite_range.h"
#include "../../util/testing/test_macros.h"

namespace zylann::tests {

void test_fnl_range() {
	Ref<ZN_FastNoiseLite> noise;
	noise.instantiate();
	noise->set_noise_type(ZN_FastNoiseLite::TYPE_OPEN_SIMPLEX_2S);
	noise->set_fractal_type(ZN_FastNoiseLite::FRACTAL_NONE);
	// noise->set_fractal_type(ZN_FastNoiseLite::FRACTAL_FBM);
	noise->set_fractal_octaves(1);
	noise->set_fractal_lacunarity(2.0);
	noise->set_fractal_gain(0.5);
	noise->set_period(512);
	noise->set_seed(0);

	const Vector3i min_pos(-1074, 1838, 5587);
	// const Vector3i max_pos(-1073, 1839, 5588);
	const Vector3i max_pos(-1058, 1854, 5603);

	const math::Interval x_range(min_pos.x, max_pos.x - 1);
	const math::Interval y_range(min_pos.y, max_pos.y - 1);
	const math::Interval z_range(min_pos.z, max_pos.z - 1);

	const math::Interval analytic_range = get_fnl_range_3d(**noise, x_range, y_range, z_range);

	math::Interval empiric_range;
	bool first_value = true;
	for (int z = min_pos.z; z < max_pos.z; ++z) {
		for (int y = min_pos.y; y < max_pos.y; ++y) {
			for (int x = min_pos.x; x < max_pos.x; ++x) {
				const float n = noise->get_noise_3d(x, y, z);
				if (first_value) {
					empiric_range.min = n;
					empiric_range.max = n;
					first_value = false;
				} else {
					empiric_range.min = math::min<real_t>(empiric_range.min, n);
					empiric_range.max = math::max<real_t>(empiric_range.max, n);
				}
			}
		}
	}

	ZN_TEST_ASSERT(analytic_range.contains(empiric_range));
}

} // namespace zylann::tests
