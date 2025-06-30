/**************************************************************************/
/*  test_fast_noise_2.cpp                                                 */
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

#include "test_fast_noise_2.h"
#include "../../util/godot/classes/image.h"
#include "../../util/io/log.h"
#include "../../util/noise/fast_noise_2.h"
#include "../../util/string/format.h"

namespace zylann::tests {

void test_fast_noise_2_basic() {
	// Very basic test. The point is to make sure it doesn't crash, so there is no special condition to check.
	Ref<FastNoise2> noise;
	noise.instantiate();
	float nv = noise->get_noise_2d_single(Vector2(42, 666));
	print_line(format("SIMD level: {}", FastNoise2::get_simd_level_name_c_str(noise->get_simd_level())));
	print_line(format("Noise: {}", nv));
	Ref<Image> im = godot::create_empty_image(256, 256, false, Image::FORMAT_RGB8);
	noise->generate_image(im, false);
	// im->save_png("zylann_test_fastnoise2.png");
}

void test_fast_noise_2_empty_encoded_node_tree() {
	Ref<FastNoise2> noise;
	noise.instantiate();
	noise->set_noise_type(FastNoise2::TYPE_ENCODED_NODE_TREE);
	// This can print an error, but should not crash
	noise->update_generator();
}

} // namespace zylann::tests
