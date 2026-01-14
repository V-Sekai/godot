/**************************************************************************/
/*  test_csg_sculpted_texture.h                                           */
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

#include "modules/sculpted_primitives/csg_sculpted_primitive_base.h"
#include "modules/sculpted_primitives/csg_sculpted_texture.h"

#include "scene/resources/image_texture.h"
#include "tests/test_macros.h"

namespace TestCSG {

TEST_CASE("[CSG] CSGSculptedTexture3D") {
	SUBCASE("Default initialization") {
		CSGSculptedTexture3D *texture_primitive = memnew(CSGSculptedTexture3D);

		// Check that it initializes without a texture
		CHECK(texture_primitive->get_sculpt_texture().is_null());

		// Check default profile curve (should be square for texture)
		CHECK(texture_primitive->get_profile_curve() == CSGSculptedPrimitive3D::PROFILE_CURVE_SQUARE);

		memdelete(texture_primitive);
	}

	SUBCASE("Texture property getters and setters") {
		CSGSculptedTexture3D *texture_primitive = memnew(CSGSculptedTexture3D);

		// Create a test texture
		Ref<Image> test_image;
		test_image.instantiate(2, 2, false, Image::FORMAT_RGB8);
		test_image->set_pixel(0, 0, Color(0.5, 0.5, 0.5));
		test_image->set_pixel(1, 0, Color(1.0, 0.5, 0.5));
		test_image->set_pixel(0, 1, Color(0.5, 1.0, 0.5));
		test_image->set_pixel(1, 1, Color(0.5, 0.5, 1.0));

		Ref<ImageTexture> texture = ImageTexture::create_from_image(test_image);
		texture_primitive->set_sculpt_texture(texture);

		// Verify texture was set
		CHECK_FALSE(texture_primitive->get_sculpt_texture().is_null());
		CHECK(texture_primitive->get_sculpt_texture()->get_width() == 2);
		CHECK(texture_primitive->get_sculpt_texture()->get_height() == 2);

		memdelete(texture_primitive);
	}

	SUBCASE("Texture coordinate mapping validation") {
		CSGSculptedTexture3D *texture_primitive = memnew(CSGSculptedTexture3D);

		// Create a simple 2x2 texture
		Ref<Image> test_image;
		test_image.instantiate(2, 2, false, Image::FORMAT_RGB8);
		// RGB values map to XYZ coordinates: R->X, G->Y, B->Z
		test_image->set_pixel(0, 0, Color(0.0, 0.0, 0.0)); // (0, 0, 0)
		test_image->set_pixel(1, 0, Color(1.0, 0.0, 0.0)); // (1, 0, 0)
		test_image->set_pixel(0, 1, Color(0.0, 1.0, 0.0)); // (0, 1, 0)
		test_image->set_pixel(1, 1, Color(0.0, 0.0, 1.0)); // (0, 0, 1)

		Ref<ImageTexture> texture = ImageTexture::create_from_image(test_image);
		texture_primitive->set_sculpt_texture(texture);

		// Verify texture dimensions
		CHECK(texture_primitive->get_sculpt_texture()->get_width() == 2);
		CHECK(texture_primitive->get_sculpt_texture()->get_height() == 2);

		// Test that we can access pixel data
		Ref<Image> retrieved_image = texture_primitive->get_sculpt_texture()->get_image();
		CHECK_FALSE(retrieved_image.is_null());
		CHECK(retrieved_image->get_width() == 2);
		CHECK(retrieved_image->get_height() == 2);

		memdelete(texture_primitive);
	}

	SUBCASE("Mirror and invert flags") {
		CSGSculptedTexture3D *texture_primitive = memnew(CSGSculptedTexture3D);

		// Test default values
		CHECK_FALSE(texture_primitive->get_mirror());
		CHECK_FALSE(texture_primitive->get_invert());

		// Test setting flags
		texture_primitive->set_mirror(true);
		texture_primitive->set_invert(true);

		CHECK(texture_primitive->get_mirror());
		CHECK(texture_primitive->get_invert());

		// Test toggling back
		texture_primitive->set_mirror(false);
		texture_primitive->set_invert(false);

		CHECK_FALSE(texture_primitive->get_mirror());
		CHECK_FALSE(texture_primitive->get_invert());

		memdelete(texture_primitive);
	}

	SUBCASE("Null texture handling") {
		CSGSculptedTexture3D *texture_primitive = memnew(CSGSculptedTexture3D);

		// Should handle null texture gracefully
		texture_primitive->set_sculpt_texture(Ref<ImageTexture>());
		CHECK(texture_primitive->get_sculpt_texture().is_null());

		memdelete(texture_primitive);
	}
}

} // namespace TestCSG
