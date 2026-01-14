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

#include "tests/test_macros.h"

namespace TestCSG {

TEST_CASE("[SceneTree][CSG] CSGSculptedTexture3D") {
	SUBCASE("[SceneTree][CSG] CSGSculptedTexture3D: Basic texture sculpting") {
		CSGSculptedTexture3D *texture_primitive = memnew(CSGSculptedTexture3D);
		SceneTree::get_singleton()->get_root()->add_child(texture_primitive);

		// Create a simple 2x2 texture for testing
		Ref<Image> test_image;
		test_image.instantiate(2, 2, false, Image::FORMAT_RGB8);
		// Set pixel values: bottom-left (0,0): (0.5, 0.5, 0.5) -> (0, 0, 0)
		// bottom-right (1,0): (1.0, 0.5, 0.5) -> (1, 0, 0)
		// top-left (0,1): (0.5, 1.0, 0.5) -> (0, 1, 0)
		// top-right (1,1): (0.5, 0.5, 1.0) -> (0, 0, 1)
		test_image->set_pixel(0, 0, Color(0.5, 0.5, 0.5));
		test_image->set_pixel(1, 0, Color(1.0, 0.5, 0.5));
		test_image->set_pixel(0, 1, Color(0.5, 1.0, 0.5));
		test_image->set_pixel(1, 1, Color(0.5, 0.5, 1.0));

		Ref<ImageTexture> texture = ImageTexture::create_from_image(test_image);
		texture_primitive->set_sculpt_texture(texture);

		Vector<Vector3> faces = texture_primitive->get_brush_faces();

		// 2x2 texture should generate 2 triangles (4 total vertices, but shared)
		CHECK_MESSAGE(faces.size() == 6, "2x2 texture should generate 2 triangles (6 vertices)");

		// Check that vertices are at expected positions (scaled by default scale of 1.0)
		// Expected vertices based on RGB mapping: R=X, G=Y, B=Z, (0-1) -> (-1 to 1)
		Vector3 expected_vertices[4] = {
			Vector3(0.0, 0.0, 0.0), // (0.5, 0.5, 0.5) -> (0, 0, 0)
			Vector3(1.0, 0.0, 0.0), // (1.0, 0.5, 0.5) -> (1, 0, 0)
			Vector3(0.0, 1.0, 0.0), // (0.5, 1.0, 0.5) -> (0, 1, 0)
			Vector3(0.0, 0.0, 1.0) // (0.5, 0.5, 1.0) -> (0, 0, 1)
		};

		// Verify all expected vertices are present in the mesh
		bool found_vertices[4] = { false, false, false, false };
		for (int i = 0; i < faces.size(); i++) {
			for (int j = 0; j < 4; j++) {
				if (faces[i].is_equal_approx(expected_vertices[j])) {
					found_vertices[j] = true;
					break;
				}
			}
		}

		for (int j = 0; j < 4; j++) {
			CHECK_MESSAGE(found_vertices[j], "Expected vertex should be present in mesh");
		}

		SceneTree::get_singleton()->get_root()->remove_child(texture_primitive);
		memdelete(texture_primitive);
	}

	SUBCASE("[SceneTree][CSG] CSGSculptedTexture3D: Mirror and invert flags") {
		CSGSculptedTexture3D *texture_primitive = memnew(CSGSculptedTexture3D);
		SceneTree::get_singleton()->get_root()->add_child(texture_primitive);

		Ref<Image> test_image;
		test_image.instantiate(2, 2, false, Image::FORMAT_RGB8);
		test_image->set_pixel(0, 0, Color(0.5, 0.5, 0.5));
		test_image->set_pixel(1, 0, Color(1.0, 0.5, 0.5));
		test_image->set_pixel(0, 1, Color(0.5, 1.0, 0.5));
		test_image->set_pixel(1, 1, Color(0.5, 0.5, 1.0));

		Ref<ImageTexture> texture = ImageTexture::create_from_image(test_image);
		texture_primitive->set_sculpt_texture(texture);
		texture_primitive->set_mirror(true);
		texture_primitive->set_invert(true);

		Vector<Vector3> faces = texture_primitive->get_brush_faces();

		// With mirror and invert, vertices should be transformed
		// mirror: x = -x, invert: z = -z
		Vector3 expected_vertices[4] = {
			Vector3(0.0, 0.0, 0.0), // (0.5, 0.5, 0.5) -> (0, 0, 0) -> mirror+invert: (0, 0, 0)
			Vector3(-1.0, 0.0, 0.0), // (1.0, 0.5, 0.5) -> (1, 0, 0) -> mirror+invert: (-1, 0, 0)
			Vector3(0.0, 1.0, 0.0), // (0.5, 1.0, 0.5) -> (0, 1, 0) -> mirror+invert: (0, 1, 0)
			Vector3(0.0, 0.0, -1.0) // (0.5, 0.5, 1.0) -> (0, 0, 1) -> mirror+invert: (0, 0, -1)
		};

		bool found_vertices[4] = { false, false, false, false };
		for (int i = 0; i < faces.size(); i++) {
			for (int j = 0; j < 4; j++) {
				if (faces[i].is_equal_approx(expected_vertices[j])) {
					found_vertices[j] = true;
					break;
				}
			}
		}

		for (int j = 0; j < 4; j++) {
			CHECK_MESSAGE(found_vertices[j], "Transformed vertex should be present in mesh");
		}

		SceneTree::get_singleton()->get_root()->remove_child(texture_primitive);
		memdelete(texture_primitive);
	}
}

} // namespace TestCSG
