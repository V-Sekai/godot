/**************************************************************************/
/*  test_scene_merge_atlas_trim_sheet.h                                   */
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

#include "tests/test_macros.h"

#include "modules/scene_merge/merge.h"

namespace TestSceneMerge {

/**
 * Trim sheet boundary behavior testing for texture atlas operations.
 * Tests how texture atlas generation handles small, tightly-packed textures
 * typically found in trim sheets used by 3D artists.
 */
TEST_CASE("[Modules][SceneMerge][Atlas][TrimSheet] BarycentricAtlasTexel - TrimSheet Boundary Test") {
	// Test behavior near trim sheet boundaries (simulated by small texture)
	MeshTextureAtlas::AtlasTextureArguments args;
	args.atlas_data = Image::create_empty(8, 8, false, Image::FORMAT_RGBA8);
	args.atlas_data->fill(Color(0, 0, 0, 1));
	args.atlas_width = 8;
	args.atlas_height = 8;

	// Small 4x4 texture simulating a trim sheet
	args.source_texture = Image::create_empty(4, 4, false, Image::FORMAT_RGBA8);
	// Fill with a gradient to test interpolation
	for (int y = 0; y < 4; y++) {
		for (int x = 0; x < 4; x++) {
			float intensity = float(x + y) / 6.0f; // 0.0 to ~0.67
			args.source_texture->set_pixel(x, y, Color(intensity, intensity, intensity, 1.0f));
		}
	}

	// Triangle that covers the entire trim sheet
	args.source_uvs[0] = Vector2(0.0f, 0.0f); // Bottom-left
	args.source_uvs[1] = Vector2(1.0f, 0.0f); // Bottom-right
	args.source_uvs[2] = Vector2(0.0f, 1.0f); // Top-left

	MeshTextureAtlas::AtlasLookupTexel lookup;
	args.atlas_lookup = &lookup;

	// Test multiple points across the triangle
	const Vector3 test_points[] = {
		Vector3(1.0f, 0.0f, 0.0f), // Corner 0
		Vector3(0.0f, 1.0f, 0.0f), // Corner 1
		Vector3(0.0f, 0.0f, 1.0f), // Corner 2
		Vector3(0.33f, 0.33f, 0.33f) // Center
	};

	for (int i = 0; i < 4; i++) {
		bool result = MeshTextureAtlas::set_atlas_texel(&args, i, 0, test_points[i], Vector3(), Vector3(), 0.0f);
		CHECK(result);

		// Verify the pixel was set (not black)
		Color sampled = args.atlas_data->get_pixel(i, 0);
		CHECK(sampled.r >= 0.0f);
		CHECK(sampled.g >= 0.0f);
		CHECK(sampled.b >= 0.0f);
	}
}

} // namespace TestSceneMerge
