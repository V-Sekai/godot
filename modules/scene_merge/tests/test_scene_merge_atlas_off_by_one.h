/**************************************************************************/
/*  test_scene_merge_atlas_off_by_one.h                                   */
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
 * Off-by-one error testing for texture atlas operations.
 * Tests boundary conditions that could cause texture sampling bugs.
 */
TEST_CASE("[Modules][SceneMerge][Atlas][OffByOne] BarycentricAtlasTexel - OffByOne Edge Cases") {
	// Test edge cases that might cause off-by-one texel errors
	MeshTextureAtlas::AtlasTextureArguments args;
	args.atlas_data = Image::create_empty(4, 4, false, Image::FORMAT_RGBA8);
	args.atlas_data->fill(Color(0, 0, 0, 1));
	args.atlas_width = 4;
	args.atlas_height = 4;

	// Create a 3x3 source texture to test boundary conditions
	args.source_texture = Image::create_empty(3, 3, false, Image::FORMAT_RGBA8);
	for (int y = 0; y < 3; y++) {
		for (int x = 0; x < 3; x++) {
			args.source_texture->set_pixel(x, y, Color(float(x) / 2.0f, float(y) / 2.0f, 0.5f, 1.0f));
		}
	}

	args.source_uvs[0] = Vector2(0.0f, 0.0f); // Corner (0,0)
	args.source_uvs[1] = Vector2(1.0f, 0.0f); // Corner (1,0)
	args.source_uvs[2] = Vector2(0.0f, 1.0f); // Corner (0,1)

	MeshTextureAtlas::AtlasLookupTexel lookup;
	args.atlas_lookup = &lookup;

	// Test UV coordinate at exactly 1.0 (edge of texture)
	// This should not cause out-of-bounds access
	bool result = MeshTextureAtlas::set_atlas_texel(&args, 0, 0, Vector3(0.0f, 1.0f, 0.0f), Vector3(), Vector3(), 0.0f);
	CHECK(result);

	// Test UV coordinate at exactly 0.0
	result = MeshTextureAtlas::set_atlas_texel(&args, 1, 1, Vector3(1.0f, 0.0f, 0.0f), Vector3(), Vector3(), 0.0f);
	CHECK(result);

	// Verify coordinates are clamped properly (no negative values)
	Pair<int, int> coords = MeshTextureAtlas::calculate_coordinates(Vector2(0.0f, 0.0f), 3, 3);
	CHECK(coords.first >= 0);
	CHECK(coords.second >= 0);
	CHECK(coords.first < 3);
	CHECK(coords.second < 3);

	coords = MeshTextureAtlas::calculate_coordinates(Vector2(1.0f, 1.0f), 3, 3);
	CHECK(coords.first >= 0);
	CHECK(coords.second >= 0);
	CHECK(coords.first < 3);
	CHECK(coords.second < 3);
}

} // namespace TestSceneMerge
