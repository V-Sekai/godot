/**************************************************************************/
/*  test_scene_merge_atlas_barycentric.h                                  */
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
 * Barycentric coordinate interpolation testing for texture atlas sampling.
 * Tests accurate UV coordinate interpolation using barycentric coordinates.
 */
TEST_CASE("[Modules][SceneMerge][Atlas][Barycentric] BarycentricAtlasTexel - Basic Triangle Corners") {
	// Test barycentric coordinate interpolation at triangle corners
	MeshTextureAtlas::AtlasTextureArguments args;
	args.atlas_data = Image::create_empty(4, 4, false, Image::FORMAT_RGBA8);
	args.atlas_data->fill(Color(0, 0, 0, 1)); // Black background
	args.atlas_width = 4;
	args.atlas_height = 4;

	// Create a simple 2x2 source texture with distinct colors
	args.source_texture = Image::create_empty(2, 2, false, Image::FORMAT_RGBA8);
	args.source_texture->set_pixel(0, 0, Color(1, 0, 0, 1)); // Red at (0,0)
	args.source_texture->set_pixel(1, 0, Color(0, 1, 0, 1)); // Green at (1,0)
	args.source_texture->set_pixel(0, 1, Color(0, 0, 1, 1)); // Blue at (0,1)
	args.source_texture->set_pixel(1, 1, Color(1, 1, 0, 1)); // Yellow at (1,1)

	// Set up triangle UV coordinates
	args.source_uvs[0] = Vector2(0.0f, 0.0f); // Bottom-left (red)
	args.source_uvs[1] = Vector2(1.0f, 0.0f); // Bottom-right (green)
	args.source_uvs[2] = Vector2(0.0f, 1.0f); // Top-left (blue)

	// Set up lookup table
	MeshTextureAtlas::AtlasLookupTexel lookup;
	args.atlas_lookup = &lookup;

	// Test corner 0: barycentric (1,0,0) should sample red
	bool result = MeshTextureAtlas::set_atlas_texel(&args, 0, 0, Vector3(1.0f, 0.0f, 0.0f), Vector3(), Vector3(), 0.0f);
	CHECK(result);
	Color sampled_color = args.atlas_data->get_pixel(0, 0);
	CHECK(sampled_color.r > 0.9f); // Should be mostly red
	CHECK(sampled_color.g < 0.1f);
	CHECK(sampled_color.b < 0.1f);

	// Test corner 1: barycentric (0,1,0) should sample green
	result = MeshTextureAtlas::set_atlas_texel(&args, 1, 0, Vector3(0.0f, 1.0f, 0.0f), Vector3(), Vector3(), 0.0f);
	CHECK(result);
	sampled_color = args.atlas_data->get_pixel(1, 0);
	CHECK(sampled_color.r < 0.1f);
	CHECK(sampled_color.g > 0.9f); // Should be mostly green
	CHECK(sampled_color.b < 0.1f);

	// Test corner 2: barycentric (0,0,1) should sample blue
	result = MeshTextureAtlas::set_atlas_texel(&args, 0, 1, Vector3(0.0f, 0.0f, 1.0f), Vector3(), Vector3(), 0.0f);
	CHECK(result);
	sampled_color = args.atlas_data->get_pixel(0, 1);
	CHECK(sampled_color.r < 0.1f);
	CHECK(sampled_color.g < 0.1f);
	CHECK(sampled_color.b > 0.9f); // Should be mostly blue
}

TEST_CASE("[Modules][SceneMerge][Atlas][Barycentric] BarycentricAtlasTexel - Center Interpolation") {
	// Test barycentric interpolation at triangle center
	MeshTextureAtlas::AtlasTextureArguments args;
	args.atlas_data = Image::create_empty(4, 4, false, Image::FORMAT_RGBA8);
	args.atlas_data->fill(Color(0, 0, 0, 1));
	args.atlas_width = 4;
	args.atlas_height = 4;

	// Create a simple 2x2 source texture
	args.source_texture = Image::create_empty(2, 2, false, Image::FORMAT_RGBA8);
	args.source_texture->set_pixel(0, 0, Color(1, 0, 0, 1)); // Red
	args.source_texture->set_pixel(1, 0, Color(0, 1, 0, 1)); // Green
	args.source_texture->set_pixel(0, 1, Color(0, 0, 1, 1)); // Blue
	args.source_texture->set_pixel(1, 1, Color(1, 1, 0, 1)); // Yellow

	// Triangle covers the bottom-left quarter
	args.source_uvs[0] = Vector2(0.0f, 0.0f); // Red
	args.source_uvs[1] = Vector2(0.5f, 0.0f); // Between red and green
	args.source_uvs[2] = Vector2(0.0f, 0.5f); // Between red and blue

	MeshTextureAtlas::AtlasLookupTexel lookup;
	args.atlas_lookup = &lookup;

	// Test center of triangle: barycentric (0.33, 0.33, 0.33)
	// Should interpolate between the three UV coordinates
	bool result = MeshTextureAtlas::set_atlas_texel(&args, 2, 2, Vector3(0.33f, 0.33f, 0.33f), Vector3(), Vector3(), 0.0f);
	CHECK(result);

	// The interpolated UV should be around (0.167, 0.167)
	// This should sample close to red since it's near the red corner
	Color sampled_color = args.atlas_data->get_pixel(2, 2);
	// Should be some blend of red, with red being dominant
	CHECK(sampled_color.r > sampled_color.g);
	CHECK(sampled_color.r > sampled_color.b);
}

} // namespace TestSceneMerge
