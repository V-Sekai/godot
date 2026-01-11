/**************************************************************************/
/*  test_scene_merge_atlas.h                                              */
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

TEST_CASE("[Modules][SceneMerge] MeshMergeMeshInstanceWithMaterialAtlasTest") {
	MeshTextureAtlas::AtlasTextureArguments args;
	args.atlas_data = Image::create_empty(1024, 1024, false, Image::FORMAT_RGBA8);
	args.atlas_data->fill(Color());
	args.source_texture = Image::create_empty(1024, 1024, false, Image::FORMAT_RGBA8);
	args.source_texture->fill(Color());
	MeshTextureAtlas::AtlasLookupTexel lookup;
	args.atlas_lookup = &lookup;
	lookup.x = 512;
	lookup.y = 512;
	bool result = MeshTextureAtlas::set_atlas_texel(&args, 512, 512, Vector3(0.33, 0.33, 0.33), Vector3(), Vector3(), 0.0f);
	CHECK(result);
	lookup.x = 1023;
	lookup.y = 1023;
	result = MeshTextureAtlas::set_atlas_texel(&args, 1023, 1023, Vector3(0.33, 0.33, 0.33), Vector3(), Vector3(), 0.0f);
	CHECK(result);
}

} // namespace TestSceneMerge