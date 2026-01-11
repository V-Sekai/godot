/**************************************************************************/
/*  test_scene_merge_unwrap_trim.h                                        */
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

#include "modules/scene_merge/scene_merge.h"
#include "scene/resources/3d/importer_mesh.h"

namespace TestSceneMerge {

// NOTE: Trim sheet tests are currently disabled because trim sheets cause mesh merge operations to fail.
// This is a known critical limitation that prevents the module from working with production workflows
// that rely on trim sheet textures.

TEST_CASE_PENDING("[Modules][SceneMerge] Trim sheet mesh creation") {
	// This test is skipped because trim sheets cause mesh merge to fail
	// Implementation would go here if trim sheet support was added
}

TEST_CASE_PENDING("[Modules][SceneMerge] Non-trim sheet mesh succeeds") {
	// This test verifies that normal UVs (within 0-1 range) work correctly
	// Implementation would go here if we wanted to test non-trim sheet success
}

} // namespace TestSceneMerge
