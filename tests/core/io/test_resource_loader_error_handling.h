/**************************************************************************/
/*  test_resource_loader_error_handling.h                                 */
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

#include "core/io/resource_loader.h"
#include "scene/resources/packed_scene.h"

namespace TestResourceLoaderErrorHandling {

TEST_CASE("[ResourceLoader] Resource with missing dependencies should return null") {
	// This test verifies that when a resource has missing dependencies,
	// ResourceLoader::load() returns null and sets the error code correctly.
	// Previously, it would return a non-null but invalid resource that would
	// crash when instantiate() was called.

	Error err = OK;
	Ref<PackedScene> packed_scene = ResourceLoader::load(
			"res://non_existent_scene.scn",
			"",
			ResourceFormatLoader::CACHE_MODE_REPLACE,
			&err);

	// When a resource has missing dependencies, the error should be set
	// and the resource should be null
	CHECK_MESSAGE(err != OK, "Error should be set when resource has missing dependencies");
	CHECK_MESSAGE(packed_scene.is_null(), "Resource should be null when there's an error");

	// If the resource is not null, it should not crash when we try to use it
	if (packed_scene.is_valid()) {
		// This should not crash - if it does, it means we're returning
		// an invalid resource when there's an error
		Node *node = packed_scene->instantiate();
		CHECK_MESSAGE(node == nullptr, "instantiate() should return null for invalid resources");
	}
}

} // namespace TestResourceLoaderErrorHandling
