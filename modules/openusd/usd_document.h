/**************************************************************************/
/*  usd_document.h                                                        */
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

#include "core/io/resource.h"
#include "scene/main/node.h"

// USD headers
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>

class UsdState;

class UsdDocument : public Resource {
	GDCLASS(UsdDocument, Resource);

protected:
	static void _bind_methods();

public:
	UsdDocument();

	// Export methods
	Error append_from_scene(Node *p_scene_root, Ref<UsdState> p_state, int32_t p_flags = 0);
	Error write_to_filesystem(Ref<UsdState> p_state, const String &p_path);
	String get_file_extension_for_format(bool p_binary) const;

	// Import methods
	Error import_from_file(const String &p_path, Node *p_parent, Ref<UsdState> p_state);

private:
	// Export helpers
	Error _convert_node_to_prim(Node *p_node, pxr::UsdStageRefPtr p_stage, const pxr::SdfPath &p_parent_path, Ref<UsdState> p_state);

	// Import helpers
	Error _import_prim_hierarchy(const pxr::UsdStageRefPtr &p_stage, const pxr::SdfPath &p_prim_path, Node *p_parent, Ref<UsdState> p_state);
};
