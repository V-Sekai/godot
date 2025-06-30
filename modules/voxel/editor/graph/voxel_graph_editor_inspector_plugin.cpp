/**************************************************************************/
/*  voxel_graph_editor_inspector_plugin.cpp                               */
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

#include "voxel_graph_editor_inspector_plugin.h"
#include "editor_property_text_change_on_submit.h"
#include "voxel_graph_node_inspector_wrapper.h"

namespace zylann::voxel {

bool VoxelGraphEditorInspectorPlugin::_zn_can_handle(const Object *obj) const {
	return obj != nullptr && Object::cast_to<VoxelGraphNodeInspectorWrapper>(obj) != nullptr;
}

bool VoxelGraphEditorInspectorPlugin::_zn_parse_property(
		Object *p_object,
		const Variant::Type p_type,
		const String &p_path,
		const PropertyHint p_hint,
		const String &p_hint_text,
		const BitField<PropertyUsageFlags> p_usage,
		const bool p_wide
) {
	if (p_type == Variant::STRING && p_hint != PROPERTY_HINT_MULTILINE_TEXT) {
		add_property_editor(p_path, memnew(ZN_EditorPropertyTextChangeOnSubmit));
		return true;
	}
	return false;
}

} // namespace zylann::voxel
