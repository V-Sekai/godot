/**************************************************************************/
/*  voxel_terrain_editor_inspector_plugin.cpp                             */
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

#include "voxel_terrain_editor_inspector_plugin.h"
#include "../../terrain/fixed_lod/voxel_terrain.h"
#include "../../terrain/variable_lod/voxel_lod_terrain.h"
#include "../../util/godot/classes/object.h"
#include "editor_property_aabb_min_max.h"

namespace zylann::voxel {

bool VoxelTerrainEditorInspectorPlugin::_zn_can_handle(const Object *p_object) const {
	const VoxelTerrain *vt = Object::cast_to<VoxelTerrain>(p_object);
	if (vt != nullptr) {
		return true;
	}
	const VoxelLodTerrain *vlt = Object::cast_to<VoxelLodTerrain>(p_object);
	if (vlt != nullptr) {
		return true;
	}
	return false;
}

bool VoxelTerrainEditorInspectorPlugin::_zn_parse_property(
		Object *p_object,
		const Variant::Type p_type,
		const String &p_path,
		const PropertyHint p_hint,
		const String &p_hint_text,
		const BitField<PropertyUsageFlags> p_usage,
		const bool p_wide
) {
	if (p_type != Variant::AABB) {
		return false;
	}
	// TODO Give the same name to these properties
	if (p_path != "voxel_bounds" && p_path != "bounds") {
		return false;
	}
	// Replace default AABB editor with this one
	ZN_EditorPropertyAABBMinMax *ed = memnew(ZN_EditorPropertyAABBMinMax);
	ed->setup(-constants::MAX_VOLUME_EXTENT, constants::MAX_VOLUME_EXTENT, 1, true);
	add_property_editor(p_path, ed);
	return true;
}

} // namespace zylann::voxel
