/**************************************************************************/
/*  voxel_mesh_sdf_editor_plugin.cpp                                      */
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

#include "voxel_mesh_sdf_editor_plugin.h"
#include "../../edition/voxel_mesh_sdf_gd.h"
#include "voxel_mesh_sdf_viewer.h"

namespace zylann::voxel {

bool VoxelMeshSDFInspectorPlugin::_zn_can_handle(const Object *p_object) const {
	return Object::cast_to<VoxelMeshSDF>(p_object) != nullptr;
}

void VoxelMeshSDFInspectorPlugin::_zn_parse_begin(Object *p_object) {
	VoxelMeshSDFViewer *viewer = memnew(VoxelMeshSDFViewer);
	add_custom_control(viewer);
	VoxelMeshSDF *mesh_sdf = Object::cast_to<VoxelMeshSDF>(p_object);
	ERR_FAIL_COND(mesh_sdf == nullptr);
	viewer->set_mesh_sdf(mesh_sdf);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VoxelMeshSDFEditorPlugin::VoxelMeshSDFEditorPlugin() {}

bool VoxelMeshSDFEditorPlugin::_zn_handles(const Object *p_object) const {
	ERR_FAIL_COND_V(p_object == nullptr, false);
	return Object::cast_to<VoxelMeshSDF>(p_object) != nullptr;
}

void VoxelMeshSDFEditorPlugin::_zn_edit(Object *p_object) {
	//_mesh_sdf = p_object;
}

void VoxelMeshSDFEditorPlugin::_zn_make_visible(bool visible) {
	//_mesh_sdf.unref();
}

void VoxelMeshSDFEditorPlugin::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		_inspector_plugin.instantiate();
		// TODO Why can other Godot plugins do this in the constructor??
		// I found I could not put this in the constructor,
		// otherwise `add_inspector_plugin` causes ANOTHER editor plugin to leak on exit... Oo
		add_inspector_plugin(_inspector_plugin);

	} else if (p_what == NOTIFICATION_EXIT_TREE) {
		remove_inspector_plugin(_inspector_plugin);
	}
}

} // namespace zylann::voxel
