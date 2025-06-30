/**************************************************************************/
/*  voxel_blocky_model_editor_inspector_plugin.cpp                        */
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

#include "voxel_blocky_model_editor_inspector_plugin.h"
#include "voxel_blocky_model_viewer.h"

namespace zylann::voxel {

void VoxelBlockyModelEditorInspectorPlugin::set_undo_redo(EditorUndoRedoManager *urm) {
	_undo_redo = urm;
}

bool VoxelBlockyModelEditorInspectorPlugin::_zn_can_handle(const Object *p_object) const {
	return Object::cast_to<VoxelBlockyModel>(p_object) != nullptr;
}

void VoxelBlockyModelEditorInspectorPlugin::_zn_parse_begin(Object *p_object) {
	const VoxelBlockyModel *model_ptr = Object::cast_to<VoxelBlockyModel>(p_object);
	ZN_ASSERT_RETURN(model_ptr != nullptr);

	Ref<VoxelBlockyModel> model(model_ptr);

	VoxelBlockyModelViewer *viewer = memnew(VoxelBlockyModelViewer);
	viewer->set_model(model);
	viewer->set_undo_redo(_undo_redo);
	add_custom_control(viewer);
	return;
}

} // namespace zylann::voxel
