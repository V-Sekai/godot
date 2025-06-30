/**************************************************************************/
/*  voxel_blocky_model_viewer.h                                           */
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

#include "../../meshers/blocky/voxel_blocky_model.h"
#include "../../util/godot/classes/h_box_container.h"

ZN_GODOT_FORWARD_DECLARE(class Camera3D);
ZN_GODOT_FORWARD_DECLARE(class MeshInstance3D);
ZN_GODOT_FORWARD_DECLARE(class EditorUndoRedoManager);

namespace zylann {

namespace voxel {

class VoxelBlockyModelViewer : public HBoxContainer {
	GDCLASS(VoxelBlockyModelViewer, HBoxContainer)
public:
	VoxelBlockyModelViewer();

	void set_model(Ref<VoxelBlockyModel> model);

	// TODO GDX: `EditorUndoRedoManager` isn't a singleton yet in GDExtension, so it has to be injected
	void set_undo_redo(EditorUndoRedoManager *urm);

	// TODO GDX: `SceneTree::get_process_time` is not exposed, can't get delta time from `_notification`
#ifdef ZN_GODOT_EXTENSION
	void _process(double delta) override;
#endif

private:
	void update_model();
	void rotate_model_90(Vector3i::Axis axis);
	void add_rotation_anim(Basis basis);

#ifdef ZN_GODOT
	void _notification(int p_what);
#endif
	void process(float delta);

	void _on_model_changed();
	void _on_rotate_x_button_pressed();
	void _on_rotate_y_button_pressed();
	void _on_rotate_z_button_pressed();

	static void _bind_methods();

	Ref<VoxelBlockyModel> _model;
	EditorUndoRedoManager *_undo_redo = nullptr;
	Camera3D *_camera = nullptr;
	MeshInstance3D *_mesh_instance = nullptr;
	MeshInstance3D *_collision_boxes_mesh_instance = nullptr;
	float _pitch = 0.f;
	float _yaw = 0.f;
	float _distance = 1.9f;
	Basis _rotation_anim_basis;
};

} // namespace voxel
} // namespace zylann
