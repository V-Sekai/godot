/**************************************************************************/
/*  voxel_mesh_sdf_viewer.h                                               */
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

#include "../../edition/voxel_mesh_sdf_gd.h"
#include "../../util/godot/classes/v_box_container.h"

ZN_GODOT_FORWARD_DECLARE(class TextureRect)
ZN_GODOT_FORWARD_DECLARE(class Button)
ZN_GODOT_FORWARD_DECLARE(class SpinBox)
ZN_GODOT_FORWARD_DECLARE(class Label)

namespace zylann::voxel {

class VoxelMeshSDFViewer : public VBoxContainer {
	GDCLASS(VoxelMeshSDFViewer, VBoxContainer)
public:
	VoxelMeshSDFViewer();

	void set_mesh_sdf(Ref<VoxelMeshSDF> mesh_sdf);

	void update_view();

private:
	void _on_bake_button_pressed();
	void _on_mesh_sdf_baked();
	void _on_slice_spinbox_value_changed(float value);

	void update_bake_button();
	void update_info_label();
	void center_slice_y();
	// void clamp_slice_y();
	void update_slice_spinbox();

	static void _bind_methods();

	TextureRect *_texture_rect = nullptr;
	Label *_info_label = nullptr;
	SpinBox *_slice_spinbox = nullptr;
	int _slice_y = 0;
	bool _slice_spinbox_ignored = false;
	Button *_bake_button = nullptr;
	Ref<VoxelMeshSDF> _mesh_sdf;
	Vector3i _size_before_baking;
};

} // namespace zylann::voxel
