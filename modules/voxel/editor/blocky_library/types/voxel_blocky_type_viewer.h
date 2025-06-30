/**************************************************************************/
/*  voxel_blocky_type_viewer.h                                            */
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

#include "../../../meshers/blocky/types/voxel_blocky_type.h"
#include "../../../util/godot/classes/h_box_container.h"
#include "../model_viewer.h"

ZN_GODOT_FORWARD_DECLARE(class MeshInstance3D);

namespace zylann::voxel {

class VoxelBlockyTypeAttributeCombinationSelector;

// 3D viewer specialized to inspect blocky types.
class VoxelBlockyTypeViewer : public ZN_ModelViewer {
	GDCLASS(VoxelBlockyTypeViewer, ZN_ModelViewer)
public:
	VoxelBlockyTypeViewer();

	void set_combination_selector(VoxelBlockyTypeAttributeCombinationSelector *selector);
	void set_type(Ref<VoxelBlockyType> type);
	void update_model();

private:
	void _on_type_changed();
	void _on_combination_changed();

	static void _bind_methods();

	Ref<VoxelBlockyType> _type;
	MeshInstance3D *_mesh_instance = nullptr;
	const VoxelBlockyTypeAttributeCombinationSelector *_combination_selector = nullptr;
};

} // namespace zylann::voxel
