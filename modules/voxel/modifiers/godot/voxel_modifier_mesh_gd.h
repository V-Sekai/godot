/**************************************************************************/
/*  voxel_modifier_mesh_gd.h                                              */
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
#include "voxel_modifier_gd.h"

namespace zylann::voxel::godot {

class VoxelModifierMesh : public VoxelModifier {
	GDCLASS(VoxelModifierMesh, VoxelModifier);

public:
	void set_mesh_sdf(Ref<VoxelMeshSDF> mesh_sdf);
	Ref<VoxelMeshSDF> get_mesh_sdf() const;

	void set_isolevel(float isolevel);
	float get_isolevel() const;

#ifdef TOOLS_ENABLED
	void get_configuration_warnings(PackedStringArray &warnings) const override;
#endif

protected:
	zylann::voxel::VoxelModifier *create(zylann::voxel::VoxelModifierStack &modifiers, uint32_t id) override;

private:
	void _on_mesh_sdf_baked();

	static void _bind_methods();

	Ref<VoxelMeshSDF> _mesh_sdf;
	float _isolevel = 0.0f;
};

} // namespace zylann::voxel::godot
