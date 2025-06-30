/**************************************************************************/
/*  voxel_modifier_mesh.h                                                 */
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

#include "../edition/voxel_mesh_sdf_gd.h"
#include "voxel_modifier_sdf.h"

namespace zylann::voxel {

class VoxelModifierMesh : public VoxelModifierSdf {
public:
	Type get_type() const override {
		return TYPE_MESH;
	};

	void set_mesh_sdf(Ref<VoxelMeshSDF> mesh_sdf);
	void set_isolevel(float isolevel);
	void apply(VoxelModifierContext ctx) const override;

#ifdef VOXEL_ENABLE_GPU
	void get_shader_data(ShaderData &out_shader_data) override;
	void request_shader_data_update();
#endif

protected:
	void update_aabb() override;

private:
	// Originally I wanted to keep the core of modifiers separate from Godot stuff, but in order to also support
	// GPU resources, putting this here was easier.
	Ref<VoxelMeshSDF> _mesh_sdf;
	float _isolevel;
};

} // namespace zylann::voxel
