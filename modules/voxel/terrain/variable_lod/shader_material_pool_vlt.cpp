/**************************************************************************/
/*  shader_material_pool_vlt.cpp                                          */
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

#include "shader_material_pool_vlt.h"
#include "../../constants/voxel_string_names.h"
#include "../../util/godot/classes/texture_2d.h"
#include "../../util/profiling.h"

namespace zylann::voxel {

void ShaderMaterialPoolVLT::recycle(Ref<ShaderMaterial> material) {
	ZN_PROFILE_SCOPE();
	ZN_ASSERT_RETURN(material.is_valid());

	const VoxelStringNames &sn = VoxelStringNames::get_singleton();

	// Reset textures to avoid hoarding them in the pool
	material->set_shader_parameter(sn.u_voxel_normalmap_atlas, Ref<Texture2D>());
	material->set_shader_parameter(sn.u_voxel_cell_lookup, Ref<Texture2D>());
	material->set_shader_parameter(sn.u_voxel_virtual_texture_offset_scale, Vector4(0, 0, 0, 1));
	// TODO Would be nice if we repurposed `u_transition_mask` to store extra flags.
	// Here we exploit cell_size==0 as "there is no virtual normalmaps on this block"
	material->set_shader_parameter(sn.u_voxel_cell_size, 0.f);
	material->set_shader_parameter(sn.u_voxel_virtual_texture_fade, 0.f);

	material->set_shader_parameter(sn.u_transition_mask, 0);
	material->set_shader_parameter(sn.u_lod_fade, Vector2(0.0, 0.0));

	zylann::godot::ShaderMaterialPool::recycle(material);
}

} // namespace zylann::voxel
