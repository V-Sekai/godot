/**************************************************************************/
/*  vox_import_funcs.cpp                                                  */
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

#include "vox_import_funcs.h"
#include "../../util/godot/classes/array_mesh.h"

namespace zylann {

namespace voxel::magica {

Ref<Mesh> build_mesh(
		const VoxelBuffer &voxels,
		VoxelMesher &mesher,
		StdVector<unsigned int> &surface_index_to_material,
		Ref<Image> &out_atlas,
		float p_scale,
		Vector3 p_offset
) {
	VoxelMesher::Output output;
	VoxelMesher::Input input{ voxels, nullptr, Vector3i(), 0, false };
	mesher.build(output, input);

	if (output.surfaces.size() == 0) {
		return Ref<ArrayMesh>();
	}

	Ref<ArrayMesh> mesh;
	mesh.instantiate();

	for (unsigned int i = 0; i < output.surfaces.size(); ++i) {
		VoxelMesher::Output::Surface &surface = output.surfaces[i];
		Array arrays = surface.arrays;

		if (arrays.is_empty()) {
			continue;
		}

		CRASH_COND(arrays.size() != Mesh::ARRAY_MAX);
		if (!zylann::godot::is_surface_triangulated(arrays)) {
			continue;
		}

		if (p_scale != 1.f) {
			zylann::godot::scale_surface(arrays, p_scale);
		}

		if (p_offset != Vector3()) {
			zylann::godot::offset_surface(arrays, p_offset);
		}

		mesh->add_surface_from_arrays(output.primitive_type, arrays, Array(), Dictionary(), output.mesh_flags);
		surface_index_to_material.push_back(i);
	}

	if (output.atlas_image.is_valid()) {
		out_atlas = output.atlas_image;
	}

	return mesh;
}

} // namespace voxel::magica
} // namespace zylann
