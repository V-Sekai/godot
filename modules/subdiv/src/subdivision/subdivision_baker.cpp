/**************************************************************************/
/*  subdivision_baker.cpp                                                 */
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

#include "subdivision_baker.hpp"

#include "quad_subdivider.hpp"
#include "triangle_subdivider.hpp"

Array SubdivisionBaker::get_baked_arrays(const Array &topology_arrays, int p_level, int64_t p_format, int topology_type) {
	switch (topology_type) {
		case TopologyDataMesh::TOPOLOGY_DATA_MESH_QUAD: {
			Ref<QuadSubdivider> subdivider;
			subdivider.instantiate();
			return subdivider->get_subdivided_arrays(topology_arrays, p_level, p_format, true);
		}

		case TopologyDataMesh::TOPOLOGY_DATA_MESH_TRIANGLE: {
			Ref<TriangleSubdivider> subdivider;
			subdivider.instantiate();
			return subdivider->get_subdivided_arrays(topology_arrays, p_level, p_format, true);
		}

		default:
			return Array();
	}
}

TypedArray<Array> SubdivisionBaker::get_baked_blend_shape_arrays(const Array &base_arrays, const Array &relative_topology_blend_shape_arrays,
		int32_t p_level, int64_t p_format, int topology_type) {
	Array blend_shape_arrays = base_arrays.duplicate(false);
	p_format &= ~Mesh::ARRAY_FORMAT_BONES;
	p_format &= ~Mesh::ARRAY_FORMAT_WEIGHTS;
	TypedArray<Array> baked_blend_shape_arrays;
	const PackedVector3Array &topology_vertex_array = base_arrays[TopologyDataMesh::ARRAY_VERTEX];
	for (int blend_shape_idx = 0; blend_shape_idx < relative_topology_blend_shape_arrays.size(); blend_shape_idx++) {
		const Array &single_blend_shape_arrays = relative_topology_blend_shape_arrays[blend_shape_idx];
		const PackedVector3Array &blend_shape_relative_vertex_array = single_blend_shape_arrays[0]; //expects relative data
		PackedVector3Array blend_shape_vertex_array_absolute = topology_vertex_array;
		for (int vertex_idx = 0; vertex_idx < topology_vertex_array.size(); vertex_idx++) {
			blend_shape_vertex_array_absolute.write[vertex_idx] += blend_shape_relative_vertex_array[vertex_idx];
		}

		blend_shape_arrays[Mesh::ARRAY_VERTEX] = blend_shape_vertex_array_absolute;

		Array full_baked_array = get_baked_arrays(blend_shape_arrays, p_level, p_format, topology_type);

		//Vertex, normal, tangent
		Array single_baked_blend_shape_array;
		single_baked_blend_shape_array.resize(Mesh::ARRAY_MAX);
		single_baked_blend_shape_array[Mesh::ARRAY_VERTEX] = full_baked_array[Mesh::ARRAY_VERTEX];
		single_baked_blend_shape_array[Mesh::ARRAY_NORMAL] = full_baked_array[Mesh::ARRAY_NORMAL];
		single_baked_blend_shape_array[Mesh::ARRAY_TANGENT] = full_baked_array[Mesh::ARRAY_TANGENT];
		baked_blend_shape_arrays.push_back(single_baked_blend_shape_array);
	}

	return baked_blend_shape_arrays;
}

Ref<ImporterMesh> SubdivisionBaker::get_importer_mesh(const Ref<ImporterMesh> &p_base, const Ref<TopologyDataMesh> &p_topology_data_mesh, int32_t p_level, bool bake_blendshapes) {
	Ref<ImporterMesh> mesh;
	if (p_base.is_valid()) {
		mesh = p_base;
	}
	if (mesh.is_null()) {
		mesh.instantiate();
	}

	if (bake_blendshapes) {
		for (int blend_shape_idx = 0; blend_shape_idx < p_topology_data_mesh->get_blend_shape_count(); blend_shape_idx++) {
			mesh->add_blend_shape(p_topology_data_mesh->get_blend_shape_name(blend_shape_idx));
		}
	}

	for (int surface_index = 0; surface_index < p_topology_data_mesh->get_surface_count(); surface_index++) {
		const Array &source_arrays = p_topology_data_mesh->surface_get_arrays(surface_index);
		int64_t p_format = p_topology_data_mesh->surface_get_format(surface_index);
		int topology_type = p_topology_data_mesh->surface_get_topology_type(surface_index);
		const String &surface_name = p_topology_data_mesh->surface_get_name(surface_index);
		const Ref<Material> &surface_material = p_topology_data_mesh->surface_get_material(surface_index);

		Array surface_baked_arrays = get_baked_arrays(source_arrays, p_level, p_format, topology_type);

		TypedArray<Array> baked_blend_shape_arrays;
		if (bake_blendshapes && p_topology_data_mesh->get_blend_shape_count() > 0) {
			baked_blend_shape_arrays = get_baked_blend_shape_arrays(source_arrays, p_topology_data_mesh->surface_get_blend_shape_arrays(surface_index),
					p_level, p_format, topology_type);
		}

		mesh->add_surface(Mesh::PRIMITIVE_TRIANGLES, surface_baked_arrays, baked_blend_shape_arrays, Dictionary(), surface_material, surface_name, p_format);
	}

	return mesh;
}

SubdivisionBaker::SubdivisionBaker() {}
SubdivisionBaker::~SubdivisionBaker() {}

void SubdivisionBaker::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_baked_arrays", "topology_arrays", "subdivision_level", "format", "topology_type"), &SubdivisionBaker::get_baked_arrays);
}
