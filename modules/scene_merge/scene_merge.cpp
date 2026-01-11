/**************************************************************************/
/*  scene_merge.cpp                                                       */
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

#include "scene_merge.h"

#include "modules/scene_merge/merge.h"
#include "scene/resources/mesh.h"
#include "thirdparty/xatlas/xatlas.h"

Node *SceneMerge::merge(Node *p_root_node) {
	return MeshTextureAtlas::merge_meshes(p_root_node);
}

Ref<ImporterMesh> SceneMerge::unwrap_mesh(Ref<ImporterMesh> p_mesh, float p_texel_density, int p_resolution, int p_padding) {
	ERR_FAIL_NULL_V(p_mesh, Ref<ImporterMesh>());

	Ref<ImporterMesh> result_mesh;
	result_mesh.instantiate();

	// Process each surface
	for (int surface_i = 0; surface_i < p_mesh->get_surface_count(); surface_i++) {
		Array surface_arrays = p_mesh->get_surface_arrays(surface_i);

		// Get mesh data
		Vector<Vector3> vertices = surface_arrays[Mesh::ARRAY_VERTEX];
		Vector<Vector3> normals = surface_arrays[Mesh::ARRAY_NORMAL];
		Vector<int> indices = surface_arrays[Mesh::ARRAY_INDEX];

		if (vertices.is_empty()) {
			continue;
		}

		// Prepare xatlas mesh
		xatlas::MeshDecl mesh_decl;
		mesh_decl.vertexCount = vertices.size();
		mesh_decl.vertexPositionData = vertices.ptr();
		mesh_decl.vertexPositionStride = sizeof(Vector3);
		mesh_decl.vertexNormalData = normals.ptr();
		mesh_decl.vertexNormalStride = sizeof(Vector3);

		if (!indices.is_empty()) {
			mesh_decl.indexCount = indices.size();
			mesh_decl.indexData = indices.ptr();
			mesh_decl.indexFormat = xatlas::IndexFormat::UInt32;
		}

		// Create atlas
		xatlas::Atlas *atlas = xatlas::Create();

		// Set chart options
		xatlas::ChartOptions chart_options;
		if (p_texel_density > 0.0f) {
			chart_options.maxChartArea = 1.0f / (p_texel_density * p_texel_density);
		}

		// Set pack options
		xatlas::PackOptions pack_options;
		pack_options.padding = p_padding;
		if (p_resolution > 0) {
			pack_options.resolution = p_resolution;
		}

		// Add mesh to atlas
		xatlas::AddMeshError error = xatlas::AddMesh(atlas, mesh_decl);
		if (error != xatlas::AddMeshError::Success) {
			xatlas::Destroy(atlas);
			WARN_PRINT("Failed to add mesh to xatlas atlas");
			continue;
		}

		// Generate atlas
		xatlas::Generate(atlas, chart_options, pack_options);

		// Get UVs from atlas
		const xatlas::Mesh &atlas_mesh = atlas->meshes[0];
		Vector<Vector2> uvs;
		uvs.resize(vertices.size());

		for (uint32_t i = 0; i < atlas_mesh.vertexCount; i++) {
			const xatlas::Vertex &vertex = atlas_mesh.vertexArray[i];
			if (vertex.xref < (uint32_t)vertices.size()) {
				uvs.write[vertex.xref] = Vector2(vertex.uv[0], vertex.uv[1]);
			}
		}

		// Update surface arrays with new UVs
		surface_arrays[Mesh::ARRAY_TEX_UV] = uvs;

		// Add surface to result mesh
		Mesh::PrimitiveType primitive = p_mesh->get_surface_primitive_type(surface_i);
		Ref<Material> material = p_mesh->get_surface_material(surface_i);
		String name = p_mesh->get_surface_name(surface_i);
		uint64_t flags = p_mesh->get_surface_format(surface_i);

		// Get blend shapes and LODs
		TypedArray<Array> blend_shapes;
		for (int bs_i = 0; bs_i < p_mesh->get_blend_shape_count(); bs_i++) {
			blend_shapes.append(p_mesh->get_surface_blend_shape_arrays(surface_i, bs_i));
		}

		Dictionary lods;
		for (int lod_i = 0; lod_i < p_mesh->get_surface_lod_count(surface_i); lod_i++) {
			Dictionary lod_dict;
			lod_dict["indices"] = p_mesh->get_surface_lod_indices(surface_i, lod_i);
			lod_dict["distance"] = p_mesh->get_surface_lod_size(surface_i, lod_i);
			lods[p_mesh->get_surface_lod_size(surface_i, lod_i)] = lod_dict;
		}

		result_mesh->add_surface(primitive, surface_arrays, blend_shapes, lods, material, name, flags);

		xatlas::Destroy(atlas);
	}

	return result_mesh;
}

void SceneMerge::_bind_methods() {
	ClassDB::bind_method(D_METHOD("merge", "root_node"), &SceneMerge::merge);
	ClassDB::bind_method(D_METHOD("unwrap_mesh", "mesh", "texel_density", "resolution", "padding"), &SceneMerge::unwrap_mesh, DEFVAL(-1.0f), DEFVAL(512), DEFVAL(1));
}
