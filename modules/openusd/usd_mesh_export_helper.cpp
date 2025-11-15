/**************************************************************************/
/*  usd_mesh_export_helper.cpp                                            */
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

#include "usd_mesh_export_helper.h"

// TODO: Update export functionality to use TinyUSDZ API
tinyusdz::Prim UsdMeshExportHelper::export_mesh_to_prim(const Ref<Mesh> p_mesh, tinyusdz::Stage *p_stage, const tinyusdz::Path &p_path) {
	ERR_PRINT("USD Export: export_mesh_to_prim not yet implemented with TinyUSDZ");
	// TODO: Create a TinyUSDZ Prim from Godot Mesh
	// This will need to use TinyUSDZ's Prim creation API
	// Create an empty Model prim as placeholder
	tinyusdz::Model model;
	return tinyusdz::Prim(model);
}

tinyusdz::Prim UsdMeshExportHelper::export_box(const Ref<BoxMesh> p_box, tinyusdz::Stage *p_stage, const tinyusdz::Path &p_path) {
	ERR_PRINT("USD Export: export_box not yet implemented with TinyUSDZ");
	// TODO: Create TinyUSDZ GeomCube prim
	tinyusdz::Model model;
	return tinyusdz::Prim(model);
}

tinyusdz::Prim UsdMeshExportHelper::export_sphere(const Ref<SphereMesh> p_sphere, tinyusdz::Stage *p_stage, const tinyusdz::Path &p_path) {
	ERR_PRINT("USD Export: export_sphere not yet implemented with TinyUSDZ");
	// TODO: Create TinyUSDZ GeomSphere prim
	tinyusdz::Model model;
	return tinyusdz::Prim(model);
}

tinyusdz::Prim UsdMeshExportHelper::export_cylinder(const Ref<CylinderMesh> p_cylinder, tinyusdz::Stage *p_stage, const tinyusdz::Path &p_path) {
	ERR_PRINT("USD Export: export_cylinder not yet implemented with TinyUSDZ");
	// TODO: Create TinyUSDZ GeomCylinder prim
	tinyusdz::Model model;
	return tinyusdz::Prim(model);
}

tinyusdz::Prim UsdMeshExportHelper::export_cone(const Ref<CylinderMesh> p_cone, tinyusdz::Stage *p_stage, const tinyusdz::Path &p_path) {
	ERR_PRINT("USD Export: export_cone not yet implemented with TinyUSDZ");
	// TODO: Create TinyUSDZ GeomCone prim
	tinyusdz::Model model;
	return tinyusdz::Prim(model);
}

tinyusdz::Prim UsdMeshExportHelper::export_capsule(const Ref<CapsuleMesh> p_capsule, tinyusdz::Stage *p_stage, const tinyusdz::Path &p_path) {
	ERR_PRINT("USD Export: export_capsule not yet implemented with TinyUSDZ");
	// TODO: Create TinyUSDZ GeomCapsule prim
	tinyusdz::Model model;
	return tinyusdz::Prim(model);
}

tinyusdz::Prim UsdMeshExportHelper::export_geom_mesh(const Ref<Mesh> p_mesh, tinyusdz::Stage *p_stage, const tinyusdz::Path &p_path) {
	if (p_mesh.is_null()) {
		ERR_PRINT("USD Export: Invalid mesh");
		tinyusdz::Model model;
		return tinyusdz::Prim(model);
	}

	// Get the first surface (minimum working subset - only export first surface)
	int surface_count = p_mesh->get_surface_count();
	if (surface_count == 0) {
		ERR_PRINT("USD Export: Mesh has no surfaces");
		tinyusdz::Model model;
		return tinyusdz::Prim(model);
	}

	Array arrays = p_mesh->surface_get_arrays(0);
	if (arrays.size() == 0) {
		ERR_PRINT("USD Export: Failed to get surface arrays");
		tinyusdz::Model model;
		return tinyusdz::Prim(model);
	}

	// Get vertices
	PackedVector3Array vertices = arrays[Mesh::ARRAY_VERTEX];
	if (vertices.size() == 0) {
		ERR_PRINT("USD Export: Mesh has no vertices");
		tinyusdz::Model model;
		return tinyusdz::Prim(model);
	}

	// Create GeomMesh
	tinyusdz::GeomMesh mesh;
	mesh.name = p_path.prim_part(); // Use path as name
	
	// Convert vertices to USD points
	std::vector<tinyusdz::value::point3f> usd_points;
	usd_points.reserve(vertices.size());
	for (int i = 0; i < vertices.size(); i++) {
		Vector3 v = vertices[i];
		usd_points.push_back({float(v.x), float(v.y), float(v.z)});
	}
	mesh.points.set_value(usd_points);

	// Get indices
	PackedInt32Array indices = arrays[Mesh::ARRAY_INDEX];
	if (indices.size() > 0) {
		// Convert indices to USD format (face vertex counts and indices)
		// Assuming triangles (minimum working subset)
		int face_count = indices.size() / 3;
		
		std::vector<int32_t> face_vertex_counts;
		face_vertex_counts.reserve(face_count);
		for (int i = 0; i < face_count; i++) {
			face_vertex_counts.push_back(3); // 3 vertices per triangle
		}
		mesh.faceVertexCounts.set_value(face_vertex_counts);

		std::vector<int32_t> face_vertex_indices;
		face_vertex_indices.reserve(indices.size());
		for (int i = 0; i < indices.size(); i++) {
			face_vertex_indices.push_back(indices[i]);
		}
		mesh.faceVertexIndices.set_value(face_vertex_indices);
	}

	// Get normals if available (simplified - only per-vertex)
	PackedVector3Array normals = arrays[Mesh::ARRAY_NORMAL];
	if (normals.size() > 0 && normals.size() == vertices.size()) {
		std::vector<tinyusdz::value::normal3f> usd_normals;
		usd_normals.reserve(normals.size());
		for (int i = 0; i < normals.size(); i++) {
			Vector3 n = normals[i];
			usd_normals.push_back({float(n.x), float(n.y), float(n.z)});
		}
		mesh.normals.set_value(usd_normals);
	}

	return tinyusdz::Prim(mesh);
}

void UsdMeshExportHelper::apply_non_uniform_scale(tinyusdz::Prim &p_prim, const tinyusdz::value::float3 &p_scale) {
	ERR_PRINT("USD Export: apply_non_uniform_scale not yet implemented with TinyUSDZ");
	// TODO: Apply non-uniform scaling to TinyUSDZ prim
}
