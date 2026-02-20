/**************************************************************************/
/*  register_types.cpp                                                    */
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
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "register_types.h"

#include "scene/resources/mesh.h"

#include <manifold/manifold.h>
#include "coacd.h"

#include "core/math/geometry_3d.h"

// Build a single convex hull from input points using Manifold::Hull (no ConvexHullComputer).
static Vector<Vector<Vector3>> fallback_single_hull_manifold(const real_t *p_vertices, int p_vertex_count) {
	if (p_vertex_count < 4) {
		return Vector<Vector<Vector3>>();
	}
	std::vector<manifold::vec3> pts;
	pts.reserve((size_t)p_vertex_count);
	for (int i = 0; i < p_vertex_count; i++) {
		pts.push_back(manifold::vec3(p_vertices[i * 3], p_vertices[i * 3 + 1], p_vertices[i * 3 + 2]));
	}
	manifold::Manifold m = manifold::Manifold::Hull(pts);
	if (m.Status() != manifold::Manifold::Error::NoError || m.IsEmpty()) {
		return Vector<Vector<Vector3>>();
	}
	manifold::MeshGL64 mesh = m.GetMeshGL64();
	const int64_t nv = (int64_t)mesh.NumVert();
	if (nv < 4 || mesh.numProp < 3) {
		return Vector<Vector<Vector3>>();
	}
	Vector<Vector3> hull_verts;
	hull_verts.resize((int)nv);
	for (int64_t i = 0; i < nv; i++) {
		const double *p = mesh.vertProperties.data() + (size_t)(i * mesh.numProp);
		hull_verts.write[(int)i] = Vector3((real_t)p[0], (real_t)p[1], (real_t)p[2]);
	}
	Vector<Vector<Vector3>> one;
	one.push_back(hull_verts);
	return one;
}

static Vector<Vector<Vector3>> convex_decompose_coacd(const real_t *p_vertices, int p_vertex_count, const uint32_t *p_triangles, int p_triangle_count, const Ref<MeshConvexDecompositionSettings> &p_settings, Vector<Vector<uint32_t>> *r_convex_indices) {
	(void)r_convex_indices;

	const int vert_count = p_vertex_count;
	const int tri_count = p_triangle_count;
	const int n_verts = vert_count;
	const int n_tris = tri_count;

	// 1. Build manifold::MeshGL64 (position-only, numProp=3)
	manifold::MeshGL64 mesh_in;
	mesh_in.numProp = 3;
	mesh_in.vertProperties.resize((size_t)vert_count * 3);
	mesh_in.triVerts.resize((size_t)tri_count * 3);
	for (int i = 0; i < vert_count; i++) {
		mesh_in.vertProperties[(size_t)i * 3 + 0] = (double)p_vertices[i * 3];
		mesh_in.vertProperties[(size_t)i * 3 + 1] = (double)p_vertices[i * 3 + 1];
		mesh_in.vertProperties[(size_t)i * 3 + 2] = (double)p_vertices[i * 3 + 2];
	}
	for (int i = 0; i < tri_count * 3; i++) {
		mesh_in.triVerts[(size_t)i] = (uint64_t)p_triangles[i];
	}
	mesh_in.Merge();
	manifold::Manifold m(mesh_in);

	if (m.Status() != manifold::Manifold::Error::NoError) {
		return fallback_single_hull_manifold(p_vertices, vert_count);
	}
	if (n_verts < 4 || n_tris <= 0) {
		return fallback_single_hull_manifold(p_vertices, vert_count);
	}

	manifold::MeshGL64 mesh_merged = m.GetMeshGL64();
	const int64_t merged_vert_count = (int64_t)mesh_merged.NumVert();
	const int64_t merged_tri_count = (int64_t)mesh_merged.NumTri();
	if (merged_vert_count < 4 || merged_tri_count <= 0 || mesh_merged.numProp < 3) {
		return fallback_single_hull_manifold(p_vertices, vert_count);
	}

	// 2. Map settings to CoACD parameters
	double threshold = 0.05;
	int max_convex_hull = -1;
	bool pca = false;
	if (p_settings.is_valid()) {
		threshold = (double)CLAMP(p_settings->get_max_concavity(), 0.01, 1.0);
		max_convex_hull = (int)p_settings->get_max_convex_hulls();
		pca = p_settings->get_normalize_mesh();
	}
	if (max_convex_hull <= 0) {
		max_convex_hull = -1;
	}

	// 3. Build coacd::Mesh from Manifold output (preprocess="off" - we did Manifold already)
	coacd::Mesh input;
	input.vertices.resize((size_t)merged_vert_count);
	for (int64_t i = 0; i < merged_vert_count; i++) {
		const double *p = mesh_merged.vertProperties.data() + (size_t)(i * mesh_merged.numProp);
		input.vertices[(size_t)i] = { p[0], p[1], p[2] };
	}
	input.indices.resize((size_t)merged_tri_count);
	for (int64_t i = 0; i < merged_tri_count; i++) {
		input.indices[(size_t)i][0] = (int)mesh_merged.triVerts[(size_t)(i * 3)];
		input.indices[(size_t)i][1] = (int)mesh_merged.triVerts[(size_t)(i * 3 + 1)];
		input.indices[(size_t)i][2] = (int)mesh_merged.triVerts[(size_t)(i * 3 + 2)];
	}

	std::vector<coacd::Mesh> parts = coacd::CoACD(input, threshold, max_convex_hull, "off", 50, 2000, 20, 150, 3, pca, true, false, 256, false, 0.01, "ch", 0);

	if (parts.empty()) {
		return fallback_single_hull_manifold(p_vertices, vert_count);
	}

	// 4. Convert CoACD result to Vector<Vector<Vector3>>
	Vector<Vector<Vector3>> result;
	result.resize((int)parts.size());
	for (size_t i = 0; i < parts.size(); i++) {
		const coacd::Mesh &part = parts[i];
		Vector<Vector3> &out = result.write[(int)i];
		out.resize((int)part.vertices.size());
		for (size_t j = 0; j < part.vertices.size(); j++) {
			out.write[(int)j] = Vector3((real_t)part.vertices[j][0], (real_t)part.vertices[j][1], (real_t)part.vertices[j][2]);
		}
	}
	return result;
}

void initialize_coacd_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	Mesh::register_convex_decomposition_backend(MeshConvexDecompositionSettings::CONVEX_DECOMPOSITION_BACKEND_COACD, convex_decompose_coacd);
}

void uninitialize_coacd_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	Mesh::unregister_convex_decomposition_backend(MeshConvexDecompositionSettings::CONVEX_DECOMPOSITION_BACKEND_COACD);
}
