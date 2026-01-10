/**************************************************************************/
/*  polygon_triangulation_godot.cpp                                       */
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

#include "polygon_triangulation_godot.h"

#include "scene/resources/3d/importer_mesh.h"

void PolygonTriangulationGodot::_bind_methods() {
	// Factory methods
	ClassDB::bind_static_method("PolygonTriangulationGodot", D_METHOD("create", "points", "normals"), &PolygonTriangulationGodot::create, DEFVAL(PackedVector3Array()));
	ClassDB::bind_static_method("PolygonTriangulationGodot", D_METHOD("create_planar", "points", "degenerate_points"), &PolygonTriangulationGodot::create_planar);

	// Configuration
	ClassDB::bind_method(D_METHOD("set_cost_weights", "triangle", "edge", "bi_triangle", "triangle_boundary", "worst_dihedral"), &PolygonTriangulationGodot::set_cost_weights);
	ClassDB::bind_method(D_METHOD("set_optimization_rounds", "rounds"), &PolygonTriangulationGodot::set_optimization_rounds);
	ClassDB::bind_method(D_METHOD("set_point_limit", "limit"), &PolygonTriangulationGodot::set_point_limit);
	ClassDB::bind_method(D_METHOD("enable_dot_output", "enable"), &PolygonTriangulationGodot::enable_dot_output);

	// Execute
	ClassDB::bind_method(D_METHOD("preprocess"), &PolygonTriangulationGodot::preprocess);
	ClassDB::bind_method(D_METHOD("triangulate"), &PolygonTriangulationGodot::triangulate);
	ClassDB::bind_method(D_METHOD("clear_cache"), &PolygonTriangulationGodot::clear_cache);

	// Results
	ClassDB::bind_method(D_METHOD("get_vertices"), &PolygonTriangulationGodot::get_vertices);
	ClassDB::bind_method(D_METHOD("get_indices"), &PolygonTriangulationGodot::get_indices);
	ClassDB::bind_method(D_METHOD("get_normals"), &PolygonTriangulationGodot::get_normals);
	ClassDB::bind_method(D_METHOD("get_mesh", "smooth", "subdivisions", "laplacian_iterations"), &PolygonTriangulationGodot::get_mesh, DEFVAL(false), DEFVAL(0), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_importer_mesh", "smooth", "subdivisions", "laplacian_iterations"), &PolygonTriangulationGodot::get_importer_mesh, DEFVAL(false), DEFVAL(0), DEFVAL(0));

	// Query
	ClassDB::bind_method(D_METHOD("get_triangle_count"), &PolygonTriangulationGodot::get_triangle_count);
	ClassDB::bind_method(D_METHOD("get_vertex_count"), &PolygonTriangulationGodot::get_vertex_count);
	ClassDB::bind_method(D_METHOD("get_statistics"), &PolygonTriangulationGodot::get_statistics);
	ClassDB::bind_method(D_METHOD("get_optimal_cost"), &PolygonTriangulationGodot::get_optimal_cost);
}

Ref<PolygonTriangulationGodot> PolygonTriangulationGodot::create(const PackedVector3Array &p_points, const PackedVector3Array &p_normals) {
	ERR_FAIL_COND_V_MSG(p_points.size() < 3, Ref<PolygonTriangulationGodot>(), "At least 3 points are required for triangulation.");

	Ref<PolygonTriangulationGodot> wrapper;
	wrapper.instantiate();

	int point_count = p_points.size();
	double *points = new double[point_count * 3];
	float *normals = nullptr;

	// Convert PackedVector3Array to C array
	for (int i = 0; i < point_count; i++) {
		Vector3 p = p_points[i];
		points[i * 3] = p.x;
		points[i * 3 + 1] = p.y;
		points[i * 3 + 2] = p.z;
	}

	// Convert normals if provided
	if (p_normals.size() > 0) {
		ERR_FAIL_COND_V_MSG(p_normals.size() != p_points.size(), Ref<PolygonTriangulationGodot>(), "Normals array must match points array size.");
		normals = new float[point_count * 3];
		for (int i = 0; i < point_count; i++) {
			Vector3 n = p_normals[i];
			normals[i * 3] = n.x;
			normals[i * 3 + 1] = n.y;
			normals[i * 3 + 2] = n.z;
		}
		wrapper->triangulator = PolygonTriangulation::_create_with_normals(point_count, points, nullptr, normals, false);
		delete[] normals;
	} else {
		wrapper->triangulator = PolygonTriangulation::_create_with_degenerates(point_count, points, nullptr, false);
	}

	delete[] points;
	return wrapper;
}

Ref<PolygonTriangulationGodot> PolygonTriangulationGodot::create_planar(const PackedVector3Array &p_points, const PackedVector3Array &p_degenerate_points) {
	ERR_FAIL_COND_V_MSG(p_points.size() < 3, Ref<PolygonTriangulationGodot>(), "At least 3 points are required for triangulation.");
	ERR_FAIL_COND_V_MSG(p_degenerate_points.size() != p_points.size(), Ref<PolygonTriangulationGodot>(), "Degenerate points array must match points array size.");

	Ref<PolygonTriangulationGodot> wrapper;
	wrapper.instantiate();

	int point_count = p_points.size();
	double *points = new double[point_count * 3];
	double *degen_points = new double[point_count * 3];

	for (int i = 0; i < point_count; i++) {
		Vector3 p = p_points[i];
		points[i * 3] = p.x;
		points[i * 3 + 1] = p.y;
		points[i * 3 + 2] = p.z;

		Vector3 d = p_degenerate_points[i];
		degen_points[i * 3] = d.x;
		degen_points[i * 3 + 1] = d.y;
		degen_points[i * 3 + 2] = d.z;
	}

	wrapper->triangulator = PolygonTriangulation::_create_with_degenerates(point_count, points, degen_points, true);

	delete[] points;
	delete[] degen_points;
	return wrapper;
}

void PolygonTriangulationGodot::set_cost_weights(float p_triangle, float p_edge, float p_bi_triangle, float p_triangle_boundary, float p_worst_dihedral) {
	ERR_FAIL_COND_MSG(triangulator.is_null(), "Triangulator not initialized.");
	triangulator->set_weights(p_triangle, p_edge, p_bi_triangle, p_triangle_boundary, p_worst_dihedral);
}

void PolygonTriangulationGodot::set_optimization_rounds(int p_rounds) {
	ERR_FAIL_COND_MSG(triangulator.is_null(), "Triangulator not initialized.");
	triangulator->set_round(p_rounds);
}

void PolygonTriangulationGodot::set_point_limit(int p_limit) {
	ERR_FAIL_COND_MSG(triangulator.is_null(), "Triangulator not initialized.");
	triangulator->set_point_limit(p_limit);
}

void PolygonTriangulationGodot::enable_dot_output(bool p_enable) {
	ERR_FAIL_COND_MSG(triangulator.is_null(), "Triangulator not initialized.");
	triangulator->set_dot(p_enable);
}

bool PolygonTriangulationGodot::preprocess() {
	ERR_FAIL_COND_V_MSG(triangulator.is_null(), false, "Triangulator not initialized.");
	triangulator->preprocess();
	return true;
}

bool PolygonTriangulationGodot::triangulate() {
	ERR_FAIL_COND_V_MSG(triangulator.is_null(), false, "Triangulator not initialized.");
	has_cached_result = false;
	return triangulator->start();
}

void PolygonTriangulationGodot::clear_cache() {
	has_cached_result = false;
	cached_vertices.clear();
	cached_indices.clear();
	cached_normals.clear();
	if (triangulator.is_valid()) {
		triangulator->clear_tiling();
	}
}

PackedVector3Array PolygonTriangulationGodot::get_vertices() const {
	ERR_FAIL_COND_V_MSG(triangulator.is_null(), PackedVector3Array(), "Triangulator not initialized.");

	if (has_cached_result) {
		return cached_vertices;
	}

	double *out_faces = nullptr;
	int out_num = 0;
	double *out_points = nullptr;
	float *out_norms = nullptr;
	int out_pn = 0;

	const_cast<PolygonTriangulation *>(triangulator.ptr())->get_result(&out_faces, &out_num, &out_points, &out_norms, &out_pn, false, 0, 0);

	PackedVector3Array vertices;
	vertices.resize(out_pn);
	for (int i = 0; i < out_pn; i++) {
		vertices.write[i] = Vector3(out_points[i * 3], out_points[i * 3 + 1], out_points[i * 3 + 2]);
	}

	const_cast<PolygonTriangulationGodot *>(this)->cached_vertices = vertices;
	const_cast<PolygonTriangulationGodot *>(this)->has_cached_result = true;

	return vertices;
}

PackedInt32Array PolygonTriangulationGodot::get_indices() const {
	ERR_FAIL_COND_V_MSG(triangulator.is_null(), PackedInt32Array(), "Triangulator not initialized.");

	if (has_cached_result && cached_indices.size() > 0) {
		return cached_indices;
	}

	double *out_faces = nullptr;
	int out_num = 0;
	double *out_points = nullptr;
	float *out_norms = nullptr;
	int out_pn = 0;

	const_cast<PolygonTriangulation *>(triangulator.ptr())->get_result(&out_faces, &out_num, &out_points, &out_norms, &out_pn, false, 0, 0);

	PackedInt32Array indices;
	indices.resize(out_num * 3);
	for (int i = 0; i < out_num * 3; i++) {
		indices.write[i] = static_cast<int>(out_faces[i]);
	}

	const_cast<PolygonTriangulationGodot *>(this)->cached_indices = indices;
	return indices;
}

PackedVector3Array PolygonTriangulationGodot::get_normals() const {
	ERR_FAIL_COND_V_MSG(triangulator.is_null(), PackedVector3Array(), "Triangulator not initialized.");

	if (has_cached_result && cached_normals.size() > 0) {
		return cached_normals;
	}

	double *out_faces = nullptr;
	int out_num = 0;
	double *out_points = nullptr;
	float *out_norms = nullptr;
	int out_pn = 0;

	const_cast<PolygonTriangulation *>(triangulator.ptr())->get_result(&out_faces, &out_num, &out_points, &out_norms, &out_pn, false, 0, 0);

	PackedVector3Array normals;
	if (out_norms != nullptr) {
		normals.resize(out_pn);
		for (int i = 0; i < out_pn; i++) {
			normals.write[i] = Vector3(out_norms[i * 3], out_norms[i * 3 + 1], out_norms[i * 3 + 2]);
		}
	}

	const_cast<PolygonTriangulationGodot *>(this)->cached_normals = normals;
	return normals;
}

Ref<ArrayMesh> PolygonTriangulationGodot::get_mesh(bool p_smooth, int p_subdivisions, int p_laplacian_iterations) const {
	ERR_FAIL_COND_V_MSG(triangulator.is_null(), Ref<ArrayMesh>(), "Triangulator not initialized.");

	double *out_faces = nullptr;
	int out_num = 0;
	double *out_points = nullptr;
	float *out_norms = nullptr;
	int out_pn = 0;

	const_cast<PolygonTriangulation *>(triangulator.ptr())->get_result(&out_faces, &out_num, &out_points, &out_norms, &out_pn, p_smooth, p_subdivisions, p_laplacian_iterations);
	Ref<ArrayMesh> mesh;

	// If RenderingServer is unavailable (headless test), bail out gracefully.
	if (!RenderingServer::get_singleton()) {
		WARN_PRINT_ONCE("RenderingServer not available; returning empty mesh. Use get_importer_mesh() for headless access.");
		return mesh;
	}

	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);

	PackedVector3Array vertices;
	vertices.resize(out_pn);
	for (int i = 0; i < out_pn; i++) {
		vertices.write[i] = Vector3(out_points[i * 3], out_points[i * 3 + 1], out_points[i * 3 + 2]);
	}
	arrays[Mesh::ARRAY_VERTEX] = vertices;

	PackedInt32Array indices;
	indices.resize(out_num * 3);
	for (int tri_idx = 0; tri_idx < out_num; tri_idx++) {
		for (int vert_idx = 0; vert_idx < 3; vert_idx++) {
			double vx = out_faces[tri_idx * 9 + vert_idx * 3 + 0];
			double vy = out_faces[tri_idx * 9 + vert_idx * 3 + 1];
			double vz = out_faces[tri_idx * 9 + vert_idx * 3 + 2];

			int found_idx = -1;
			for (int pt_idx = 0; pt_idx < out_pn; pt_idx++) {
				if (Math::is_equal_approx(out_points[pt_idx * 3 + 0], vx) &&
						Math::is_equal_approx(out_points[pt_idx * 3 + 1], vy) &&
						Math::is_equal_approx(out_points[pt_idx * 3 + 2], vz)) {
					found_idx = pt_idx;
					break;
				}
			}

			ERR_FAIL_COND_V_MSG(found_idx == -1, Ref<ArrayMesh>(),
					vformat("Failed to find vertex index for triangle %d vertex %d", tri_idx, vert_idx));

			indices.write[tri_idx * 3 + vert_idx] = found_idx;
		}
	}
	arrays[Mesh::ARRAY_INDEX] = indices;

	if (out_norms != nullptr) {
		PackedVector3Array normals;
		normals.resize(out_pn);
		for (int i = 0; i < out_pn; i++) {
			normals.write[i] = Vector3(out_norms[i * 3], out_norms[i * 3 + 1], out_norms[i * 3 + 2]);
		}
		arrays[Mesh::ARRAY_NORMAL] = normals;
	}

	// Cache results for count helpers.
	const_cast<PolygonTriangulationGodot *>(this)->cached_vertices = vertices;
	const_cast<PolygonTriangulationGodot *>(this)->cached_indices = indices;
	const_cast<PolygonTriangulationGodot *>(this)->cached_normals = arrays[Mesh::ARRAY_NORMAL];
	const_cast<PolygonTriangulationGodot *>(this)->has_cached_result = true;

	mesh.instantiate();
	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
	return mesh;
}

Ref<ImporterMesh> PolygonTriangulationGodot::get_importer_mesh(bool p_smooth, int p_subdivisions, int p_laplacian_iterations) const {
	ERR_FAIL_COND_V_MSG(triangulator.is_null(), Ref<ImporterMesh>(), "Triangulator not initialized.");

	double *out_faces = nullptr;
	int out_num = 0;
	double *out_points = nullptr;
	float *out_norms = nullptr;
	int out_pn = 0;

	const_cast<PolygonTriangulation *>(triangulator.ptr())->get_result(&out_faces, &out_num, &out_points, &out_norms, &out_pn, p_smooth, p_subdivisions, p_laplacian_iterations);

	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);

	PackedVector3Array vertices;
	vertices.resize(out_pn);
	for (int i = 0; i < out_pn; i++) {
		vertices.write[i] = Vector3(out_points[i * 3], out_points[i * 3 + 1], out_points[i * 3 + 2]);
	}
	arrays[Mesh::ARRAY_VERTEX] = vertices;

	PackedInt32Array indices;
	indices.resize(out_num * 3);
	for (int tri_idx = 0; tri_idx < out_num; tri_idx++) {
		for (int vert_idx = 0; vert_idx < 3; vert_idx++) {
			double vx = out_faces[tri_idx * 9 + vert_idx * 3 + 0];
			double vy = out_faces[tri_idx * 9 + vert_idx * 3 + 1];
			double vz = out_faces[tri_idx * 9 + vert_idx * 3 + 2];

			int found_idx = -1;
			for (int pt_idx = 0; pt_idx < out_pn; pt_idx++) {
				if (Math::is_equal_approx(out_points[pt_idx * 3 + 0], vx) &&
						Math::is_equal_approx(out_points[pt_idx * 3 + 1], vy) &&
						Math::is_equal_approx(out_points[pt_idx * 3 + 2], vz)) {
					found_idx = pt_idx;
					break;
				}
			}

			if (found_idx == -1) {
				print_line(vformat("[Cassie] get_importer_mesh: failed to find index for tri=%d vert=%d", tri_idx, vert_idx));
				ERR_FAIL_COND_V_MSG(found_idx == -1, Ref<ImporterMesh>(),
						vformat("Failed to find vertex index for triangle %d vertex %d", tri_idx, vert_idx));
			}

			indices.write[tri_idx * 3 + vert_idx] = found_idx;
		}
	}
	arrays[Mesh::ARRAY_INDEX] = indices;

	if (out_norms != nullptr) {
		PackedVector3Array normals;
		normals.resize(out_pn);
		for (int i = 0; i < out_pn; i++) {
			normals.write[i] = Vector3(out_norms[i * 3], out_norms[i * 3 + 1], out_norms[i * 3 + 2]);
		}
		arrays[Mesh::ARRAY_NORMAL] = normals;
	}

	Ref<ImporterMesh> importer_mesh;
	importer_mesh.instantiate();
	importer_mesh->add_surface(Mesh::PRIMITIVE_TRIANGLES, arrays);

	// Free the buffers allocated by get_result
	delete[] out_faces;
	delete[] out_points;
	if (out_norms != nullptr) {
		delete[] out_norms;
	}

	return importer_mesh;
}

int PolygonTriangulationGodot::get_triangle_count() const {
	if (cached_indices.size() == 0 && triangulator.is_valid()) {
		const_cast<PolygonTriangulationGodot *>(this)->get_indices();
	}
	return cached_indices.size() / 3;
}

int PolygonTriangulationGodot::get_vertex_count() const {
	if (cached_vertices.size() == 0 && triangulator.is_valid()) {
		const_cast<PolygonTriangulationGodot *>(this)->get_vertices();
	}
	return cached_vertices.size();
}

Dictionary PolygonTriangulationGodot::get_statistics() const {
	Dictionary stats;
	ERR_FAIL_COND_V_MSG(triangulator.is_null(), stats, "Triangulator not initialized.");

	stats["optimal_cost"] = triangulator->optimalCost;
	stats["triangle_count"] = get_triangle_count();
	stats["vertex_count"] = get_vertex_count();

	return stats;
}

float PolygonTriangulationGodot::get_optimal_cost() const {
	ERR_FAIL_COND_V_MSG(triangulator.is_null(), 0.0f, "Triangulator not initialized.");
	return triangulator->optimalCost;
}

PolygonTriangulationGodot::PolygonTriangulationGodot() {
}

PolygonTriangulationGodot::~PolygonTriangulationGodot() {
}
