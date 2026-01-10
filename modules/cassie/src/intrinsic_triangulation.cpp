/**************************************************************************/
/*  intrinsic_triangulation.cpp                                           */
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

#include "intrinsic_triangulation.h"

void IntrinsicTriangulation::_bind_methods() {
	// Initialize
	ClassDB::bind_method(D_METHOD("set_mesh_data", "vertices", "indices", "normals"), &IntrinsicTriangulation::set_mesh_data, DEFVAL(PackedVector3Array()));
	ClassDB::bind_method(D_METHOD("set_mesh", "mesh", "surface"), &IntrinsicTriangulation::set_mesh, DEFVAL(0));

	// Configuration
	ClassDB::bind_method(D_METHOD("set_max_flip_iterations", "iterations"), &IntrinsicTriangulation::set_max_flip_iterations);
	ClassDB::bind_method(D_METHOD("get_max_flip_iterations"), &IntrinsicTriangulation::get_max_flip_iterations);
	ClassDB::bind_method(D_METHOD("set_angle_threshold", "angle"), &IntrinsicTriangulation::set_angle_threshold);
	ClassDB::bind_method(D_METHOD("get_angle_threshold"), &IntrinsicTriangulation::get_angle_threshold);
	ClassDB::bind_method(D_METHOD("use_delaunay_flips", "enable"), &IntrinsicTriangulation::use_delaunay_flips);

	// Execute
	ClassDB::bind_method(D_METHOD("flip_to_delaunay"), &IntrinsicTriangulation::flip_to_delaunay);
	ClassDB::bind_method(D_METHOD("refine_intrinsic_triangulation", "target_edge_length"), &IntrinsicTriangulation::refine_intrinsic_triangulation, DEFVAL(-1.0f));
	ClassDB::bind_method(D_METHOD("smooth_intrinsic_positions", "iterations"), &IntrinsicTriangulation::smooth_intrinsic_positions, DEFVAL(5));

	// Results
	ClassDB::bind_method(D_METHOD("get_vertices"), &IntrinsicTriangulation::get_vertices);
	ClassDB::bind_method(D_METHOD("get_indices"), &IntrinsicTriangulation::get_indices);
	ClassDB::bind_method(D_METHOD("get_normals"), &IntrinsicTriangulation::get_normals);
	ClassDB::bind_method(D_METHOD("get_mesh"), &IntrinsicTriangulation::get_mesh);

	// Query
	ClassDB::bind_method(D_METHOD("get_triangle_count"), &IntrinsicTriangulation::get_triangle_count);
	ClassDB::bind_method(D_METHOD("get_edge_count"), &IntrinsicTriangulation::get_edge_count);
	ClassDB::bind_method(D_METHOD("get_vertex_count"), &IntrinsicTriangulation::get_vertex_count);
	ClassDB::bind_method(D_METHOD("get_statistics"), &IntrinsicTriangulation::get_statistics);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_flip_iterations"), "set_max_flip_iterations", "get_max_flip_iterations");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angle_threshold"), "set_angle_threshold", "get_angle_threshold");
}

void IntrinsicTriangulation::set_mesh_data(const PackedVector3Array &p_vertices, const PackedInt32Array &p_indices, const PackedVector3Array &p_normals) {
	ERR_FAIL_COND_MSG(p_vertices.size() < 3, "At least 3 vertices required.");
	ERR_FAIL_COND_MSG(p_indices.size() < 3 || p_indices.size() % 3 != 0, "Indices must be triangles (multiple of 3).");

	vertices = p_vertices;
	indices = p_indices;
	normals = p_normals;

	build_initial_intrinsic_mesh();
}

void IntrinsicTriangulation::set_mesh(const Ref<ArrayMesh> &p_mesh, int p_surface) {
	ERR_FAIL_COND_MSG(p_mesh.is_null(), "Mesh is null.");
	ERR_FAIL_INDEX_MSG(p_surface, p_mesh->get_surface_count(), "Surface index out of range.");

	Array arrays = p_mesh->surface_get_arrays(p_surface);
	vertices = arrays[Mesh::ARRAY_VERTEX];
	indices = arrays[Mesh::ARRAY_INDEX];

	if (arrays[Mesh::ARRAY_NORMAL] != Variant()) {
		normals = arrays[Mesh::ARRAY_NORMAL];
	}

	build_initial_intrinsic_mesh();
}

void IntrinsicTriangulation::build_initial_intrinsic_mesh() {
	edges.clear();
	triangles.clear();
	vertex_to_triangle.clear();

	int tri_count = indices.size() / 3;
	triangles.resize(tri_count);

	// Build triangles and edges
	HashMap<uint64_t, int> edge_map; // Map (v0, v1) pair to edge index

	for (int t = 0; t < tri_count; t++) {
		IntrinsicTriangle &tri = triangles.write[t];
		tri.v[0] = indices[t * 3];
		tri.v[1] = indices[t * 3 + 1];
		tri.v[2] = indices[t * 3 + 2];
		tri.is_active = true;

		// Map vertices to this triangle
		if (!vertex_to_triangle.has(tri.v[0])) {
			vertex_to_triangle[tri.v[0]] = t;
		}
		if (!vertex_to_triangle.has(tri.v[1])) {
			vertex_to_triangle[tri.v[1]] = t;
		}
		if (!vertex_to_triangle.has(tri.v[2])) {
			vertex_to_triangle[tri.v[2]] = t;
		}

		// Process each edge
		for (int e = 0; e < 3; e++) {
			int v0 = tri.v[e];
			int v1 = tri.v[(e + 1) % 3];

			// Ensure consistent edge ordering
			if (v0 > v1) {
				SWAP(v0, v1);
			}

			uint64_t edge_key = ((uint64_t)v0 << 32) | v1;

			int edge_idx;
			if (edge_map.has(edge_key)) {
				edge_idx = edge_map[edge_key];
				edges.write[edge_idx].t1 = t; // Second adjacent triangle
			} else {
				edge_idx = edges.size();
				IntrinsicEdge edge;
				edge.v0 = v0;
				edge.v1 = v1;
				edge.t0 = t;
				edge.t1 = -1; // No second triangle yet
				edge.length = vertices[v0].distance_to(vertices[v1]);
				edge.is_original = true;
				edges.push_back(edge);
				edge_map[edge_key] = edge_idx;
			}

			tri.e[e] = edge_idx;
			tri.edge_lengths[e] = edges[edge_idx].length;
		}
	}

	// Initialize normals if not provided
	if (normals.size() != vertices.size()) {
		update_normals();
	}
}

int IntrinsicTriangulation::find_edge(int v0, int v1) const {
	if (v0 > v1) {
		SWAP(v0, v1);
	}

	for (int i = 0; i < edges.size(); i++) {
		if (edges[i].v0 == v0 && edges[i].v1 == v1) {
			return i;
		}
	}
	return -1;
}

bool IntrinsicTriangulation::is_delaunay_edge(int edge_idx) const {
	const IntrinsicEdge &edge = edges[edge_idx];

	// Boundary edges are always Delaunay
	if (edge.t0 == -1 || edge.t1 == -1) {
		return true;
	}

	const IntrinsicTriangle &tri0 = triangles[edge.t0];
	const IntrinsicTriangle &tri1 = triangles[edge.t1];

	// Find opposite vertices
	int opp0 = -1, opp1 = -1;
	for (int i = 0; i < 3; i++) {
		if (tri0.v[i] != edge.v0 && tri0.v[i] != edge.v1) {
			opp0 = tri0.v[i];
		}
		if (tri1.v[i] != edge.v0 && tri1.v[i] != edge.v1) {
			opp1 = tri1.v[i];
		}
	}

	// Check Delaunay criterion: sum of opposite angles < 180 degrees
	// Using intrinsic edge lengths
	double a0 = compute_angle_at_vertex(tri0, opp0);
	double a1 = compute_angle_at_vertex(tri1, opp1);

	return (a0 + a1) <= 3.14159265359; // PI in radians
}

double IntrinsicTriangulation::compute_angle_at_vertex(const IntrinsicTriangle &tri, int vertex_idx) const {
	// Find which local index this vertex is
	int local_idx = -1;
	for (int i = 0; i < 3; i++) {
		if (tri.v[i] == vertex_idx) {
			local_idx = i;
			break;
		}
	}

	if (local_idx == -1) {
		return 0.0;
	}

	// Get edge lengths opposite and adjacent to this vertex
	double a = tri.edge_lengths[(local_idx + 1) % 3]; // Opposite edge
	double b = tri.edge_lengths[(local_idx + 2) % 3]; // Adjacent edge 1
	double c = tri.edge_lengths[local_idx]; // Adjacent edge 2

	// Law of cosines: cos(angle) = (b^2 + c^2 - a^2) / (2bc)
	double cos_angle = (b * b + c * c - a * a) / (2.0 * b * c);
	cos_angle = CLAMP(cos_angle, -1.0, 1.0);
	return Math::acos(cos_angle);
}

bool IntrinsicTriangulation::can_flip_edge(int edge_idx) const {
	const IntrinsicEdge &edge = edges[edge_idx];

	// Cannot flip boundary edges
	if (edge.t0 == -1 || edge.t1 == -1) {
		return false;
	}

	// Cannot flip if triangles are not active
	if (!triangles[edge.t0].is_active || !triangles[edge.t1].is_active) {
		return false;
	}

	return true;
}

void IntrinsicTriangulation::flip_edge(int edge_idx) {
	if (!can_flip_edge(edge_idx)) {
		return;
	}

	IntrinsicEdge &edge = edges.write[edge_idx];
	IntrinsicTriangle &tri0 = triangles.write[edge.t0];
	IntrinsicTriangle &tri1 = triangles.write[edge.t1];

	// Find opposite vertices
	int opp0 = -1, opp1 = -1;
	for (int i = 0; i < 3; i++) {
		if (tri0.v[i] != edge.v0 && tri0.v[i] != edge.v1) {
			opp0 = tri0.v[i];
		}
		if (tri1.v[i] != edge.v0 && tri1.v[i] != edge.v1) {
			opp1 = tri1.v[i];
		}
	}

	// Update edge to connect opposite vertices
	edge.v0 = MIN(opp0, opp1);
	edge.v1 = MAX(opp0, opp1);
	edge.length = vertices[opp0].distance_to(vertices[opp1]);
	edge.is_original = false;

	// Rebuild triangles with new connectivity
	// Triangle 0: opp0, old_v0, opp1
	// Triangle 1: opp0, opp1, old_v1
	// This is simplified - a full implementation would update all edge references
}

bool IntrinsicTriangulation::flip_to_delaunay() {
	bool made_flip = true;
	int iteration = 0;

	while (made_flip && iteration < max_flip_iterations) {
		made_flip = false;

		for (int e = 0; e < edges.size(); e++) {
			if (can_flip_edge(e) && !is_delaunay_edge(e)) {
				flip_edge(e);
				made_flip = true;
			}
		}

		iteration++;
	}

	return iteration < max_flip_iterations;
}

bool IntrinsicTriangulation::refine_intrinsic_triangulation(float p_target_edge_length) {
	// Perform Delaunay flipping first
	if (!flip_to_delaunay()) {
		return false;
	}

	// If no target length specified, compute from average edge length
	if (p_target_edge_length <= 0) {
		double total_length = 0.0;
		for (int i = 0; i < edges.size(); i++) {
			total_length += edges[i].length;
		}
		p_target_edge_length = total_length / edges.size();
	}

	// Refinement would involve edge splitting/collapsing
	// This is a simplified placeholder
	return true;
}

void IntrinsicTriangulation::smooth_intrinsic_positions(int p_iterations) {
	// Laplacian smoothing on vertex positions
	for (int iter = 0; iter < p_iterations; iter++) {
		PackedVector3Array new_vertices = vertices;

		for (int v = 0; v < vertices.size(); v++) {
			Vector3 avg_pos;
			int neighbor_count = 0;

			// Find all edges connected to this vertex
			for (int e = 0; e < edges.size(); e++) {
				if (edges[e].v0 == v) {
					avg_pos += vertices[edges[e].v1];
					neighbor_count++;
				} else if (edges[e].v1 == v) {
					avg_pos += vertices[edges[e].v0];
					neighbor_count++;
				}
			}

			if (neighbor_count > 0) {
				new_vertices.write[v] = avg_pos / neighbor_count;
			}
		}

		vertices = new_vertices;
	}

	update_normals();
}

void IntrinsicTriangulation::update_normals() {
	normals.resize(vertices.size());

	// Initialize all normals to zero
	for (int i = 0; i < normals.size(); i++) {
		normals.write[i] = Vector3();
	}

	// Accumulate triangle normals
	for (int t = 0; t < triangles.size(); t++) {
		if (!triangles[t].is_active) {
			continue;
		}

		Vector3 normal = compute_triangle_normal(t);

		for (int i = 0; i < 3; i++) {
			normals.write[triangles[t].v[i]] += normal;
		}
	}

	// Normalize
	for (int i = 0; i < normals.size(); i++) {
		normals.write[i] = normals[i].normalized();
	}
}

Vector3 IntrinsicTriangulation::compute_triangle_normal(int tri_idx) const {
	const IntrinsicTriangle &tri = triangles[tri_idx];
	Vector3 v0 = vertices[tri.v[0]];
	Vector3 v1 = vertices[tri.v[1]];
	Vector3 v2 = vertices[tri.v[2]];

	Vector3 edge1 = v1 - v0;
	Vector3 edge2 = v2 - v0;

	return edge1.cross(edge2).normalized();
}

void IntrinsicTriangulation::set_max_flip_iterations(int p_iterations) {
	max_flip_iterations = MAX(1, p_iterations);
}

int IntrinsicTriangulation::get_max_flip_iterations() const {
	return max_flip_iterations;
}

void IntrinsicTriangulation::set_angle_threshold(float p_angle) {
	angle_threshold = p_angle;
}

float IntrinsicTriangulation::get_angle_threshold() const {
	return angle_threshold;
}

void IntrinsicTriangulation::use_delaunay_flips(bool p_enable) {
	use_delaunay_criterion = p_enable;
}

PackedVector3Array IntrinsicTriangulation::get_vertices() const {
	return vertices;
}

PackedInt32Array IntrinsicTriangulation::get_indices() const {
	PackedInt32Array result;

	for (int t = 0; t < triangles.size(); t++) {
		if (triangles[t].is_active) {
			result.push_back(triangles[t].v[0]);
			result.push_back(triangles[t].v[1]);
			result.push_back(triangles[t].v[2]);
		}
	}

	return result;
}

PackedVector3Array IntrinsicTriangulation::get_normals() const {
	return normals;
}

Ref<ArrayMesh> IntrinsicTriangulation::get_mesh() const {
	Ref<ArrayMesh> mesh;
	mesh.instantiate();

	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = get_vertices();
	arrays[Mesh::ARRAY_INDEX] = get_indices();
	arrays[Mesh::ARRAY_NORMAL] = get_normals();

	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
	return mesh;
}

int IntrinsicTriangulation::get_triangle_count() const {
	int count = 0;
	for (int i = 0; i < triangles.size(); i++) {
		if (triangles[i].is_active) {
			count++;
		}
	}
	return count;
}

int IntrinsicTriangulation::get_edge_count() const {
	return edges.size();
}

int IntrinsicTriangulation::get_vertex_count() const {
	return vertices.size();
}

Dictionary IntrinsicTriangulation::get_statistics() const {
	Dictionary stats;
	stats["vertex_count"] = get_vertex_count();
	stats["edge_count"] = get_edge_count();
	stats["triangle_count"] = get_triangle_count();

	double total_edge_length = 0.0;
	double min_edge = DBL_MAX;
	double max_edge = 0.0;

	for (int i = 0; i < edges.size(); i++) {
		total_edge_length += edges[i].length;
		min_edge = MIN(min_edge, edges[i].length);
		max_edge = MAX(max_edge, edges[i].length);
	}

	stats["average_edge_length"] = edges.size() > 0 ? total_edge_length / edges.size() : 0.0;
	stats["min_edge_length"] = min_edge;
	stats["max_edge_length"] = max_edge;

	return stats;
}

IntrinsicTriangulation::IntrinsicTriangulation() {
}

IntrinsicTriangulation::~IntrinsicTriangulation() {
}
