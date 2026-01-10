/**************************************************************************/
/*  cassie_surface.cpp                                                    */
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

#include "cassie_surface.h"

CassieSurface::CassieSurface() {
}

CassieSurface::~CassieSurface() {
}

void CassieSurface::_bind_methods() {
	// Path management
	ClassDB::bind_method(D_METHOD("add_boundary_path", "path"), &CassieSurface::add_boundary_path);
	ClassDB::bind_method(D_METHOD("clear_boundary_paths"), &CassieSurface::clear_boundary_paths);
	ClassDB::bind_method(D_METHOD("get_boundary_path_count"), &CassieSurface::get_boundary_path_count);
	ClassDB::bind_method(D_METHOD("get_boundary_path", "index"), &CassieSurface::get_boundary_path);

	// Configuration
	ClassDB::bind_method(D_METHOD("set_auto_beautify", "enable"), &CassieSurface::set_auto_beautify);
	ClassDB::bind_method(D_METHOD("get_auto_beautify"), &CassieSurface::get_auto_beautify);

	ClassDB::bind_method(D_METHOD("set_auto_resample", "enable"), &CassieSurface::set_auto_resample);
	ClassDB::bind_method(D_METHOD("get_auto_resample"), &CassieSurface::get_auto_resample);

	ClassDB::bind_method(D_METHOD("set_use_intrinsic_remeshing", "enable"), &CassieSurface::set_use_intrinsic_remeshing);
	ClassDB::bind_method(D_METHOD("get_use_intrinsic_remeshing"), &CassieSurface::get_use_intrinsic_remeshing);

	ClassDB::bind_method(D_METHOD("set_target_boundary_points", "count"), &CassieSurface::set_target_boundary_points);
	ClassDB::bind_method(D_METHOD("get_target_boundary_points"), &CassieSurface::get_target_boundary_points);

	ClassDB::bind_method(D_METHOD("set_beautify_lambda", "lambda"), &CassieSurface::set_beautify_lambda);
	ClassDB::bind_method(D_METHOD("get_beautify_lambda"), &CassieSurface::get_beautify_lambda);

	ClassDB::bind_method(D_METHOD("set_beautify_mu", "mu"), &CassieSurface::set_beautify_mu);
	ClassDB::bind_method(D_METHOD("get_beautify_mu"), &CassieSurface::get_beautify_mu);

	ClassDB::bind_method(D_METHOD("set_beautify_iterations", "iterations"), &CassieSurface::set_beautify_iterations);
	ClassDB::bind_method(D_METHOD("get_beautify_iterations"), &CassieSurface::get_beautify_iterations);

	ClassDB::bind_method(D_METHOD("set_max_flip_iterations", "iterations"), &CassieSurface::set_max_flip_iterations);
	ClassDB::bind_method(D_METHOD("get_max_flip_iterations"), &CassieSurface::get_max_flip_iterations);

	ClassDB::bind_method(D_METHOD("set_smooth_iterations", "iterations"), &CassieSurface::set_smooth_iterations);
	ClassDB::bind_method(D_METHOD("get_smooth_iterations"), &CassieSurface::get_smooth_iterations);

	ClassDB::bind_method(D_METHOD("set_target_edge_length", "length"), &CassieSurface::set_target_edge_length);
	ClassDB::bind_method(D_METHOD("get_target_edge_length"), &CassieSurface::get_target_edge_length);

	// Surface generation
	ClassDB::bind_method(D_METHOD("generate_surface"), &CassieSurface::generate_surface);
	ClassDB::bind_method(D_METHOD("get_generated_mesh"), &CassieSurface::get_generated_mesh);
	ClassDB::bind_method(D_METHOD("clear_generated_mesh"), &CassieSurface::clear_generated_mesh);

	// Properties
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_beautify"), "set_auto_beautify", "get_auto_beautify");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_resample"), "set_auto_resample", "get_auto_resample");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_intrinsic_remeshing"), "set_use_intrinsic_remeshing", "get_use_intrinsic_remeshing");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "target_boundary_points"), "set_target_boundary_points", "get_target_boundary_points");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "beautify_lambda"), "set_beautify_lambda", "get_beautify_lambda");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "beautify_mu"), "set_beautify_mu", "get_beautify_mu");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "beautify_iterations"), "set_beautify_iterations", "get_beautify_iterations");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_flip_iterations"), "set_max_flip_iterations", "get_max_flip_iterations");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "smooth_iterations"), "set_smooth_iterations", "get_smooth_iterations");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "target_edge_length"), "set_target_edge_length", "get_target_edge_length");
}

// Path management
void CassieSurface::add_boundary_path(const Ref<CassiePath3D> &p_path) {
	ERR_FAIL_COND_MSG(p_path.is_null(), "Cannot add null boundary path.");
	boundary_paths.push_back(p_path);
}

void CassieSurface::clear_boundary_paths() {
	boundary_paths.clear();
}

int CassieSurface::get_boundary_path_count() const {
	return boundary_paths.size();
}

Ref<CassiePath3D> CassieSurface::get_boundary_path(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, boundary_paths.size(), Ref<CassiePath3D>());
	return boundary_paths[p_index];
}

// Configuration setters/getters
void CassieSurface::set_auto_beautify(bool p_enable) {
	auto_beautify = p_enable;
}

bool CassieSurface::get_auto_beautify() const {
	return auto_beautify;
}

void CassieSurface::set_auto_resample(bool p_enable) {
	auto_resample = p_enable;
}

bool CassieSurface::get_auto_resample() const {
	return auto_resample;
}

void CassieSurface::set_use_intrinsic_remeshing(bool p_enable) {
	use_intrinsic_remeshing = p_enable;
}

bool CassieSurface::get_use_intrinsic_remeshing() const {
	return use_intrinsic_remeshing;
}

void CassieSurface::set_target_boundary_points(int p_count) {
	target_boundary_points = MAX(3, p_count);
}

int CassieSurface::get_target_boundary_points() const {
	return target_boundary_points;
}

void CassieSurface::set_beautify_lambda(float p_lambda) {
	beautify_lambda = CLAMP(p_lambda, 0.0f, 1.0f);
}

float CassieSurface::get_beautify_lambda() const {
	return beautify_lambda;
}

void CassieSurface::set_beautify_mu(float p_mu) {
	beautify_mu = CLAMP(p_mu, -1.0f, 0.0f);
}

float CassieSurface::get_beautify_mu() const {
	return beautify_mu;
}

void CassieSurface::set_beautify_iterations(int p_iterations) {
	beautify_iterations = MAX(0, p_iterations);
}

int CassieSurface::get_beautify_iterations() const {
	return beautify_iterations;
}

void CassieSurface::set_max_flip_iterations(int p_iterations) {
	max_flip_iterations = MAX(0, p_iterations);
}

int CassieSurface::get_max_flip_iterations() const {
	return max_flip_iterations;
}

void CassieSurface::set_smooth_iterations(int p_iterations) {
	smooth_iterations = MAX(0, p_iterations);
}

int CassieSurface::get_smooth_iterations() const {
	return smooth_iterations;
}

void CassieSurface::set_target_edge_length(float p_length) {
	target_edge_length = p_length;
}

float CassieSurface::get_target_edge_length() const {
	return target_edge_length;
}

// Surface generation
Ref<ArrayMesh> CassieSurface::generate_surface() {
	print_line("[Cassie] Starting generate_surface");
	ERR_FAIL_COND_V_MSG(boundary_paths.size() == 0, Ref<ArrayMesh>(), "No boundary paths added.");

	// Step 1: Process boundary paths
	Vector<PackedVector3Array> processed_boundaries;
	for (int i = 0; i < boundary_paths.size(); i++) {
		Ref<CassiePath3D> path = boundary_paths[i];
		ERR_CONTINUE_MSG(path.is_null(), "Boundary path at index " + itos(i) + " is null.");

		// Beautify if enabled
		if (auto_beautify) {
			path->beautify_taubin(beautify_lambda, beautify_mu, beautify_iterations);
		}

		// Resample if enabled
		if (auto_resample) {
			path->resample_uniform(target_boundary_points);
		}

		processed_boundaries.push_back(path->get_points());
	}

	// Step 2: Triangulate
	ERR_FAIL_COND_V_MSG(processed_boundaries.size() == 0, Ref<ArrayMesh>(), "No valid boundaries after processing.");
	print_line("[Cassie] Creating triangulator");

	Ref<PolygonTriangulationGodot> triangulator;
	if (processed_boundaries.size() == 1) {
		// Single boundary - simple triangulation
		triangulator = PolygonTriangulationGodot::create(processed_boundaries[0]);
	} else {
		// Multiple boundaries - need proper hole handling
		// For now, just use the first boundary
		// TODO: Implement proper multi-boundary support
		triangulator = PolygonTriangulationGodot::create(processed_boundaries[0]);
	}

	ERR_FAIL_COND_V_MSG(triangulator.is_null(), Ref<ArrayMesh>(), "Failed to create triangulator.");
	print_line("[Cassie] Preprocessing");

	// Preprocess and triangulate
	bool preprocess_ok = triangulator->preprocess();
	ERR_FAIL_COND_V_MSG(!preprocess_ok, Ref<ArrayMesh>(), "Triangulation preprocessing failed.");
	print_line("[Cassie] Triangulating");

	bool triangulate_ok = triangulator->triangulate();
	ERR_FAIL_COND_V_MSG(!triangulate_ok, Ref<ArrayMesh>(), "Triangulation failed.");
	print_line("[Cassie] Triangulation done");

	Ref<ArrayMesh> base_mesh;
	if (RenderingServer::get_singleton()) {
		base_mesh = triangulator->get_mesh();
	} else {
		// Headless mode: create ArrayMesh directly from triangulator data
		base_mesh.instantiate();
		Array arrays;
		arrays.resize(Mesh::ARRAY_MAX);
		PackedVector3Array vertices = triangulator->get_vertices();
		PackedInt32Array indices = triangulator->get_indices();
		PackedVector3Array normals = triangulator->get_normals();

		// Validate indices
		int vertex_count = vertices.size();
		int index_count = indices.size();
		print_line(vformat("[Cassie] Surface: vertices %d, indices %d", vertex_count, index_count));
		for (int i = 0; i < index_count; i++) {
			int idx = indices[i];
			if (idx < 0 || idx >= vertex_count) {
				print_line(vformat("[Cassie] Invalid index at %d: %d", i, idx));
				ERR_FAIL_COND_V_MSG(idx < 0 || idx >= vertex_count, Ref<ArrayMesh>(), vformat("Invalid index %d at %d, vertex count %d", idx, i, vertex_count));
			}
		}

		arrays[Mesh::ARRAY_VERTEX] = vertices;
		arrays[Mesh::ARRAY_INDEX] = indices;
		if (normals.size() > 0) {
			arrays[Mesh::ARRAY_NORMAL] = normals;
		}
		// Set other arrays to empty to avoid null
		arrays[Mesh::ARRAY_TANGENT] = PackedFloat32Array();
		arrays[Mesh::ARRAY_COLOR] = PackedColorArray();
		arrays[Mesh::ARRAY_TEX_UV] = PackedVector2Array();
		arrays[Mesh::ARRAY_TEX_UV2] = PackedVector2Array();
		arrays[Mesh::ARRAY_CUSTOM0] = PackedByteArray();
		arrays[Mesh::ARRAY_CUSTOM1] = PackedByteArray();
		arrays[Mesh::ARRAY_CUSTOM2] = PackedByteArray();
		arrays[Mesh::ARRAY_CUSTOM3] = PackedByteArray();
		arrays[Mesh::ARRAY_BONES] = PackedInt32Array();
		arrays[Mesh::ARRAY_WEIGHTS] = PackedFloat32Array();

		base_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
	}
	ERR_FAIL_COND_V_MSG(base_mesh.is_null() || base_mesh->get_surface_count() == 0, Ref<ArrayMesh>(), "Failed to get triangulated mesh.");

	// Step 3: Intrinsic remeshing if enabled
	if (use_intrinsic_remeshing) {
		Ref<IntrinsicTriangulation> intrinsic = memnew(IntrinsicTriangulation);
		intrinsic->set_mesh(base_mesh);
		intrinsic->set_max_flip_iterations(max_flip_iterations);

		// Flip to Delaunay
		intrinsic->flip_to_delaunay();

		// Refine if target edge length is set
		if (target_edge_length > 0.0f) {
			intrinsic->refine_intrinsic_triangulation(target_edge_length);
		}

		// Smooth positions
		if (smooth_iterations > 0) {
			intrinsic->smooth_intrinsic_positions(smooth_iterations);
		}

		generated_mesh = intrinsic->get_mesh();
	} else {
		generated_mesh = base_mesh;
	}

	return generated_mesh;
}

Ref<ArrayMesh> CassieSurface::get_generated_mesh() const {
	return generated_mesh;
}

void CassieSurface::clear_generated_mesh() {
	generated_mesh.unref();
}
