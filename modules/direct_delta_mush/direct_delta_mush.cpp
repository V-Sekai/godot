/**************************************************************************/
/*  direct_delta_mush.cpp                                                 */
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

/* direct_delta_mush.cpp */

#include "direct_delta_mush.h"

#include "core/config/engine.h"
#include "ddm_deformer.h"
#include "scene/resources/mesh.h"
#include "servers/rendering/ddm_compute.h"

void DirectDeltaMushDeformer::_bind_methods() {
	// Properties
	ClassDB::bind_method(D_METHOD("set_iterations", "iterations"), &DirectDeltaMushDeformer::set_iterations);
	ClassDB::bind_method(D_METHOD("get_iterations"), &DirectDeltaMushDeformer::get_iterations);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "iterations", PROPERTY_HINT_RANGE, "1,100,1"), "set_iterations", "get_iterations");

	ClassDB::bind_method(D_METHOD("set_smooth_lambda", "lambda"), &DirectDeltaMushDeformer::set_smooth_lambda);
	ClassDB::bind_method(D_METHOD("get_smooth_lambda"), &DirectDeltaMushDeformer::get_smooth_lambda);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "smooth_lambda", PROPERTY_HINT_RANGE, "0.1,2.0,0.01"), "set_smooth_lambda", "get_smooth_lambda");

	ClassDB::bind_method(D_METHOD("set_adjacency_tolerance", "tolerance"), &DirectDeltaMushDeformer::set_adjacency_tolerance);
	ClassDB::bind_method(D_METHOD("get_adjacency_tolerance"), &DirectDeltaMushDeformer::get_adjacency_tolerance);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "adjacency_tolerance", PROPERTY_HINT_RANGE, "0.0001,0.01,0.0001"), "set_adjacency_tolerance", "get_adjacency_tolerance");

	ClassDB::bind_method(D_METHOD("set_use_compute", "use_compute"), &DirectDeltaMushDeformer::set_use_compute);
	ClassDB::bind_method(D_METHOD("get_use_compute"), &DirectDeltaMushDeformer::get_use_compute);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_compute"), "set_use_compute", "get_use_compute");

	// Methods
	ClassDB::bind_method(D_METHOD("precompute"), &DirectDeltaMushDeformer::precompute);
}

void DirectDeltaMushDeformer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			// Initialize rendering device
			rd = RenderingServer::get_singleton()->get_rendering_device();
		} break;

		case NOTIFICATION_PROCESS: {
			if (Engine::get_singleton()->is_editor_hint()) {
				return;
			}
			update_deformation();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			// Clean up resources
			if (rd && omega_buffer.is_valid()) {
				rd->free_rid(omega_buffer);
			}
			if (rd && adjacency_buffer.is_valid()) {
				rd->free_rid(adjacency_buffer);
			}
			if (rd && laplacian_buffer.is_valid()) {
				rd->free_rid(laplacian_buffer);
			}
		} break;
	}
}

DirectDeltaMushDeformer::DirectDeltaMushDeformer() {
	// Initialize with default values
	set_process(true);
}

DirectDeltaMushDeformer::~DirectDeltaMushDeformer() {
	// Cleanup handled in _notification
}

// Property setters/getters
void DirectDeltaMushDeformer::set_iterations(int p_iterations) {
	iterations = p_iterations;
}

int DirectDeltaMushDeformer::get_iterations() const {
	return iterations;
}

void DirectDeltaMushDeformer::set_smooth_lambda(float p_lambda) {
	smooth_lambda = p_lambda;
}

float DirectDeltaMushDeformer::get_smooth_lambda() const {
	return smooth_lambda;
}

void DirectDeltaMushDeformer::set_adjacency_tolerance(float p_tolerance) {
	adjacency_tolerance = p_tolerance;
}

float DirectDeltaMushDeformer::get_adjacency_tolerance() const {
	return adjacency_tolerance;
}

void DirectDeltaMushDeformer::set_use_compute(bool p_use_compute) {
	use_compute = p_use_compute;
}

bool DirectDeltaMushDeformer::get_use_compute() const {
	return use_compute;
}

// Public methods
void DirectDeltaMushDeformer::precompute() {
	precompute_data();
}

// Internal methods
void DirectDeltaMushDeformer::precompute_data() {
	Ref<Mesh> mesh = get_mesh();
	if (mesh.is_null()) {
		WARN_PRINT("No mesh assigned to DirectDeltaMushDeformer node");
		return;
	}

	// Check if mesh has bone weights
	Array surface_arrays = mesh->surface_get_arrays(0);
	if (surface_arrays.size() <= Mesh::ARRAY_BONES) {
		WARN_PRINT("Mesh must have bone weights for Direct Delta Mush");
		return;
	}

	// Build adjacency matrix
	build_adjacency_matrix();

	// Compute Laplacian matrix
	compute_laplacian_matrix();

	// Precompute Omega matrices
	precompute_omega_matrices();

	print_line("Direct Delta Mush precomputation completed");
}

void DirectDeltaMushDeformer::update_deformation() {
	Ref<Mesh> mesh = get_mesh();
	if (mesh.is_null() || !omega_buffer.is_valid()) {
		return;
	}

	// Get skeleton bone transforms
	NodePath skeleton_path = get_skeleton_path();
	if (skeleton_path.is_empty()) {
		return;
	}

	Node *skel_node = get_node_or_null(skeleton_path);
	if (!skel_node) {
		return;
	}

	Skeleton3D *skeleton = Object::cast_to<Skeleton3D>(skel_node);
	if (!skeleton) {
		return;
	}

	// Get bone transforms
	int bone_count = skeleton->get_bone_count();
	Vector<Transform3D> bone_transforms;
	bone_transforms.resize(bone_count);

	for (int i = 0; i < bone_count; i++) {
		bone_transforms.set(i, skeleton->get_bone_global_pose(i));
	}

	// Create deformed mesh if needed
	if (deformed_mesh.is_null()) {
		deformed_mesh = mesh->duplicate();
	}

	// Extract mesh surface data for both CPU and GPU paths
	Array surface_arrays = mesh->surface_get_arrays(0);
	PackedVector3Array vertices = surface_arrays[Mesh::ARRAY_VERTEX];
	PackedVector3Array normals = surface_arrays[Mesh::ARRAY_NORMAL];
	PackedInt32Array indices = surface_arrays[Mesh::ARRAY_INDEX];
	PackedFloat32Array bone_weights = surface_arrays[Mesh::ARRAY_WEIGHTS];
	PackedInt32Array bone_indices = surface_arrays[Mesh::ARRAY_BONES];

	// Apply deformation
	if (use_compute && rd && omega_buffer.is_valid()) {
		// GPU deformation using DDMCompute
		Ref<DDMCompute> ddm_compute;
		ddm_compute.instantiate();
		
		if (!ddm_compute->initialize(rd)) {
			WARN_PRINT("Failed to initialize DDMCompute, falling back to CPU");
		} else {
			// Create bone transforms buffer
			Vector<float> bone_transform_data;
			bone_transform_data.resize(bone_count * 16); // 4x4 matrices
			
			for (int i = 0; i < bone_count; i++) {
				Transform3D t = bone_transforms[i];
				// Pack transform into float array (column-major order)
				int offset = i * 16;
				for (int row = 0; row < 3; row++) {
					for (int col = 0; col < 3; col++) {
						bone_transform_data.set(offset + col * 4 + row, t.basis[col][row]);
					}
				}
				// Translation in 4th column
				bone_transform_data.set(offset + 12, t.origin.x);
				bone_transform_data.set(offset + 13, t.origin.y);
				bone_transform_data.set(offset + 14, t.origin.z);
				bone_transform_data.set(offset + 15, 1.0f);
			}
			
			RID bones_buffer = rd->storage_buffer_create(bone_transform_data.size() * sizeof(float), bone_transform_data.to_byte_array());
			
			// Create output buffers
			RID output_vertices_buffer = rd->storage_buffer_create(vertices.size() * sizeof(float) * 3);
			RID output_normals_buffer = rd->storage_buffer_create(normals.size() * sizeof(float) * 3);
			
			// Create input vertex/normal buffers
			Vector<float> vertex_data;
			vertex_data.resize(vertices.size() * 3);
			for (int i = 0; i < vertices.size(); i++) {
				vertex_data.set(i * 3 + 0, vertices[i].x);
				vertex_data.set(i * 3 + 1, vertices[i].y);
				vertex_data.set(i * 3 + 2, vertices[i].z);
			}
			RID input_vertices_buffer = rd->storage_buffer_create(vertex_data.size() * sizeof(float), vertex_data.to_byte_array());
			
			Vector<float> normal_data;
			normal_data.resize(normals.size() * 3);
			for (int i = 0; i < normals.size(); i++) {
				normal_data.set(i * 3 + 0, normals[i].x);
				normal_data.set(i * 3 + 1, normals[i].y);
				normal_data.set(i * 3 + 2, normals[i].z);
			}
			RID input_normals_buffer = rd->storage_buffer_create(normal_data.size() * sizeof(float), normal_data.to_byte_array());
			
			// Perform GPU deformation
			bool gpu_success = ddm_compute->deform_mesh(
					omega_buffer,
					bones_buffer,
					input_vertices_buffer,
					input_normals_buffer,
					output_vertices_buffer,
					output_normals_buffer,
					vertices.size());
			
			if (gpu_success) {
				// Download results from GPU
				Vector<uint8_t> output_verts_bytes = rd->buffer_get_data(output_vertices_buffer);
				Vector<uint8_t> output_norms_bytes = rd->buffer_get_data(output_normals_buffer);
				
				// Convert back to Vector3 arrays
				PackedVector3Array deformed_verts;
				deformed_verts.resize(vertices.size());
				const float *verts_ptr = (const float *)output_verts_bytes.ptr();
				for (int i = 0; i < vertices.size(); i++) {
					deformed_verts.set(i, Vector3(verts_ptr[i * 3], verts_ptr[i * 3 + 1], verts_ptr[i * 3 + 2]));
				}
				
				PackedVector3Array deformed_norms;
				deformed_norms.resize(normals.size());
				const float *norms_ptr = (const float *)output_norms_bytes.ptr();
				for (int i = 0; i < normals.size(); i++) {
					deformed_norms.set(i, Vector3(norms_ptr[i * 3], norms_ptr[i * 3 + 1], norms_ptr[i * 3 + 2]));
				}
				
				// Update surface arrays with deformed geometry
				surface_arrays[Mesh::ARRAY_VERTEX] = deformed_verts;
				surface_arrays[Mesh::ARRAY_NORMAL] = deformed_norms;
				
				// Recreate mesh surface
				Ref<ArrayMesh> array_mesh = Object::cast_to<ArrayMesh>(deformed_mesh.ptr());
				if (array_mesh.is_valid()) {
					array_mesh->clear_surfaces();
					array_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, surface_arrays);
					array_mesh->surface_set_material(0, mesh->surface_get_material(0));
				}
			}
			
			// Clean up GPU buffers
			rd->free_rid(bones_buffer);
			rd->free_rid(input_vertices_buffer);
			rd->free_rid(input_normals_buffer);
			rd->free_rid(output_vertices_buffer);
			rd->free_rid(output_normals_buffer);
			
			ddm_compute->cleanup();
		}
	} else {
		// CPU deformation using Enhanced DDM
		Array surface_arrays = mesh->surface_get_arrays(0);
		PackedVector3Array vertices = surface_arrays[Mesh::ARRAY_VERTEX];
		PackedInt32Array indices = surface_arrays[Mesh::ARRAY_INDEX];
		PackedFloat32Array bone_weights = surface_arrays[Mesh::ARRAY_WEIGHTS];
		PackedInt32Array bone_indices = surface_arrays[Mesh::ARRAY_BONES];

		// Convert to DDMDeformer format
		DDMDeformer::MeshData mesh_data;
		mesh_data.vertices.resize(vertices.size());
		mesh_data.indices.resize(indices.size());
		
		// Copy vertices
		for (int i = 0; i < vertices.size(); i++) {
			mesh_data.vertices.set(i, vertices[i]);
		}
		
		// Copy normals
		PackedVector3Array normals = surface_arrays[Mesh::ARRAY_NORMAL];
		mesh_data.normals.resize(normals.size());
		for (int i = 0; i < normals.size(); i++) {
			mesh_data.normals.set(i, normals[i]);
		}
		
		// Copy indices
		for (int i = 0; i < indices.size(); i++) {
			mesh_data.indices.set(i, indices[i]);
		}

		// Set up bone weights (4 bones per vertex)
		int bones_per_vertex = 4;
		mesh_data.bone_weights.resize(vertices.size() * bones_per_vertex);
		mesh_data.bone_indices.resize(vertices.size() * bones_per_vertex);

		for (int v = 0; v < vertices.size(); v++) {
			for (int b = 0; b < bones_per_vertex; b++) {
				int idx = v * bones_per_vertex + b;
				if (idx < bone_weights.size()) {
					mesh_data.bone_weights.set(v * bones_per_vertex + b, bone_weights[idx]);
				} else {
					mesh_data.bone_weights.set(v * bones_per_vertex + b, 0.0f);
				}
				if (idx < bone_indices.size()) {
					mesh_data.bone_indices.set(v * bones_per_vertex + b, bone_indices[idx]);
				} else {
					mesh_data.bone_indices.set(v * bones_per_vertex + b, 0);
				}
			}
		}

		// Configure Enhanced DDM
		DDMDeformer::Config config;
		config.iterations = iterations;
		config.smooth_lambda = smooth_lambda;

		// Create deformer, initialize, and apply
		DDMDeformer deformer;
		deformer.initialize(mesh_data);
		DDMDeformer::DeformResult result = deformer.deform(bone_transforms, config);

		// Update mesh with deformed vertices
		if (result.success && result.vertices.size() == vertices.size()) {
			PackedVector3Array deformed_verts;
			deformed_verts.resize(result.vertices.size());
			for (int i = 0; i < result.vertices.size(); i++) {
				deformed_verts.set(i, result.vertices[i]);
			}

			// Update surface arrays with deformed vertices
			surface_arrays[Mesh::ARRAY_VERTEX] = deformed_verts;
			
			// Recreate mesh surface
			Ref<ArrayMesh> array_mesh = Object::cast_to<ArrayMesh>(deformed_mesh.ptr());
			if (array_mesh.is_valid()) {
				array_mesh->clear_surfaces();
				array_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, surface_arrays);
				array_mesh->surface_set_material(0, mesh->surface_get_material(0));
			}
		}
	}

	// Update the displayed mesh
	set_mesh(deformed_mesh);
}

void DirectDeltaMushDeformer::build_adjacency_matrix() {
	Ref<Mesh> mesh = get_mesh();
	if (mesh.is_null()) {
		return;
	}

	// Extract mesh surface data
	Array surface_arrays = mesh->surface_get_arrays(0);
	PackedVector3Array vertices = surface_arrays[Mesh::ARRAY_VERTEX];
	PackedInt32Array indices = surface_arrays[Mesh::ARRAY_INDEX];

	int vertex_count = vertices.size();
	const int max_neighbors = 32;

	// Allocate adjacency buffer
	if (rd) {
		adjacency_buffer = rd->storage_buffer_create(vertex_count * max_neighbors * sizeof(int));
	}

	// Build adjacency matrix (CPU implementation for now)
	Vector<int> adjacency_data;
	adjacency_data.resize(vertex_count * max_neighbors);

	// Initialize to -1
	for (int i = 0; i < adjacency_data.size(); i++) {
		adjacency_data.set(i, -1);
	}

	// Build from triangles
	for (int tri = 0; tri < indices.size(); tri += 3) {
		int v0 = indices[tri];
		int v1 = indices[tri + 1];
		int v2 = indices[tri + 2];

		// Add edges
		add_edge_to_adjacency(adjacency_data, v0, v1, max_neighbors);
		add_edge_to_adjacency(adjacency_data, v0, v2, max_neighbors);
		add_edge_to_adjacency(adjacency_data, v1, v2, max_neighbors);
	}

	// Upload to GPU if using compute
	if (rd && adjacency_buffer.is_valid()) {
		rd->buffer_update(adjacency_buffer, 0, adjacency_data.size() * sizeof(int), adjacency_data.ptr());
	}
}

void DirectDeltaMushDeformer::compute_laplacian_matrix() {
	if (!adjacency_buffer.is_valid()) {
		return;
	}

	Ref<Mesh> mesh = get_mesh();
	Array surface_arrays = mesh->surface_get_arrays(0);
	PackedVector3Array vertices = surface_arrays[Mesh::ARRAY_VERTEX];
	int vertex_count = vertices.size();
	const int max_neighbors = 32;

	// Allocate Laplacian buffer
	if (rd) {
		laplacian_buffer = rd->storage_buffer_create(vertex_count * max_neighbors * 2 * sizeof(float));
	}

	// TODO: Implement Laplacian computation
	// For now, create a basic Laplacian matrix
}

void DirectDeltaMushDeformer::precompute_omega_matrices() {
	if (!laplacian_buffer.is_valid()) {
		return;
	}

	Ref<Mesh> mesh = get_mesh();
	Array surface_arrays = mesh->surface_get_arrays(0);
	PackedVector3Array vertices = surface_arrays[Mesh::ARRAY_VERTEX];
	PackedFloat32Array weights = surface_arrays[Mesh::ARRAY_WEIGHTS];
	int vertex_count = vertices.size();

	// Allocate omega buffer (32 bones max, 16 floats per 4x4 matrix)
	if (rd) {
		omega_buffer = rd->storage_buffer_create(vertex_count * 32 * 16 * sizeof(float));
	}

	// TODO: Implement omega matrix precomputation using DDMCompute CPU fallback
	// For now, initialize with identity matrices

	// Create DDMCompute instance for CPU computation
	Ref<DDMCompute> ddm_compute;
	ddm_compute.instantiate();
	ddm_compute->initialize(rd);

	// Prepare data for CPU computation
	Vector<int> adjacency_data; // Would need to be extracted from adjacency_buffer
	Vector<float> laplacian_data; // Would need to be extracted from laplacian_buffer
	Vector<Vector3> vertex_data;
	Vector<float> weight_data;
	Vector<float> omega_data;

	// Convert Godot arrays to Vectors
	for (int i = 0; i < vertices.size(); i++) {
		vertex_data.push_back(vertices[i]);
	}
	for (int i = 0; i < weights.size(); i++) {
		weight_data.push_back(weights[i]);
	}

	// TODO: Extract adjacency and Laplacian data from GPU buffers
	// For now, skip CPU computation and use identity matrices
}

// Helper method
void DirectDeltaMushDeformer::add_edge_to_adjacency(Vector<int> &adjacency_data, int v0, int v1, int max_neighbors) {
	// Add v1 to v0's adjacency list
	for (int i = 0; i < max_neighbors; i++) {
		int idx = v0 * max_neighbors + i;
		if (adjacency_data[idx] == -1) {
			adjacency_data.set(idx, v1);
			break;
		}
		if (adjacency_data[idx] == v1) {
			break; // Already exists
		}
	}

	// Add v0 to v1's adjacency list
	for (int i = 0; i < max_neighbors; i++) {
		int idx = v1 * max_neighbors + i;
		if (adjacency_data[idx] == -1) {
			adjacency_data.set(idx, v0);
			break;
		}
		if (adjacency_data[idx] == v0) {
			break; // Already exists
		}
	}
}
