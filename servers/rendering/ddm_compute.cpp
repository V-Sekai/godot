/**************************************************************************/
/*  ddm_compute.cpp                                                       */
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

/* ddm_compute.cpp */

#include "ddm_compute.h"

#include <vector>

#include "core/io/file_access.h"

void DDMCompute::_bind_methods() {
	ClassDB::bind_method(D_METHOD("initialize", "rendering_device"), &DDMCompute::initialize);
	ClassDB::bind_method(D_METHOD("cleanup"), &DDMCompute::cleanup);
}

bool DDMCompute::initialize(RenderingDevice *p_rd) {
	rd = p_rd;
	if (!rd) {
		return false;
	}

	// Load and compile compute shaders
	adjacency_shader = load_shader_from_file("servers/rendering/shaders/ddm/adjacency.compute.glsl");
	laplacian_shader = load_shader_from_file("servers/rendering/shaders/ddm/laplacian.compute.glsl");
	omega_shader = load_shader_from_file("servers/rendering/shaders/ddm/omega_precompute.compute.glsl");
	deform_shader = load_shader_from_file("servers/rendering/shaders/ddm/deform.compute.glsl");

	if (adjacency_shader.is_null() || laplacian_shader.is_null() ||
			omega_shader.is_null() || deform_shader.is_null()) {
		ERR_PRINT("Failed to load Direct Delta Mush compute shaders");
		return false;
	}

	// Create compute pipelines
	Vector<StringName> adjacency_uniforms = { "Vertices", "Indices", "Adjacency", "Params" };
	Vector<StringName> laplacian_uniforms = { "Adjacency", "Laplacian", "Params" };
	Vector<StringName> omega_uniforms = { "Vertices", "Weights", "Laplacian", "Omegas", "Params" };
	Vector<StringName> deform_uniforms = { "Vertices", "Normals", "Bones", "Omegas", "OutputVertices", "OutputNormals", "Params" };

	if (!create_pipeline(adjacency_pipeline, adjacency_shader, adjacency_uniforms) ||
			!create_pipeline(laplacian_pipeline, laplacian_shader, laplacian_uniforms) ||
			!create_pipeline(omega_pipeline, omega_shader, omega_uniforms) ||
			!create_pipeline(deform_pipeline, deform_shader, deform_uniforms)) {
		ERR_PRINT("Failed to create Direct Delta Mush compute pipelines");
		return false;
	}

	return true;
}

void DDMCompute::cleanup() {
	// TODO: Clean up GPU resources
	if (rd) {
		// rd->free_rid(adjacency_pipeline);
		// rd->free_rid(laplacian_pipeline);
		// rd->free_rid(omega_pipeline);
		// rd->free_rid(deform_pipeline);
		// rd->free_rid(adjacency_shader);
		// rd->free_rid(laplacian_shader);
		// rd->free_rid(omega_shader);
		// rd->free_rid(deform_shader);
	}
}

void DDMCompute::add_edge_to_adjacency_cpu(Vector<int> &adjacency_data, int v0, int v1, int max_neighbors) {
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

RID DDMCompute::load_shader_from_file(const String &shader_path) {
	if (!rd) {
		return RID();
	}

	// Load shader file
	Ref<FileAccess> file = FileAccess::open(shader_path, FileAccess::READ);
	if (file.is_null()) {
		ERR_PRINT("Failed to open shader file: " + shader_path);
		return RID();
	}

	String shader_code = file->get_as_text();

	// Create shader from source
	return load_shader(shader_code);
}

RID DDMCompute::load_shader(const String &shader_code) {
	if (!rd) {
		return RID();
	}

	// Compile GLSL compute shader
	String error_string;
	Vector<uint8_t> spirv_data = rd->shader_compile_spirv_from_source(
			RD::ShaderStage::SHADER_STAGE_COMPUTE,
			shader_code,
			RD::ShaderLanguage::SHADER_LANGUAGE_GLSL,
			&error_string,
			false); // allow_cache

	bool compile_ok = !spirv_data.is_empty();

	if (!compile_ok) {
		ERR_PRINT("Failed to compile compute shader: " + error_string);
		return RID();
	}

	// Create SPIRV data structure for shader creation
	Vector<RD::ShaderStageSPIRVData> spirv_stages;
	RD::ShaderStageSPIRVData stage_data;
	stage_data.shader_stage = RD::ShaderStage::SHADER_STAGE_COMPUTE;
	stage_data.spirv = spirv_data;
	spirv_stages.push_back(stage_data);

	// Create shader RID
	return rd->shader_create_from_spirv(spirv_stages);
}

bool DDMCompute::create_pipeline(RID &pipeline, RID shader, const Vector<StringName> &uniform_names) {
	if (!rd || shader.is_null()) {
		return false;
	}

	// Create compute pipeline
	pipeline = rd->compute_pipeline_create(shader);
	return !pipeline.is_null();
}

bool DDMCompute::compute_adjacency(const RID &vertex_buffer, const RID &index_buffer,
		RID &output_buffer, int vertex_count) {
	if (!rd || adjacency_pipeline.is_null()) {
		ERR_PRINT("DDMCompute not properly initialized");
		return false;
	}

	// Create uniform set for adjacency computation
	Vector<RD::Uniform> uniforms;

	// Uniform 0: Vertex buffer (read-only)
	RD::Uniform vertex_uniform;
	vertex_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	vertex_uniform.binding = 0;
	vertex_uniform.append_id(vertex_buffer);
	uniforms.push_back(vertex_uniform);

	// Uniform 1: Index buffer (read-only)
	RD::Uniform index_uniform;
	index_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	index_uniform.binding = 1;
	index_uniform.append_id(index_buffer);
	uniforms.push_back(index_uniform);

	// Uniform 2: Output adjacency buffer (write)
	RD::Uniform adjacency_uniform;
	adjacency_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	adjacency_uniform.binding = 2;
	adjacency_uniform.append_id(output_buffer);
	uniforms.push_back(adjacency_uniform);

	// Create uniform set
	RID uniform_set = rd->uniform_set_create(uniforms, adjacency_shader, 0);
	if (uniform_set.is_null()) {
		ERR_PRINT("Failed to create uniform set for adjacency computation");
		return false;
	}

	// Dispatch compute shader
	RD::ComputeListID compute_list = rd->compute_list_begin();
	rd->compute_list_bind_compute_pipeline(compute_list, adjacency_pipeline);
	rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);

	// Dispatch with work groups (256 threads per group)
	int work_groups = (vertex_count + 255) / 256;
	rd->compute_list_dispatch(compute_list, work_groups, 1, 1);

	rd->compute_list_end();

	// Wait for completion
	rd->submit();
	rd->sync();

	// Clean up uniform set
	rd->free_rid(uniform_set);

	return true;
}

bool DDMCompute::compute_laplacian(const RID &adjacency_buffer, RID &output_buffer, int vertex_count) {
	if (!rd || laplacian_pipeline.is_null()) {
		ERR_PRINT("DDMCompute not properly initialized");
		return false;
	}

	// Create uniform set for Laplacian computation
	Vector<RD::Uniform> uniforms;

	// Uniform 0: Adjacency buffer (read-only)
	RD::Uniform adjacency_uniform;
	adjacency_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	adjacency_uniform.binding = 0;
	adjacency_uniform.append_id(adjacency_buffer);
	uniforms.push_back(adjacency_uniform);

	// Uniform 1: Output Laplacian buffer (write)
	RD::Uniform laplacian_uniform;
	laplacian_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	laplacian_uniform.binding = 1;
	laplacian_uniform.append_id(output_buffer);
	uniforms.push_back(laplacian_uniform);

	// Create uniform set
	RID uniform_set = rd->uniform_set_create(uniforms, laplacian_shader, 0);
	if (uniform_set.is_null()) {
		ERR_PRINT("Failed to create uniform set for Laplacian computation");
		return false;
	}

	// Dispatch compute shader
	RD::ComputeListID compute_list = rd->compute_list_begin();
	rd->compute_list_bind_compute_pipeline(compute_list, laplacian_pipeline);
	rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);

	// Dispatch with work groups (256 threads per group)
	int work_groups = (vertex_count + 255) / 256;
	rd->compute_list_dispatch(compute_list, work_groups, 1, 1);

	rd->compute_list_end();

	// Wait for completion
	rd->submit();
	rd->sync();

	// Clean up uniform set
	rd->free_rid(uniform_set);

	return true;
}

bool DDMCompute::compute_omega_matrices(const RID &laplacian_buffer, const RID &vertex_buffer,
		const RID &weights_buffer, RID &output_buffer,
		int vertex_count, int bone_count, int iterations, float lambda) {
	if (!rd || omega_pipeline.is_null()) {
		ERR_PRINT("DDMCompute not properly initialized");
		return false;
	}

	// Create uniform set for omega matrix precomputation
	Vector<RD::Uniform> uniforms;

	// Uniform 0: Vertex buffer (read-only)
	RD::Uniform vertex_uniform;
	vertex_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	vertex_uniform.binding = 0;
	vertex_uniform.append_id(vertex_buffer);
	uniforms.push_back(vertex_uniform);

	// Uniform 1: Weights buffer (read-only)
	RD::Uniform weights_uniform;
	weights_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	weights_uniform.binding = 1;
	weights_uniform.append_id(weights_buffer);
	uniforms.push_back(weights_uniform);

	// Uniform 2: Laplacian buffer (read-only)
	RD::Uniform laplacian_uniform;
	laplacian_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	laplacian_uniform.binding = 2;
	laplacian_uniform.append_id(laplacian_buffer);
	uniforms.push_back(laplacian_uniform);

	// Uniform 3: Output omega matrices buffer (write)
	RD::Uniform omega_uniform;
	omega_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	omega_uniform.binding = 3;
	omega_uniform.append_id(output_buffer);
	uniforms.push_back(omega_uniform);

	// Create uniform set
	RID uniform_set = rd->uniform_set_create(uniforms, omega_shader, 0);
	if (uniform_set.is_null()) {
		ERR_PRINT("Failed to create uniform set for omega matrix computation");
		return false;
	}

	// Dispatch compute shader
	RD::ComputeListID compute_list = rd->compute_list_begin();
	rd->compute_list_bind_compute_pipeline(compute_list, omega_pipeline);
	rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);

	// Dispatch with work groups (256 threads per group)
	int work_groups = (vertex_count + 255) / 256;
	rd->compute_list_dispatch(compute_list, work_groups, 1, 1);

	rd->compute_list_end();

	// Wait for completion
	rd->submit();
	rd->sync();

	// Clean up uniform set
	rd->free_rid(uniform_set);

	return true;
}

bool DDMCompute::deform_mesh(const RID &omega_buffer, const RID &bones_buffer,
		const RID &input_vertices, const RID &input_normals,
		RID &output_vertices, RID &output_normals, int vertex_count) {
	if (!rd || deform_pipeline.is_null()) {
		ERR_PRINT("DDMCompute not properly initialized");
		return false;
	}

	// Create uniform set for mesh deformation
	Vector<RD::Uniform> uniforms;

	// Uniform 0: Input vertices buffer (read-only)
	RD::Uniform input_verts_uniform;
	input_verts_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	input_verts_uniform.binding = 0;
	input_verts_uniform.append_id(input_vertices);
	uniforms.push_back(input_verts_uniform);

	// Uniform 1: Input normals buffer (read-only)
	RD::Uniform input_norms_uniform;
	input_norms_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	input_norms_uniform.binding = 1;
	input_norms_uniform.append_id(input_normals);
	uniforms.push_back(input_norms_uniform);

	// Uniform 2: Bones buffer (read-only)
	RD::Uniform bones_uniform;
	bones_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	bones_uniform.binding = 2;
	bones_uniform.append_id(bones_buffer);
	uniforms.push_back(bones_uniform);

	// Uniform 3: Omega matrices buffer (read-only)
	RD::Uniform omega_uniform;
	omega_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	omega_uniform.binding = 3;
	omega_uniform.append_id(omega_buffer);
	uniforms.push_back(omega_uniform);

	// Uniform 4: Output vertices buffer (write)
	RD::Uniform output_verts_uniform;
	output_verts_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	output_verts_uniform.binding = 4;
	output_verts_uniform.append_id(output_vertices);
	uniforms.push_back(output_verts_uniform);

	// Uniform 5: Output normals buffer (write)
	RD::Uniform output_norms_uniform;
	output_norms_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	output_norms_uniform.binding = 5;
	output_norms_uniform.append_id(output_normals);
	uniforms.push_back(output_norms_uniform);

	// Create uniform set
	RID uniform_set = rd->uniform_set_create(uniforms, deform_shader, 0);
	if (uniform_set.is_null()) {
		ERR_PRINT("Failed to create uniform set for mesh deformation");
		return false;
	}

	// Dispatch compute shader
	RD::ComputeListID compute_list = rd->compute_list_begin();
	rd->compute_list_bind_compute_pipeline(compute_list, deform_pipeline);
	rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);

	// Dispatch with work groups (256 threads per group)
	int work_groups = (vertex_count + 255) / 256;
	rd->compute_list_dispatch(compute_list, work_groups, 1, 1);

	rd->compute_list_end();

	// Wait for completion
	rd->submit();
	rd->sync();

	// Clean up uniform set
	rd->free_rid(uniform_set);

	return true;
}

// CPU implementation for omega matrix precomputation using Gauss-Seidel iteration
// NOTE: This function is for standard Direct Delta Mush, not Enhanced DDM.
// Enhanced DDM uses polar decomposition (see ddm_deformer.cpp) and doesn't need omega matrices.
// This is kept for potential future standard DDM support.
bool DDMCompute::compute_omega_matrices_cpu(const Vector<int> &adjacency_matrix,
		const Vector<float> &laplacian_matrix,
		const Vector<Vector3> &vertices,
		const Vector<float> &weights,
		Vector<float> &omega_matrices,
		int vertex_count, int bone_count, int iterations, float lambda) {
	// Validate input
	if (adjacency_matrix.is_empty() || laplacian_matrix.is_empty()) {
		return false;
	}

	const int max_neighbors = 32; // DDMMesh::maxOmegaCount

	// Allocate omega matrices: 4x4 matrix per vertex per bone
	omega_matrices.resize(vertex_count * bone_count * 16);

	// For each bone, solve L * ω = B using Gauss-Seidel iteration
	// where L is the Laplacian matrix and B contains bone weights
	for (int bone = 0; bone < bone_count; bone++) {
		// Initialize omega values for this bone
		Vector<float> omega;
		omega.resize(vertex_count);
		for (int vi = 0; vi < vertex_count; vi++) {
			omega.set(vi, 0.0f);
		}

		// Extract bone weights for this bone (RHS vector B)
		Vector<float> rhs;
		rhs.resize(vertex_count);
		for (int vi = 0; vi < vertex_count; vi++) {
			int weight_idx = vi * bone_count * 4 + bone * 4;
			// Use the x-component as the scalar weight for this bone
			rhs.set(vi, (weight_idx < weights.size()) ? weights[weight_idx] : 0.0f);
		}

		// Gauss-Seidel iterations with convergence checking and robustness
		// x_new[i] = (b[i] - sum(L[i,j] * x[j])) / L[i,i]
		const float convergence_tolerance = 1e-6f;
		const float min_diagonal = 1e-8f; // Prevent division by zero
		bool converged = false;

		for (int iter = 0; iter < iterations && !converged; iter++) {
			float max_change = 0.0f;

			for (int vi = 0; vi < vertex_count; vi++) {
				float sum = 0.0f;

				// Sum contributions from neighbors using Laplacian weights
				// Laplacian format: [neighbor_index, weight, neighbor_index, weight, ...]
				for (int j = 0; j < max_neighbors; j++) {
					int entry_idx = vi * max_neighbors * 2 + j * 2;
					if (entry_idx + 1 >= laplacian_matrix.size()) {
						break;
					}

					int neighbor_idx = laplacian_matrix[entry_idx];
					float weight = laplacian_matrix[entry_idx + 1];

					if (neighbor_idx >= 0 && neighbor_idx < vertex_count && neighbor_idx != vi) {
						// Use latest omega value (Gauss-Seidel property)
						sum += weight * omega[neighbor_idx];
					}
				}

				// Update omega value with robust diagonal handling
				// L[i,i] is 1.0 for normalized Laplacian, but add regularization for stability
				float diagonal = 1.0f + lambda;

				// Guard against degenerate cases
				if (Math::abs(diagonal) < min_diagonal) {
					diagonal = min_diagonal;
				}

				float old_value = omega[vi];
				float new_value = (rhs[vi] - sum) / diagonal;

				// Check for NaN/Inf and clamp if needed
				if (!Math::is_finite(new_value)) {
					new_value = 0.0f; // Fallback to zero
					WARN_PRINT_ONCE("DDM omega computation produced non-finite value, using fallback");
				}

				omega.set(vi, new_value);

				// Track convergence
				float change = Math::abs(new_value - old_value);
				max_change = MAX(max_change, change);
			}

			// Check for convergence
			if (max_change < convergence_tolerance) {
				converged = true;
			}
		}

		// Warn if didn't converge
		if (!converged && iterations > 0) {
			WARN_PRINT_ONCE("DDM Gauss-Seidel solver did not fully converge within iteration limit");
		}

		// Store omega values in output matrix (4x4 identity with omega on diagonal)
		for (int vi = 0; vi < vertex_count; vi++) {
			int matrix_idx = (vi * bone_count + bone) * 16;

			// Set 4x4 identity matrix with omega value on diagonal
			omega_matrices.set(matrix_idx + 0, omega[vi]);
			omega_matrices.set(matrix_idx + 1, 0.0f);
			omega_matrices.set(matrix_idx + 2, 0.0f);
			omega_matrices.set(matrix_idx + 3, 0.0f);

			omega_matrices.set(matrix_idx + 4, 0.0f);
			omega_matrices.set(matrix_idx + 5, omega[vi]);
			omega_matrices.set(matrix_idx + 6, 0.0f);
			omega_matrices.set(matrix_idx + 7, 0.0f);

			omega_matrices.set(matrix_idx + 8, 0.0f);
			omega_matrices.set(matrix_idx + 9, 0.0f);
			omega_matrices.set(matrix_idx + 10, omega[vi]);
			omega_matrices.set(matrix_idx + 11, 0.0f);

			omega_matrices.set(matrix_idx + 12, 0.0f);
			omega_matrices.set(matrix_idx + 13, 0.0f);
			omega_matrices.set(matrix_idx + 14, 0.0f);
			omega_matrices.set(matrix_idx + 15, omega[vi]);
		}
	}

	return true;
}
