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
	// TODO: Implement adjacency computation on GPU using proper uniform sets
	return false;
}

bool DDMCompute::compute_laplacian(const RID &adjacency_buffer, RID &output_buffer, int vertex_count) {
	// TODO: Implement Laplacian matrix computation on GPU using proper uniform sets
	return false;
}

bool DDMCompute::compute_omega_matrices(const RID &laplacian_buffer, const RID &vertex_buffer,
		const RID &weights_buffer, RID &output_buffer,
		int vertex_count, int bone_count, int iterations, float lambda) {
	// TODO: Implement Omega matrix precomputation on GPU using proper uniform sets
	return false;
}

bool DDMCompute::deform_mesh(const RID &omega_buffer, const RID &bones_buffer,
		const RID &input_vertices, const RID &input_normals,
		RID &output_vertices, RID &output_normals, int vertex_count) {
	// TODO: Implement runtime mesh deformation on GPU using proper uniform sets
	return false;
}

// CPU implementation using AMGCL for omega matrix precomputation
bool DDMCompute::compute_omega_matrices_cpu(const Vector<int> &adjacency_matrix,
		const Vector<float> &laplacian_matrix,
		const Vector<Vector3> &vertices,
		const Vector<float> &weights,
		Vector<float> &omega_matrices,
		int vertex_count, int bone_count, int iterations, float lambda) {
	if (adjacency_matrix.is_empty() || laplacian_matrix.is_empty()) {
		return false;
	}

	const int max_neighbors = 32; // DDMMesh::maxOmegaCount

	// Build AMGCL-compatible sparse matrix from Laplacian data
	// Laplacian format: [neighbor_index, weight, neighbor_index, weight, ...] per vertex
	std::vector<int> ptr(vertex_count + 1, 0);
	std::vector<int> col;
	std::vector<double> val;

	// Convert Laplacian matrix to CRS format
	for (int vi = 0; vi < vertex_count; vi++) {
		ptr[vi] = col.size();

		// Add diagonal element (always present in Laplacian)
		col.push_back(vi);
		val.push_back(1.0); // Diagonal is 1.0 for normalized Laplacian

		// Add off-diagonal elements
		for (int j = 0; j < max_neighbors; j++) {
			int entry_idx = vi * max_neighbors * 2 + j * 2;
			if (entry_idx + 1 >= laplacian_matrix.size()) {
				break;
			}

			int neighbor_idx = laplacian_matrix[entry_idx];
			float weight = laplacian_matrix[entry_idx + 1];

			if (neighbor_idx >= 0 && neighbor_idx != vi) { // Skip invalid and diagonal
				col.push_back(neighbor_idx);
				val.push_back(weight);
			}
		}
	}
	ptr[vertex_count] = col.size();

	// Create AMGCL solver for the Laplacian system
	typedef amgcl::make_solver<
			amgcl::amg<
					amgcl::backend::builtin<double>,
					amgcl::coarsening::smoothed_aggregation,
					amgcl::relaxation::spai0>,
			amgcl::solver::cg<amgcl::backend::builtin<double>>>
			Solver;

	Solver solver(std::tie(vertex_count, ptr, col, val));

	// For each bone, solve L * ω = B where B contains bone weights
	omega_matrices.resize(vertex_count * bone_count * 16); // 4x4 matrices per vertex per bone

	for (int bone = 0; bone < bone_count; bone++) {
		// Build RHS vector B for this bone (bone weights)
		std::vector<double> rhs(vertex_count, 0.0);

		// Extract bone weights for this bone (4 weights per vertex: x,y,z,w)
		for (int vi = 0; vi < vertex_count; vi++) {
			int weight_idx = vi * bone_count * 4 + bone * 4;
			// Use the x-component as the scalar weight for this bone
			if (weight_idx < weights.size()) {
				rhs[vi] = weights[weight_idx];
			}
		}

		// Solve L * ω = B
		std::vector<double> omega(vertex_count, 0.0);
		std::tie(iterations, lambda) = solver(rhs, omega); // lambda is residual tolerance

		// Store omega values in output matrix (4x4 identity with omega on diagonal for now)
		for (int vi = 0; vi < vertex_count; vi++) {
			int matrix_idx = (vi * bone_count + bone) * 16;
			// Store as identity matrix with omega on diagonal
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					int idx = matrix_idx + i * 4 + j;
					if (i == j) {
						omega_matrices.set(idx, omega[vi]);
					} else {
						omega_matrices.set(idx, (i == j) ? 1.0f : 0.0f);
					}
				}
			}
		}
	}

	return true;
}
