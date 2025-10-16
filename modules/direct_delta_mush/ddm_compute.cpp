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
	adjacency_shader = load_shader_from_file("shaders/adjacency.compute.glsl");
	laplacian_shader = load_shader_from_file("shaders/laplacian.compute.glsl");
	omega_shader = load_shader_from_file("shaders/omega_precompute.compute.glsl");
	deform_shader = load_shader_from_file("shaders/deform.compute.glsl");

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
	// TODO: Implement adjacency matrix computation on GPU using proper uniform sets
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
