/**************************************************************************/
/*  ddm_compute.h                                                         */
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

#pragma once

/* ddm_compute.h */

#ifndef DDM_COMPUTE_H
#define DDM_COMPUTE_H

#include "core/object/ref_counted.h"
#include "servers/rendering/rendering_device.h"

// Eigen for matrix operations
#include <Eigen/Core>
#include <Eigen/Dense>

class DDMCompute : public RefCounted {
	GDCLASS(DDMCompute, RefCounted);

private:
	RenderingDevice *rd = nullptr;

	// Compute pipelines
	RID adjacency_pipeline;
	RID laplacian_pipeline;
	RID omega_pipeline;
	RID deform_pipeline;

	// Shader RIDs
	RID adjacency_shader;
	RID laplacian_shader;
	RID omega_shader;
	RID deform_shader;

public:
	bool initialize(RenderingDevice *p_rd);

	// Compute operations
	bool compute_adjacency(const RID &vertex_buffer, const RID &index_buffer,
			RID &output_buffer, int vertex_count);
	bool compute_laplacian(const RID &adjacency_buffer, RID &output_buffer, int vertex_count);
	bool compute_omega_matrices(const RID &laplacian_buffer, const RID &vertex_buffer,
			const RID &weights_buffer, RID &output_buffer,
			int vertex_count, int bone_count, int iterations, float lambda);
	bool deform_mesh(const RID &omega_buffer, const RID &bones_buffer,
			const RID &input_vertices, const RID &input_normals,
			RID &output_vertices, RID &output_normals, int vertex_count);

	void cleanup();

protected:
	static void _bind_methods();

private:
	RID load_shader(const String &shader_code);
	bool create_pipeline(RID &pipeline, RID shader, const Vector<StringName> &uniform_names);
};

#endif // DDM_COMPUTE_H
