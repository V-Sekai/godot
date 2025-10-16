/**************************************************************************/
/*  direct_delta_mush.h                                                   */
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

/* direct_delta_mush.h */

#ifndef DIRECT_DELTA_MUSH_H
#define DIRECT_DELTA_MUSH_H

#include "scene/3d/mesh_instance_3d.h"
#include "servers/rendering/rendering_device.h"

class DirectDeltaMushMeshInstance3D : public MeshInstance3D {
GDCLASS(DirectDeltaMushMeshInstance3D, MeshInstance3D);

private:
	// Direct Delta Mush parameters
	int iterations = 30;
	float smooth_lambda = 0.9f;
	float adjacency_tolerance = 1e-4f;
	bool use_compute = true;

	// Precomputed data
	RID omega_buffer;
	RID adjacency_buffer;
	RID laplacian_buffer;

	// Runtime state
	Ref<Mesh> deformed_mesh;
	RenderingDevice *rd = nullptr;

	// Internal methods
	void precompute_data();
	void update_deformation();
	void build_adjacency_matrix();
	void compute_laplacian_matrix();
	void precompute_omega_matrices();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
DirectDeltaMushMeshInstance3D();
~DirectDeltaMushMeshInstance3D();

	// Property setters/getters
	void set_iterations(int p_iterations);
	int get_iterations() const;

	void set_smooth_lambda(float p_lambda);
	float get_smooth_lambda() const;

	void set_adjacency_tolerance(float p_tolerance);
	float get_adjacency_tolerance() const;

	void set_use_compute(bool p_use_compute);
	bool get_use_compute() const;

	// Public methods
	void precompute();
};

#endif // DIRECT_DELTA_MUSH_H
