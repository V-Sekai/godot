/**************************************************************************/
/*  ddm_precomputer.h                                                     */
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

/* ddm_precomputer.h */

#ifndef DDM_PRECOMPUTER_H
#define DDM_PRECOMPUTER_H

#include "core/object/ref_counted.h"
#include "scene/resources/mesh.h"

// Eigen for matrix operations
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

class DDMPrecomputer : public RefCounted {
	GDCLASS(DDMPrecomputer, RefCounted);

private:
	// Precomputation data structures
	Vector<int> adjacency_matrix;
	Vector<float> laplacian_matrix;
	Vector<float> omega_matrices;

public:
	// Main precomputation methods
	bool precompute(const Ref<Mesh> &mesh, int iterations, float lambda, float tolerance);

	// Individual computation steps
	bool build_adjacency_matrix(const Ref<Mesh> &mesh, float tolerance);
	bool compute_laplacian_matrix();
	bool precompute_omega_matrices(const Ref<Mesh> &mesh, int iterations, float lambda);

	// Data access
	const Vector<int> &get_adjacency_matrix() const { return adjacency_matrix; }
	const Vector<float> &get_laplacian_matrix() const { return laplacian_matrix; }
	const Vector<float> &get_omega_matrices() const { return omega_matrices; }

protected:
	static void _bind_methods();
};

#endif // DDM_PRECOMPUTER_H
