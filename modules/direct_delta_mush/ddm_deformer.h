/**************************************************************************/
/*  ddm_deformer.h                                                        */
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

#ifndef DDM_DEFORMER_H
#define DDM_DEFORMER_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/templates/vector.h"
#include "core/variant/variant.h"

// Stage 1: CPU-Only Basic Direct Delta Mush
// Simple implementation of the basic DDM algorithm without GPU, double precision, or enhanced features

class DDMDeformer : public RefCounted {
	GDCLASS(DDMDeformer, RefCounted);

public:
	// Configuration
	struct Config {
		int iterations = 10; // Laplacian smoothing iterations
		float smooth_lambda = 0.5; // Smoothing weight (0.0 = no smoothing, 1.0 = full smoothing)
		bool use_uniform_weights = true; // Use uniform weights (true) or cotangent weights (false)
	};

	// Mesh data required for DDM
	struct MeshData {
		Vector<Vector3> vertices; // Original vertex positions
		Vector<Vector3> normals; // Original vertex normals
		Vector<int> indices; // Triangle indices
		Vector<float> bone_weights; // Bone weights (4 per vertex)
		Vector<int> bone_indices; // Bone indices (4 per vertex)
	};

	// Deformation result
	struct DeformResult {
		Vector<Vector3> vertices; // Deformed vertex positions
		Vector<Vector3> normals; // Deformed vertex normals
		bool success = false;
	};

	DDMDeformer();
	~DDMDeformer();

	// Initialize with mesh data
	void initialize(const MeshData &p_mesh);

	// Perform DDM deformation
	DeformResult deform(const Vector<Transform3D> &p_bone_transforms, const Config &p_config);

protected:
	static void _bind_methods();

private:
	// Mesh data
	MeshData mesh_data;
	
	// Adjacency list (neighbors for each vertex)
	LocalVector<LocalVector<int>> adjacency;
	
	// Build adjacency list from triangles
	void build_adjacency();
	
	// Apply Laplacian smoothing
	Vector<Vector3> apply_laplacian_smoothing(
			const Vector<Vector3> &positions,
			int iterations,
			float lambda);
	
	// Compute local transformation at a vertex
	Transform3D compute_vertex_transform(
			int vertex_index,
			const Vector<Vector3> &original_positions,
			const Vector<Vector3> &smooth_positions,
			const Vector<Transform3D> &bone_transforms);
	
	// Helper: Compute weighted average position (for Laplacian)
	Vector3 compute_laplacian_average(
			int vertex_index,
			const Vector<Vector3> &positions,
			bool uniform_weights);
	
	// Enhanced DDM: Polar decomposition helpers
	// Factor transform M into M_rigid * M_scale using square root method (paper equation 2)
	void decompose_transform(const Transform3D &M, Transform3D &M_rigid, Transform3D &M_scale);
	
	// Compute symmetric matrix square root inverse: (M^T * M)^(-1/2)
	Basis compute_symmetric_sqrt_inverse(const Basis &M);
};

#endif // DDM_DEFORMER_H
