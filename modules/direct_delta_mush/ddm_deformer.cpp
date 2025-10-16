/**************************************************************************/
/*  ddm_deformer.cpp                                                      */
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

#include "ddm_deformer.h"

#include "core/math/basis.h"

// Stage 1: CPU-Only Basic Direct Delta Mush Implementation
// Simple, readable implementation focusing on correctness over performance

DDMDeformer::DDMDeformer() {
}

DDMDeformer::~DDMDeformer() {
}

void DDMDeformer::_bind_methods() {
	// No GDScript bindings needed for Stage 1 (internal use only)
}

void DDMDeformer::initialize(const MeshData &p_mesh) {
	mesh_data = p_mesh;
	build_adjacency();
}

void DDMDeformer::build_adjacency() {
	int vertex_count = mesh_data.vertices.size();
	
	// Initialize adjacency lists
	adjacency.resize(vertex_count);
	for (int i = 0; i < vertex_count; i++) {
		adjacency[i].clear();
	}
	
	// Build adjacency from triangles
	for (int i = 0; i < mesh_data.indices.size(); i += 3) {
		int v0 = mesh_data.indices[i];
		int v1 = mesh_data.indices[i + 1];
		int v2 = mesh_data.indices[i + 2];
		
		// Helper lambda to check if neighbor exists in adjacency list
		auto has_neighbor = [](const LocalVector<int> &list, int neighbor) -> bool {
			for (unsigned int i = 0; i < list.size(); i++) {
				if (list[i] == neighbor) {
					return true;
				}
			}
			return false;
		};
		
		// Add edges (bidirectional)
		// v0 <-> v1
		if (!has_neighbor(adjacency[v0], v1)) {
			adjacency[v0].push_back(v1);
		}
		if (!has_neighbor(adjacency[v1], v0)) {
			adjacency[v1].push_back(v0);
		}
		
		// v0 <-> v2
		if (!has_neighbor(adjacency[v0], v2)) {
			adjacency[v0].push_back(v2);
		}
		if (!has_neighbor(adjacency[v2], v0)) {
			adjacency[v2].push_back(v0);
		}
		
		// v1 <-> v2
		if (!has_neighbor(adjacency[v1], v2)) {
			adjacency[v1].push_back(v2);
		}
		if (!has_neighbor(adjacency[v2], v1)) {
			adjacency[v2].push_back(v1);
		}
	}
}

Basis DDMDeformer::compute_symmetric_sqrt_inverse(const Basis &M) {
	// Compute (M^T * M)^(-1/2) using eigenvalue decomposition
	// This is used for polar decomposition in Enhanced DDM (paper equation 2)
	
	Basis MTM = M.transposed() * M;
	
	// For a 3x3 symmetric matrix, we can use a simplified approach
	// Get eigenvalues using characteristic polynomial
	// For now, use iterative power method for largest eigenvalue
	
	// Simplified approach: Use matrix inverse and approximate square root
	// This is less accurate but sufficient for Stage 1
	Basis MTM_inv = MTM.inverse();
	
	// Approximate sqrt by averaging M^-1 with its transpose and scaling
	// Better: Use Denman-Beavers iteration for matrix square root
	Basis X = MTM_inv;
	Basis Y = Basis(); // Identity
	
	// 3-4 iterations usually sufficient for convergence
	for (int iter = 0; iter < 4; iter++) {
		Basis X_next = 0.5 * (X + Y.inverse());
		Basis Y_next = 0.5 * (Y + X.inverse());
		X = X_next;
		Y = Y_next;
	}
	
	return X; // This is (M^T * M)^(-1/2)
}

void DDMDeformer::decompose_transform(const Transform3D &M, Transform3D &M_rigid, Transform3D &M_scale) {
	// Enhanced DDM polar decomposition (paper equation 2)
	// Factor M = R * S where R is rotation and S is scale/shear
	// Using: R = M * (M^T * M)^(-1/2)
	
	Basis M_basis = M.basis;
	
	// Compute (M^T * M)^(-1/2)
	Basis sqrt_inv = compute_symmetric_sqrt_inverse(M_basis);
	
	// R = M * (M^T * M)^(-1/2)
	Basis R = M_basis * sqrt_inv;
	
	// S = R^T * M (since M = R * S, then S = R^-1 * M = R^T * M for rotation)
	Basis S = R.transposed() * M_basis;
	
	// Build output transforms
	M_rigid.basis = R;
	M_rigid.origin = M.origin; // Rigid part gets full translation
	
	M_scale.basis = S;
	M_scale.origin = Vector3(0, 0, 0); // Scale part has no translation
}

Vector3 DDMDeformer::compute_laplacian_average(
		int vertex_index,
		const Vector<Vector3> &positions,
		bool uniform_weights) {
	
	const LocalVector<int> &neighbors = adjacency[vertex_index];
	
	if (neighbors.size() == 0) {
		// No neighbors - return original position
		return positions[vertex_index];
	}
	
	Vector3 sum = Vector3(0, 0, 0);
	
	if (uniform_weights) {
		// Simple uniform weights: average of all neighbors
		for (unsigned int i = 0; i < neighbors.size(); i++) {
			sum += positions[neighbors[i]];
		}
		return sum / float(neighbors.size());
	} else {
		// TODO: Cotangent weights (Stage 3 enhancement)
		// For now, fall back to uniform weights
		for (unsigned int i = 0; i < neighbors.size(); i++) {
			sum += positions[neighbors[i]];
		}
		return sum / float(neighbors.size());
	}
}

Vector<Vector3> DDMDeformer::apply_laplacian_smoothing(
		const Vector<Vector3> &positions,
		int iterations,
		float lambda) {
	
	Vector<Vector3> smoothed = positions;
	Vector<Vector3> temp;
	temp.resize(positions.size());
	
	for (int iter = 0; iter < iterations; iter++) {
		// Compute Laplacian for each vertex
		for (int vi = 0; vi < smoothed.size(); vi++) {
			Vector3 laplacian_avg = compute_laplacian_average(vi, smoothed, true);
			
			// Apply smoothing: new_pos = pos + lambda * (avg - pos)
			temp.set(vi, smoothed[vi] + lambda * (laplacian_avg - smoothed[vi]));
		}
		
		// Swap buffers
		smoothed = temp;
	}
	
	return smoothed;
}

Transform3D DDMDeformer::compute_vertex_transform(
		int vertex_index,
		const Vector<Vector3> &original_positions,
		const Vector<Vector3> &smooth_positions,
		const Vector<Transform3D> &bone_transforms) {
	
	// Stage 1 Simplified Approach:
	// Use the difference between original and smoothed positions to estimate local transformation
	
	Vector3 original_pos = original_positions[vertex_index];
	Vector3 smooth_pos = smooth_positions[vertex_index];
	
	// Compute displacement
	Vector3 displacement = smooth_pos - original_pos;
	
	// For Stage 1, we'll use a simple approach:
	// Build a local frame from the neighborhood
	const LocalVector<int> &neighbors = adjacency[vertex_index];
	
	if (neighbors.size() < 2) {
		// Not enough neighbors for meaningful transformation
		// Return identity with translation
		Transform3D result = Transform3D();
		result.origin = displacement;
		return result;
	}
	
	// Build local coordinate frame from first two neighbor directions
	Vector3 v1 = (original_positions[neighbors[0]] - original_pos).normalized();
	Vector3 v2 = (original_positions[neighbors[1]] - original_pos).normalized();
	Vector3 normal = v1.cross(v2).normalized();
	
	if (normal.length_squared() < 0.001) {
		// Degenerate case - use identity
		Transform3D result = Transform3D();
		result.origin = displacement;
		return result;
	}
	
	// Orthogonalize
	Vector3 tangent = normal.cross(v1).normalized();
	Vector3 bitangent = normal.cross(tangent);
	
	// Build basis
	Basis local_basis;
	local_basis.set_column(0, tangent);
	local_basis.set_column(1, bitangent);
	local_basis.set_column(2, normal);
	
	Transform3D result;
	result.basis = local_basis;
	result.origin = displacement;
	
	return result;
}

DDMDeformer::DeformResult DDMDeformer::deform(
		const Vector<Transform3D> &p_bone_transforms,
		const Config &p_config) {
	
	DeformResult result;
	
	if (mesh_data.vertices.size() == 0) {
		return result; // No mesh data
	}
	
	// ENHANCED DDM IMPLEMENTATION (paper equation 1)
	
	// Step 1: Decompose bone transforms into rigid and scale components
	Vector<Transform3D> bone_rigid_transforms;
	Vector<Transform3D> bone_scale_transforms;
	bone_rigid_transforms.resize(p_bone_transforms.size());
	bone_scale_transforms.resize(p_bone_transforms.size());
	
	for (int bi = 0; bi < p_bone_transforms.size(); bi++) {
		Transform3D M_rigid, M_scale;
		decompose_transform(p_bone_transforms[bi], M_rigid, M_scale);
		bone_rigid_transforms.set(bi, M_rigid);
		bone_scale_transforms.set(bi, M_scale);
	}
	
	// Step 2: Compute non-rigid displacement for each vertex (paper: d_ij = M_sj * u_i - u_i)
	Vector<Vector<Vector3>> non_rigid_displacements; // [vertex][bone]
	non_rigid_displacements.resize(mesh_data.vertices.size());
	
	for (int vi = 0; vi < mesh_data.vertices.size(); vi++) {
		Vector3 pos = mesh_data.vertices[vi];
		Vector<Vector3> vertex_displacements;
		vertex_displacements.resize(p_bone_transforms.size());
		
		for (int bi = 0; bi < p_bone_transforms.size(); bi++) {
			// d_ij = M_scale * vertex_pos - vertex_pos
			Vector3 scaled_pos = bone_scale_transforms[bi].xform(pos);
			vertex_displacements.set(bi, scaled_pos - pos);
		}
		
		non_rigid_displacements.set(vi, vertex_displacements);
	}
	
	// Step 3: Apply weighted bone transforms to original positions (equation 1 part 1)
	Vector<Vector3> bone_deformed_positions;
	bone_deformed_positions.resize(mesh_data.vertices.size());
	
	for (int vi = 0; vi < mesh_data.vertices.size(); vi++) {
		Vector3 pos = mesh_data.vertices[vi];
		Vector3 deformed_pos = Vector3(0, 0, 0);
		float total_weight = 0.0f;
		
		// Weighted sum: Σ w_ij * (M_rj * D_ij * Ω_ij)
		// Simplified for Stage 1: Ω_ij = I (identity), so just M_rj * (I + d_ij)
		for (int bi = 0; bi < 4; bi++) {
			int bone_idx_offset = vi * 4 + bi;
			if (bone_idx_offset >= mesh_data.bone_indices.size()) {
				break;
			}
			
			int bone_idx = mesh_data.bone_indices[bone_idx_offset];
			float weight = mesh_data.bone_weights[bone_idx_offset];
			
			if (bone_idx >= 0 && bone_idx < p_bone_transforms.size() && weight > 0.0001f) {
				// Get non-rigid displacement for this bone
				Vector3 d_ij = non_rigid_displacements[vi][bone_idx];
				
				// Apply: M_rigid * (pos + d_ij)
				Vector3 displaced_pos = pos + d_ij;
				Vector3 transformed_pos = bone_rigid_transforms[bone_idx].xform(displaced_pos);
				
				deformed_pos += transformed_pos * weight;
				total_weight += weight;
			}
		}
		
		if (total_weight > 0.0001f) {
			bone_deformed_positions.set(vi, deformed_pos / total_weight);
		} else {
			// No bone influence - use original position
			bone_deformed_positions.set(vi, pos);
		}
	}
	
	// Step 4: Apply Laplacian smoothing to bone-deformed positions
	Vector<Vector3> smoothed_positions = apply_laplacian_smoothing(
			bone_deformed_positions,
			p_config.iterations,
			p_config.smooth_lambda);
	
	// Step 5: Final vertex positions
	result.vertices.resize(mesh_data.vertices.size());
	result.normals.resize(mesh_data.normals.size());
	
	for (int vi = 0; vi < mesh_data.vertices.size(); vi++) {
		result.vertices.set(vi, smoothed_positions[vi]);
		
		// Transform normals (simplified)
		if (vi < mesh_data.normals.size()) {
			// For now, keep original normals
			// TODO: Properly transform normals based on local deformation
			result.normals.set(vi, mesh_data.normals[vi]);
		}
	}
	
	result.success = true;
	return result;
}
