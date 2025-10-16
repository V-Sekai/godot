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
		
		// Add edges (bidirectional)
		// v0 <-> v1
		if (!adjacency[v0].has(v1)) {
			adjacency[v0].push_back(v1);
		}
		if (!adjacency[v1].has(v0)) {
			adjacency[v1].push_back(v0);
		}
		
		// v0 <-> v2
		if (!adjacency[v0].has(v2)) {
			adjacency[v0].push_back(v2);
		}
		if (!adjacency[v2].has(v0)) {
			adjacency[v2].push_back(v0);
		}
		
		// v1 <-> v2
		if (!adjacency[v1].has(v2)) {
			adjacency[v1].push_back(v2);
		}
		if (!adjacency[v2].has(v1)) {
			adjacency[v2].push_back(v1);
		}
	}
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
	
	// Step 1: Apply bone transformations to get initial deformed positions
	Vector<Vector3> bone_deformed_positions;
	bone_deformed_positions.resize(mesh_data.vertices.size());
	
	for (int vi = 0; vi < mesh_data.vertices.size(); vi++) {
		Vector3 pos = mesh_data.vertices[vi];
		Vector3 deformed_pos = Vector3(0, 0, 0);
		float total_weight = 0.0f;
		
		// Apply bone transforms with weights (linear blend skinning)
		for (int bi = 0; bi < 4; bi++) {
			int bone_idx_offset = vi * 4 + bi;
			if (bone_idx_offset >= mesh_data.bone_indices.size()) {
				break;
			}
			
			int bone_idx = mesh_data.bone_indices[bone_idx_offset];
			float weight = mesh_data.bone_weights[bone_idx_offset];
			
			if (bone_idx >= 0 && bone_idx < p_bone_transforms.size() && weight > 0.0001f) {
				deformed_pos += p_bone_transforms[bone_idx].xform(pos) * weight;
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
	
	// Step 2: Apply Laplacian smoothing to bone-deformed positions
	Vector<Vector3> smoothed_positions = apply_laplacian_smoothing(
			bone_deformed_positions,
			p_config.iterations,
			p_config.smooth_lambda);
	
	// Step 3: Compute local transformations and apply to original mesh
	result.vertices.resize(mesh_data.vertices.size());
	result.normals.resize(mesh_data.normals.size());
	
	for (int vi = 0; vi < mesh_data.vertices.size(); vi++) {
		// For Stage 1: Simplified approach - use smoothed position directly
		// (Full DDM would compute transformation matrices here)
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
