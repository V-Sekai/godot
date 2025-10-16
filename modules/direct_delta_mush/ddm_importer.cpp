/**************************************************************************/
/*  ddm_importer.cpp                                                      */
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

/* ddm_importer.cpp */

#include "ddm_importer.h"

#include "core/config/engine.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/resources/surface_tool.h"

void DDMImporter::_bind_methods() {
	ClassDB::bind_method(D_METHOD("import_mesh", "mesh", "import_mode"), &DDMImporter::import_mesh);

	BIND_ENUM_CONSTANT(IMPORT_TIME_PRECOMPUTE);
	BIND_ENUM_CONSTANT(RUNTIME_PRECOMPUTE);
	BIND_ENUM_CONSTANT(FULL_RUNTIME);
}

DDMImporter::MeshSurfaceData::MeshSurfaceData(const Array &p_mesh_arrays) {
	vertex_array = p_mesh_arrays[Mesh::ARRAY_VERTEX];
	normal_array = p_mesh_arrays[Mesh::ARRAY_NORMAL];
	index_array = p_mesh_arrays[Mesh::ARRAY_INDEX];
	if (p_mesh_arrays.size() > Mesh::ARRAY_BONES) {
		bones_array = p_mesh_arrays[Mesh::ARRAY_BONES];
		weights_array = p_mesh_arrays[Mesh::ARRAY_WEIGHTS];
	}
}

DDMImporter::DDMImporter() {
	// Initialize with default values
}

DDMImporter::~DDMImporter() {
	// Cleanup
}

Ref<DDMMesh> DDMImporter::import_mesh(const Ref<Mesh> &mesh, ImportMode import_mode) {
	ERR_FAIL_COND_V(!mesh.is_valid(), nullptr);

	if (!validate_mesh_data(mesh)) {
		ERR_PRINT("Invalid mesh data for Direct Delta Mush import");
		return nullptr;
	}

	MeshSurfaceData surface_data;
	if (!extract_mesh_data(mesh, surface_data)) {
		ERR_PRINT("Failed to extract mesh surface data");
		return nullptr;
	}

	// Build adjacency matrix (always done at import time for performance)
	if (!build_adjacency_matrix(surface_data, 1e-4f)) {
		ERR_PRINT("Failed to build adjacency matrix");
		return nullptr;
	}

	// Compute Laplacian matrix (always done at import time)
	if (!compute_laplacian_matrix()) {
		ERR_PRINT("Failed to compute Laplacian matrix");
		return nullptr;
	}

	// Create the DDM mesh resource
	Ref<DDMMesh> ddm_mesh = create_ddm_mesh(mesh, surface_data);
	if (!ddm_mesh.is_valid()) {
		ERR_PRINT("Failed to create DDM mesh");
		return nullptr;
	}

	// Set precomputed data
	ddm_mesh->set_precomputed_data(adjacency_matrix, laplacian_matrix, Vector<float>());

	return ddm_mesh;
}

bool DDMImporter::extract_mesh_data(const Ref<Mesh> &mesh, MeshSurfaceData &surface_data) {
	// Get the first surface (assuming single surface mesh for now)
	Array surface_arrays = mesh->surface_get_arrays(0);
	if (surface_arrays.is_empty()) {
		return false;
	}

	surface_data = MeshSurfaceData(surface_arrays);
	return true;
}

bool DDMImporter::build_adjacency_matrix(const MeshSurfaceData &surface_data, float tolerance) {
	// Ported from Unity MeshUtils.BuildAdjacencyMatrix

	int vertex_count = surface_data.vertex_array.size();
	const int max_neighbors = 32; // DDMMesh::maxOmegaCount

	adjacency_matrix.resize(vertex_count * max_neighbors);

	// Initialize adjacency matrix
	for (int i = 0; i < adjacency_matrix.size(); i++) {
		adjacency_matrix.set(i, -1);
	}

	if (Math::is_zero_approx(tolerance)) {
		// Direct adjacency building without vertex merging
		for (int tri = 0; tri < surface_data.index_array.size(); tri += 3) {
			add_edge_to_adjacency_direct(surface_data.index_array[tri],
					surface_data.index_array[tri + 1]);
			add_edge_to_adjacency_direct(surface_data.index_array[tri],
					surface_data.index_array[tri + 2]);
			add_edge_to_adjacency_direct(surface_data.index_array[tri + 1],
					surface_data.index_array[tri + 2]);
		}
	} else {
		// Handle vertex merging for close positions
		Vector<int> map_to_unique = map_vertices_to_unique_positions(surface_data.vertex_array, tolerance);

		for (int tri = 0; tri < surface_data.index_array.size(); tri += 3) {
			add_edge_to_adjacency_with_mapping(map_to_unique,
					surface_data.index_array[tri],
					surface_data.index_array[tri + 1]);
			add_edge_to_adjacency_with_mapping(map_to_unique,
					surface_data.index_array[tri],
					surface_data.index_array[tri + 2]);
			add_edge_to_adjacency_with_mapping(map_to_unique,
					surface_data.index_array[tri + 1],
					surface_data.index_array[tri + 2]);
		}

		broadcast_adjacency_from_unique_to_all(map_to_unique);
	}

	return true;
}

bool DDMImporter::compute_laplacian_matrix() {
	// Ported from Unity MeshUtils.BuildLaplacianMatrixFromAdjacentMatrix

	if (adjacency_matrix.is_empty()) {
		return false;
	}

	const int max_neighbors = 32;
	int vertex_count = adjacency_matrix.size() / max_neighbors;

	// Laplacian matrix stored as [index, weight] pairs, similar to adjacency
	// Format: [neighbor_index, weight, neighbor_index, weight, ...]
	// For each vertex, we store up to max_neighbors * 2 entries (index + weight)
	laplacian_matrix.resize(vertex_count * max_neighbors * 2);

	// Initialize to -1 (no neighbor)
	for (int i = 0; i < laplacian_matrix.size(); i++) {
		laplacian_matrix.set(i, -1.0f);
	}

	for (int vi = 0; vi < vertex_count; vi++) {
		// Calculate vertex degree (number of neighbors)
		int vi_degree = 0;
		for (int j = 0; j < max_neighbors; j++) {
			int neighbor_idx = adjacency_matrix[vi * max_neighbors + j];
			if (neighbor_idx < 0) {
				break; // No more neighbors
			}
			vi_degree++;
		}

		if (vi_degree == 0) {
			continue; // Isolated vertex
		}

		// Build Laplacian row for vertex vi
		// For normalized Laplacian: L[i,j] = -1/deg(i) for neighbors, L[i,i] = 1
		int laplacian_row_start = vi * max_neighbors * 2;

		for (int j = 0; j < max_neighbors; j++) {
			int neighbor_idx = adjacency_matrix[vi * max_neighbors + j];
			if (neighbor_idx < 0) {
				break; // No more neighbors
			}

			// Store neighbor index and weight
			int entry_idx = laplacian_row_start + j * 2;
			laplacian_matrix.set(entry_idx, neighbor_idx); // neighbor index
			laplacian_matrix.set(entry_idx + 1, -1.0f / vi_degree); // weight: -1/degree
		}

		// Set diagonal element (L[i,i] = 1 for normalized Laplacian)
		// Store at the end of the row
		int diagonal_idx = laplacian_row_start + (max_neighbors - 1) * 2;
		laplacian_matrix.set(diagonal_idx, vi); // self index
		laplacian_matrix.set(diagonal_idx + 1, 1.0f); // weight: 1.0
	}

	return true;
}

Ref<DDMMesh> DDMImporter::create_ddm_mesh(const Ref<Mesh> &source_mesh, const MeshSurfaceData &surface_data) {
	Ref<DDMMesh> ddm_mesh;
	ddm_mesh.instantiate();

	// Set mesh data
	ddm_mesh->set_mesh_data(source_mesh->get_rid(),
			surface_data.vertex_array.size(),
			surface_data.bones_array.size() / 4); // 4 bones per vertex

	return ddm_mesh;
}

MeshInstance3D *DDMImporter::replace_mesh_instance_with_ddm(MeshInstance3D *mesh_instance, ImportMode import_mode) {
	// TODO: Implement replacement logic similar to TopologyDataImporter
	// This would create a DirectDeltaMushMeshInstance3D node and replace the MeshInstance3D
	return mesh_instance;
}

bool DDMImporter::validate_mesh_data(const Ref<Mesh> &mesh) const {
	if (!mesh.is_valid()) {
		return false;
	}

	// Check for bone weights
	Array surface_arrays = mesh->surface_get_arrays(0);
	if (surface_arrays.size() <= Mesh::ARRAY_BONES) {
		ERR_PRINT("Mesh must have bone weights for Direct Delta Mush");
		return false;
	}

	return true;
}

int DDMImporter::find_vertex_adjacency(const MeshSurfaceData &surface_data, int vertex_index, float tolerance) {
	// TODO: Implement vertex adjacency finding with tolerance
	return -1;
}

// Helper methods for adjacency matrix building

Vector<int> DDMImporter::map_vertices_to_unique_positions(const PackedVector3Array &vertices, float min_sqr_distance) const {
	// Ported from Unity MeshUtils.MapVerticesToUniquePositions

	Vector<int> map_to_unique;
	map_to_unique.resize(vertices.size());

	// Initialize all to -1
	for (int i = 0; i < map_to_unique.size(); i++) {
		map_to_unique.set(i, -1);
	}

	for (int i = 0; i < vertices.size(); i++) {
		for (int j = i; j < vertices.size(); j++) {
			if (map_to_unique[j] != -1) {
				continue; // Skip if already pointing to unique position
			}

			int u = map_to_unique[i];
			if (u == -1) {
				u = i;
			}

			Vector3 vi = vertices[i];
			Vector3 vj = vertices[j];

			float dx = vi.x - vj.x;
			float dy = vi.y - vj.y;
			float dz = vi.z - vj.z;

			if (dx * dx + dy * dy + dz * dz <= min_sqr_distance) {
				if (map_to_unique[i] == -1) {
					map_to_unique.set(i, u); // Found new unique vertex
				}
				map_to_unique.set(j, u);
			}
		}
	}

	// Verify all vertices are mapped
	for (int i = 0; i < map_to_unique.size(); i++) {
		ERR_FAIL_COND_V(map_to_unique[i] == -1, Vector<int>());
	}

	return map_to_unique;
}

void DDMImporter::add_vertex_to_adjacency(int adjacency_idx, int from, int to) {
	const int max_neighbors = 32;
	for (int i = 0; i < max_neighbors; i++) {
		int idx = from * max_neighbors + i;
		if (adjacency_matrix[idx] == to) {
			break; // Already exists
		}
		if (adjacency_matrix[idx] == -1) {
			adjacency_matrix.set(idx, to);
			break;
		}
	}
}

void DDMImporter::add_edge_to_adjacency_direct(int v0, int v1) {
	add_vertex_to_adjacency(0, v0, v1);
	add_vertex_to_adjacency(0, v1, v0);
}

void DDMImporter::add_edge_to_adjacency_with_mapping(const Vector<int> &map_to_unique, int v0, int v1) {
	int u0 = map_to_unique[v0];
	int u1 = map_to_unique[v1];
	add_edge_to_adjacency_direct(u0, u1);
}

void DDMImporter::broadcast_adjacency_from_unique_to_all(const Vector<int> &map_to_unique) const {
	// Ported from Unity MeshUtils.BroadcastAdjacencyFromUniqueToAllVertices

	const int max_neighbors = 32;
	ERR_FAIL_COND(adjacency_matrix.size() != map_to_unique.size() * max_neighbors);

	for (int i = 0; i < map_to_unique.size(); i++) {
		int u = map_to_unique[i];
		if (u == i) {
			continue;
		}

		// Check if adjacency slot is empty for this vertex
		bool is_empty = true;
		for (int j = 0; j < max_neighbors; j++) {
			int idx = i * max_neighbors + j;
			if (adjacency_matrix[idx] != -1) {
				is_empty = false;
				break;
			}
		}

		if (is_empty) {
			// Copy adjacency from unique vertex
			for (int j = 0; j < max_neighbors; j++) {
				int src_idx = u * max_neighbors + j;
				int dst_idx = i * max_neighbors + j;

				if (adjacency_matrix[src_idx] == -1) {
					break;
				}
				adjacency_matrix.set(dst_idx, adjacency_matrix[src_idx]);
			}
		}
	}
}
