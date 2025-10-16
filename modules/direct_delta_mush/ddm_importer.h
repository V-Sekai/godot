/**************************************************************************/
/*  ddm_importer.h                                                        */
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

/* ddm_importer.h */

#ifndef DDM_IMPORTER_H
#define DDM_IMPORTER_H

#include "core/object/ref_counted.h"
#include "core/templates/vector.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/resources/mesh.h"

#include "ddm_mesh.h"

class DDMImporter : public RefCounted {
	GDCLASS(DDMImporter, RefCounted);

public:
	enum ImportMode {
		IMPORT_TIME_PRECOMPUTE = 0, // Precompute everything at import time
		RUNTIME_PRECOMPUTE, // Precompute adjacency/laplacian at import, omega at runtime
		FULL_RUNTIME, // Everything at runtime
	};

private:
	struct MeshSurfaceData {
		PackedVector3Array vertex_array;
		PackedVector3Array normal_array;
		PackedInt32Array index_array;
		PackedInt32Array bones_array;
		Vector<float> weights_array;

		MeshSurfaceData(const Array &p_mesh_arrays);
		MeshSurfaceData() {}
	};

	// Precomputation data structures
	Vector<int> adjacency_matrix;
	Vector<float> laplacian_matrix;

protected:
	static void _bind_methods();

public:
	DDMImporter();
	~DDMImporter();

	// Main import function
	Ref<DDMMesh> import_mesh(const Ref<Mesh> &mesh, ImportMode import_mode);

	// Individual processing steps
	bool extract_mesh_data(const Ref<Mesh> &mesh, MeshSurfaceData &surface_data);
	bool build_adjacency_matrix(const MeshSurfaceData &surface_data, float tolerance);
	bool compute_laplacian_matrix();
	Ref<DDMMesh> create_ddm_mesh(const Ref<Mesh> &source_mesh, const MeshSurfaceData &surface_data);

	// Utility functions
	static MeshInstance3D *replace_mesh_instance_with_ddm(MeshInstance3D *mesh_instance, ImportMode import_mode);

private:
	// Helper methods
	bool validate_mesh_data(const Ref<Mesh> &mesh) const;
	int find_vertex_adjacency(const MeshSurfaceData &surface_data, int vertex_index, float tolerance);

	// Processing methods
	Vector<int> map_vertices_to_unique_positions(const PackedVector3Array &vertices, float min_sqr_distance) const;
	void add_vertex_to_adjacency(int adjacency_idx, int from, int to);
	void add_edge_to_adjacency_direct(int v0, int v1);
	void add_edge_to_adjacency_with_mapping(const Vector<int> &map_to_unique, int v0, int v1);
	void broadcast_adjacency_from_unique_to_all(const Vector<int> &map_to_unique);
};

VARIANT_ENUM_CAST(DDMImporter::ImportMode);

#endif // DDM_IMPORTER_H
