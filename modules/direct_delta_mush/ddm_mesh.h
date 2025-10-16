/**************************************************************************/
/*  ddm_mesh.h                                                            */
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

/* ddm_mesh.h */

#ifndef DDM_MESH_H
#define DDM_MESH_H

#include "core/math/transform_3d.h"
#include "core/templates/rid.h"
#include "core/templates/vector.h"
#include "scene/resources/mesh.h"

// DDMMesh is a mesh resource that contains precomputed Direct Delta Mush data
class DDMMesh : public Mesh {
	GDCLASS(DDMMesh, Mesh);

private:
	RID source_mesh; // Original mesh with bone weights
	RID ddm_mesh; // Processed mesh for rendering

	// Precomputed data
	Vector<int> adjacency_matrix;
	Vector<float> laplacian_matrix;
	Vector<float> omega_matrices;

	int vertex_count = 0;
	int bone_count = 0;

protected:
	static void _bind_methods();

public:
	DDMMesh();
	~DDMMesh();

	RID get_rid() const;
	void set_rid(RID p_rid);

	// Precomputation
	void set_precomputed_data(const Vector<int> &p_adjacency,
			const Vector<float> &p_laplacian,
			const Vector<float> &p_omega);
	void set_mesh_data(RID p_source_mesh, int p_vertex_count, int p_bone_count);

	// Data access
	const Vector<int> &get_adjacency_matrix() const { return adjacency_matrix; }
	const Vector<float> &get_laplacian_matrix() const { return laplacian_matrix; }
	const Vector<float> &get_omega_matrices() const { return omega_matrices; }

	int get_vertex_count() const { return vertex_count; }
	int get_bone_count() const { return bone_count; }

	// Runtime deformation
	void update_deformation(const Vector<Transform3D> &bone_transforms);
};

#endif // DDM_MESH_H
