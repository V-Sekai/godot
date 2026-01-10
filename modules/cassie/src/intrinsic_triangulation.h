/**************************************************************************/
/*  intrinsic_triangulation.h                                             */
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

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/vector.h"
#include "scene/resources/mesh.h"

/// @brief Intrinsic triangulation and remeshing for triangle meshes.
///
/// IntrinsicTriangulation implements intrinsic Delaunay triangulation and mesh
/// refinement. This technique improves mesh quality by flipping edges based on
/// intrinsic edge lengths rather than embedded geometry, preserving the surface
/// while optimizing triangle quality.
///
/// The intrinsic approach allows for:
/// - Robust triangulation even with poor-quality input meshes
/// - Edge flipping without changing vertex positions
/// - Optimization based on intrinsic geodesic distances rather than 3D embedding
///
/// @par Algorithms Implemented:
/// - **Edge Flipping to Delaunay**: Iteratively flips edges to satisfy Delaunay criterion
/// - **Delaunay Criterion**: Sum of opposite angles ≤ π for intrinsic metric
/// - **Mesh Refinement**: Split/collapse edges for uniform edge lengths
/// - **Position Smoothing**: Laplacian smoothing on 3D vertex positions
///
/// @par Delaunay Criterion:
/// An edge is Delaunay if the sum of the two opposite angles is ≤ π.
/// This is checked using the circumcircle test applied to intrinsic distances.
///
/// @par Usage Example:
/// @code
/// var intrinsic = IntrinsicTriangulation.new()
///
/// # Initialize from triangulated mesh
/// intrinsic.set_mesh(triangulated_mesh)
///
/// # Configure flipping parameters
/// intrinsic.set_max_flip_iterations(100)
///
/// # Achieve Delaunay triangulation
/// var converged = intrinsic.flip_to_delaunay()
///
/// # Refine to uniform edge lengths
/// intrinsic.refine_intrinsic_triangulation(0.5)
///
/// # Smooth vertex positions
/// intrinsic.smooth_intrinsic_positions(5)
///
/// # Get refined result
/// var refined_mesh = intrinsic.get_mesh()
/// @endcode
///
/// @par References:
/// - Gillespie, Sharp, Crane: "Integer Coordinates for Intrinsic Geometry Processing"
/// - Applies intrinsic methods with unified coordinate system
///
/// @see CassieSurface for automatic pipeline control
/// @see PolygonTriangulationGodot for initial triangulation

// Edge structure for intrinsic triangulation
struct IntrinsicEdge {
	int v0, v1; // Vertex indices
	int t0, t1; // Adjacent triangle indices (-1 if boundary)
	double length; // Intrinsic edge length
	bool is_original; // True if original mesh edge, false if flipped

	IntrinsicEdge() :
			v0(-1), v1(-1), t0(-1), t1(-1), length(0.0), is_original(true) {}
};

// Triangle structure for intrinsic triangulation
struct IntrinsicTriangle {
	int v[3]; // Vertex indices
	int e[3]; // Edge indices
	double edge_lengths[3]; // Intrinsic edge lengths
	bool is_active; // False if removed during flip operations

	IntrinsicTriangle() :
			is_active(true) {
		v[0] = v[1] = v[2] = -1;
		e[0] = e[1] = e[2] = -1;
		edge_lengths[0] = edge_lengths[1] = edge_lengths[2] = 0.0;
	}
};

class IntrinsicTriangulation : public RefCounted {
	GDCLASS(IntrinsicTriangulation, RefCounted);

private:
	// Mesh data
	PackedVector3Array vertices;
	PackedInt32Array indices;
	PackedVector3Array normals;

	// Intrinsic data structures
	Vector<IntrinsicEdge> edges;
	Vector<IntrinsicTriangle> triangles;
	HashMap<int, int> vertex_to_triangle; // Map vertex to one of its triangles

	// Configuration
	int max_flip_iterations = 100;
	float angle_threshold = 0.523599; // ~30 degrees (pi/6)
	bool use_delaunay_criterion = true;

	// Helper methods
	void build_initial_intrinsic_mesh();
	int find_edge(int v0, int v1) const;
	int find_opposite_vertex(int triangle_idx, int edge_idx) const;
	bool is_delaunay_edge(int edge_idx) const;
	bool can_flip_edge(int edge_idx) const;
	void flip_edge(int edge_idx);
	double compute_intrinsic_edge_length(int v0, int v1, int tri_idx) const;
	double triangle_circumradius(const IntrinsicTriangle &tri) const;
	Vector3 compute_triangle_normal(int tri_idx) const;
	void update_normals();

	// Intrinsic geometry helpers
	double compute_angle_at_vertex(const IntrinsicTriangle &tri, int local_vertex_idx) const;
	bool point_in_circumcircle(const IntrinsicTriangle &tri, int point_idx) const;

protected:
	static void _bind_methods();

public:
	// Initialize from mesh data
	void set_mesh_data(const PackedVector3Array &p_vertices, const PackedInt32Array &p_indices, const PackedVector3Array &p_normals = PackedVector3Array());
	void set_mesh(const Ref<ArrayMesh> &p_mesh, int p_surface = 0);

	// Configuration
	void set_max_flip_iterations(int p_iterations);
	int get_max_flip_iterations() const;
	void set_angle_threshold(float p_angle);
	float get_angle_threshold() const;
	void use_delaunay_flips(bool p_enable);

	// Execute intrinsic triangulation
	bool flip_to_delaunay();
	bool refine_intrinsic_triangulation(float p_target_edge_length = -1.0f);
	void smooth_intrinsic_positions(int p_iterations = 5);

	// Extract results
	PackedVector3Array get_vertices() const;
	PackedInt32Array get_indices() const;
	PackedVector3Array get_normals() const;
	Ref<ArrayMesh> get_mesh() const;

	// Query information
	int get_triangle_count() const;
	int get_edge_count() const;
	int get_vertex_count() const;
	Dictionary get_statistics() const;

	IntrinsicTriangulation();
	~IntrinsicTriangulation();
};
