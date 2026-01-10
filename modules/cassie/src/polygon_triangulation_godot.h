/**************************************************************************/
/*  polygon_triangulation_godot.h                                         */
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

#include "../thirdparty/multipolygon_triangulator/DMWT.h"
#include "core/object/ref_counted.h"
#include "scene/resources/mesh.h"

class ImporterMesh;

/// @brief Godot-friendly wrapper around C++ DMWT triangulation algorithm.
///
/// PolygonTriangulationGodot provides low-level control over polygon triangulation
/// using the DMWT (Dynamic Programming Multi-Way Tiling) algorithm. It handles
/// automatic conversion between Godot types (PackedVector3Array) and C++ arrays,
/// and provides access to detailed triangulation statistics.
///
/// The DMWT algorithm optimizes polygon triangulation by:
/// 1. Building a planar Delaunay triangulation of the boundary polygon
/// 2. Using dynamic programming to select the optimal triangle configuration
/// 3. Supporting custom cost weights for triangle quality optimization
///
/// @par Algorithm Details:
/// - **Complexity**: O(n³) preprocessing, O(n²) triangulation
/// - **Output**: Optimal triangulation maximizing quality according to weights
/// - **References**: "Curve and Surface Sketching in Immersive Environments" (CHI 2021)
///
/// @par Usage Example:
/// @code
/// var boundary = PackedVector3Array([Vector3(0,0,0), Vector3(1,0,0), Vector3(1,1,0)])
/// var tri = PolygonTriangulationGodot.create(boundary)
///
/// # Configure for high-quality output
/// tri.set_cost_weights(1.0, 0.5, 0.0, 1.0, 0.0)
/// tri.set_optimization_rounds(3)
///
/// # Execute triangulation
/// if tri.preprocess():
///     if tri.triangulate():
///         var mesh = tri.get_mesh()
/// @endcode
///
/// @par Cost Weight Parameters:
/// - **triangle_cost**: Area-based triangle cost (default: 1.0)
/// - **edge_cost**: Edge length cost (default: 0.5)
/// - **bi_triangle_cost**: Two-triangle configuration cost (default: 0.0)
/// - **triangle_boundary_cost**: Boundary triangle cost (default: 1.0)
/// - **worst_dihedral_cost**: Dihedral angle penalty (default: 0.0)
///
/// @note For most applications, use CassieSurface for automatic pipeline control
///
/// @see CassieSurface for high-level surface generation
class PolygonTriangulationGodot : public RefCounted {
	GDCLASS(PolygonTriangulationGodot, RefCounted);

private:
	Ref<PolygonTriangulation> triangulator;
	PackedVector3Array cached_vertices;
	PackedInt32Array cached_indices;
	PackedVector3Array cached_normals;
	bool has_cached_result = false;

protected:
	static void _bind_methods();

public:
	// Factory methods
	static Ref<PolygonTriangulationGodot> create(const PackedVector3Array &p_points, const PackedVector3Array &p_normals = PackedVector3Array());
	static Ref<PolygonTriangulationGodot> create_planar(const PackedVector3Array &p_points, const PackedVector3Array &p_degenerate_points);

	// Configuration
	void set_cost_weights(float p_triangle, float p_edge, float p_bi_triangle, float p_triangle_boundary, float p_worst_dihedral);
	void set_optimization_rounds(int p_rounds);
	void set_point_limit(int p_limit);
	void enable_dot_output(bool p_enable);

	// Execute triangulation
	bool preprocess();
	bool triangulate();
	void clear_cache();

	// Extract results as Godot types
	PackedVector3Array get_vertices() const;
	PackedInt32Array get_indices() const;
	PackedVector3Array get_normals() const;
	Ref<ArrayMesh> get_mesh(bool p_smooth = false, int p_subdivisions = 0, int p_laplacian_iterations = 0) const;
	Ref<ImporterMesh> get_importer_mesh(bool p_smooth = false, int p_subdivisions = 0, int p_laplacian_iterations = 0) const;

	// Query information
	int get_triangle_count() const;
	int get_vertex_count() const;
	Dictionary get_statistics() const;
	float get_optimal_cost() const;

	PolygonTriangulationGodot();
	~PolygonTriangulationGodot();
};
