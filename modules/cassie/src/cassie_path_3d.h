/**************************************************************************/
/*  cassie_path_3d.h                                                      */
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

#include "core/io/resource.h"

/// @brief Represents a 3D curve path for sketch-based surface modeling.
///
/// CassiePath3D stores and manipulates 3D curve data for sketch-based surface
/// generation. It provides tools for curve beautification, resampling, and
/// normal smoothing using mathematically robust algorithms.
///
/// This class is designed for use in immersive sketching workflows where users
/// draw 3D curves that define surface boundaries or control curves.
///
/// @par Supported Algorithms:
/// - **Laplacian Smoothing**: Simple local averaging smoothing
/// - **Taubin Smoothing**: Two-pass filter that prevents volume shrinkage
/// - **Uniform Arc-Length Resampling**: Evenly spaces points along curve length
/// - **Normal Smoothing**: Averages normals with neighbors
///
/// @par Usage Example:
/// @code
/// var path = CassiePath3D.new()
/// path.add_point(Vector3(0, 0, 0), Vector3(0, 1, 0))
/// path.add_point(Vector3(1, 0.5, 0), Vector3(0, 1, 0))
/// path.add_point(Vector3(2, 0, 0), Vector3(0, 1, 0))
///
/// # Beautify using Taubin smoothing (0.5 = lambda, -0.53 = mu, 5 = iterations)
/// path.beautify_taubin(0.5, -0.53, 5)
///
/// # Resample to uniform spacing (20 points total)
/// path.resample_uniform(20)
///
/// # Get processed boundary for triangulation
/// var boundary = path.get_points()
/// @endcode
///
/// @par Taubin Smoothing Parameters:
/// - **lambda (λ)**: Smoothing strength, typically 0.5-0.6
/// - **mu (μ)**: Anti-shrinkage compensation, typically -λ/(1+λ) ≈ -0.53
///
/// @note The path can be marked as closed to form a loop, affecting smoothing
/// behavior at boundaries
///
/// @see CassieSurface for high-level surface generation
class CassiePath3D : public Resource {
	GDCLASS(CassiePath3D, Resource);

private:
	PackedVector3Array points;
	PackedVector3Array normals;
	bool is_closed = false;

protected:
	static void _bind_methods();

public:
	// Point management
	void add_point(const Vector3 &p_position, const Vector3 &p_normal = Vector3(0, 1, 0));
	void insert_point(int p_index, const Vector3 &p_position, const Vector3 &p_normal = Vector3(0, 1, 0));
	void remove_point(int p_index);
	void set_point_position(int p_index, const Vector3 &p_position);
	void set_point_normal(int p_index, const Vector3 &p_normal);
	Vector3 get_point_position(int p_index) const;
	Vector3 get_point_normal(int p_index) const;
	int get_point_count() const;
	void clear_points();

	// Path properties
	void set_closed(bool p_closed);
	bool is_path_closed() const;

	// Curve processing
	void beautify_laplacian(float p_lambda = 0.5f, int p_iterations = 5);
	void beautify_taubin(float p_lambda = 0.5f, float p_mu = -0.53f, int p_iterations = 5);
	void resample_uniform(int p_target_count);
	void smooth_normals();

	// Sampling
	PackedVector3Array get_sample_points(int p_count) const;
	PackedVector3Array get_sample_normals(int p_count) const;
	PackedVector3Array get_points() const { return points; }
	PackedVector3Array get_normals() const { return normals; }

	// Analysis
	float get_total_length() const;
	float get_average_segment_length() const;

	CassiePath3D();
	~CassiePath3D();
};
