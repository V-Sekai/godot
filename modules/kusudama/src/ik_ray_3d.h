/**************************************************************************/
/*  ik_ray_3d.h                                                           */
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

#include "core/math/math_defs.h"
#include "core/math/vector3.h"
#include <stdint.h>

struct IKRay3D {
	Vector3 tta, ttb, ttc;
	Vector3 I, u, v, n, dir, w0;
	Vector3 m, at, bt, ct, pt;
	Vector3 bc, ca, ac;

	Vector3 point_1;
	Vector3 point_2;
	Vector3 working_vector;

	IKRay3D();
	IKRay3D(Vector3 p_point_one, Vector3 p_point_two);
	~IKRay3D() {}

	Vector3 get_heading();
	void set_heading(const Vector3 &p_new_head);

	/**
	 * Returns the scalar projection of the input vector on this
	 * ray. See original documentation in previous class.
	 */
	real_t get_scaled_projection(const Vector3 p_input);

	/**
	 * adds the specified length to the ray in both directions.
	 */
	void elongate(real_t p_amount);

	/**
	 * @param ta the first vertex of a triangle on the plane
	 * @param tb the second vertex of a triangle on the plane
	 * @param tc the third vertex of a triangle on the plane
	 * @return the point where this ray intersects the plane specified by the
	 *         triangle ta,tb,tc.
	 */
	Vector3 get_intersects_plane(Vector3 p_vertex_a, Vector3 p_vertex_b, Vector3 p_vertex_c);

	/*
	 * Find where this ray intersects a sphere
	 */
	int intersects_sphere(Vector3 p_sphere_center, real_t p_radius, Vector3 *r_first_intersection, Vector3 *r_second_intersection);

	void set_point_1(Vector3 p_point);
	void set_point_2(Vector3 p_point);
	Vector3 get_point_2();
	Vector3 get_point_1();

	int intersects_sphere(Vector3 p_rp1, Vector3 p_rp2, real_t p_radius, Vector3 *r_first_intersection, Vector3 *r_second_intersection);
	real_t triangle_area_2d(real_t p_x1, real_t p_y1, real_t p_x2, real_t p_y2, real_t p_x3, real_t p_y3);
	void barycentric(Vector3 p_a, Vector3 p_b, Vector3 p_c, Vector3 p_p, Vector3 *r_uvw);
	Vector3 plane_intersect_test(Vector3 p_vertex_a, Vector3 p_vertex_b, Vector3 p_vertex_c, Vector3 *uvw);

	operator String() const {
		return String(L"(") + point_1.x + L" ->  " + point_2.x + L") \n " + L"(" + point_1.y + L" ->  " + point_2.y + L") \n " + L"(" + point_1.z + L" ->  " + point_2.z + L") \n ";
	}
};
