/**************************************************************************/
/*  ik_kusudama_3d.h                                                      */
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
#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include "core/math/quaternion.h"
#include "core/math/vector3.h"
#include "core/object/ref_counted.h"
#include "core/variant/typed_array.h"

class IKNode3D;

struct IKLimitCone3D {
	Vector3 control_point = Vector3(0, 1, 0);
	double radius = 0;
	double radius_cosine = 0;

	// Tangent circle data (precalculated)
	Vector3 tangent_circle_center_next_1;
	Vector3 tangent_circle_center_next_2;
	double tangent_circle_radius_next = 0;
	double tangent_circle_radius_next_cos = 0;

	IKLimitCone3D() {}
	void update_tangent_handles(IKLimitCone3D *p_next);
	Vector3 get_on_great_tangent_triangle(const IKLimitCone3D *next, Vector3 input) const;
	Vector3 closest_to_cone(Vector3 input, Vector<double> *in_bounds) const;
	Vector3 get_closest_path_point(const IKLimitCone3D *next, Vector3 input) const;
	Vector3 get_control_point() const { return control_point; }
	void set_control_point(Vector3 p_control_point);
	double get_radius() const { return radius; }
	void set_radius(double p_radius);
	static Vector3 get_orthogonal(Vector3 p_input);

	// Getters for tangent circle data
	Vector3 get_tangent_circle_center_next_1() const { return tangent_circle_center_next_1; }
	void set_tangent_circle_center_next_1(Vector3 p_center) { tangent_circle_center_next_1 = p_center; }
	Vector3 get_tangent_circle_center_next_2() const { return tangent_circle_center_next_2; }
	void set_tangent_circle_center_next_2(Vector3 p_center) { tangent_circle_center_next_2 = p_center; }
	double get_tangent_circle_radius_next() const { return tangent_circle_radius_next; }
	void set_tangent_circle_radius_next(double p_radius) {
		tangent_circle_radius_next = p_radius;
		tangent_circle_radius_next_cos = Math::cos(p_radius);
	}
	double _get_tangent_circle_radius_next_cos() const { return tangent_circle_radius_next_cos; }
};

class IKKusudama3D : public Resource {
	GDCLASS(IKKusudama3D, Resource);

	Vector<IKLimitCone3D> open_cones;
	Quaternion twist_min_rot;
	Vector3 twist_min_vec;
	Vector3 twist_max_vec;
	Vector3 twist_center_vec;
	Quaternion twist_center_rot;
	Quaternion twist_max_rot;
	real_t twist_half_range_half_cos = 0;
	Vector3 twist_tan;
	bool flipped_bounds = false;
	real_t resistance = 0;
	real_t min_axial_angle = 0.0;
	real_t range_angle = Math::TAU;
	bool orientationally_constrained = false;
	bool axially_constrained = false;

protected:
	static void _bind_methods();
	virtual Vector3 _solve(const Vector3 &p_direction) const;

public:
	~IKKusudama3D() {}
	IKKusudama3D() {}

	void _update_constraint(Ref<IKNode3D> p_limiting_axes);
	void update_tangent_radii();
	double unit_hyper_area = 2 * Math::pow(Math::PI, 2);
	double unit_area = 4 * Math::PI;

	static void get_swing_twist(Quaternion p_rotation, Vector3 p_axis, Quaternion &r_swing, Quaternion &r_twist);
	static Quaternion get_quaternion_axis_angle(const Vector3 &p_axis, real_t p_angle);
	void snap_to_orientation_limit(Ref<IKNode3D> p_bone_direction, Ref<IKNode3D> p_to_set, Ref<IKNode3D> p_limiting_axes, real_t p_dampening, real_t p_cos_half_angle_dampen);
	bool is_nan_vector(const Vector3 &vec);
	void set_axial_limits(real_t p_min_angle, real_t p_in_range);
	void set_snap_to_twist_limit(Ref<IKNode3D> p_bone_direction, Ref<IKNode3D> p_to_set, Ref<IKNode3D> p_limiting_axes, real_t p_dampening, real_t p_cos_half_dampen);
	Vector3 get_local_point_in_limits(Vector3 in_point, Vector<double> *in_bounds);
	Vector3 local_point_on_path_sequence(Vector3 in_point, Ref<IKNode3D> limiting_axes);
	void add_open_cone(const IKLimitCone3D &p_open_cone);
	void remove_open_cone(const IKLimitCone3D &limitCone);
	real_t get_min_axial_angle();
	real_t get_range_angle();
	bool is_axially_constrained();
	bool is_orientationally_constrained() const;
	void disable_orientational_limits();
	void enable_orientational_limits();
	void toggle_orientational_limits();
	void disable_axial_limits();
	void enable_axial_limits();
	void toggle_axial_limits();
	bool is_enabled() const;
	void disable();
	void enable();
	void clear_open_cones();
	TypedArray<IKLimitCone3D> get_open_cones() const;
	void set_open_cones(TypedArray<IKLimitCone3D> p_cones);
	float get_resistance();
	void set_resistance(float p_resistance);
	static Quaternion clamp_to_quadrance_angle(Quaternion p_rotation, double p_cos_half_angle);
};
