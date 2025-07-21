/**************************************************************************/
/*  kusudama_bone_constraint_3d.h                                         */
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

#include "ik_open_cone_3d.h"
#include "ik_ray_3d.h"

class IKNode3D;
class IKBoneSegment3D;

class KusudamaBoneConstraint3D : public BoneConstraint3D {
	GDCLASS(KusudamaBoneConstraint3D, BoneConstraint3D);

private:
	Vector<Ref<IKLimitCone3D>> open_cones;

	bool axially_constrained = true;
	bool orientationally_constrained = true;

	real_t min_axial_angle = 0.0;
	real_t range_angle = Math::PI * 2.0;

	Quaternion twist_min_rot;
	Vector3 twist_min_vec;
	Vector3 twist_center_vec;
	Quaternion twist_center_rot;
	real_t twist_half_range_half_cos = 0.0;
	Vector3 twist_max_vec;
	Quaternion twist_max_rot;

	real_t resistance = 0.0;

	Ref<IKRay3D> bone_ray;
	Ref<IKRay3D> constrained_ray;

protected:
	static void _bind_methods();

	virtual void _process_constraint(int p_index, Skeleton3D *p_skeleton, int p_apply_bone, int p_reference_bone, float p_amount) override;

public:
	void set_open_cones(const TypedArray<IKLimitCone3D> &p_cones);
	TypedArray<IKLimitCone3D> get_open_cones() const;

	void add_open_cone(const Ref<IKLimitCone3D> &p_cone);
	void remove_open_cone(const Ref<IKLimitCone3D> &p_cone);
	void clear_open_cones();

	void set_axial_limits(real_t min_angle, real_t in_range);
	real_t get_min_axial_angle() const;
	real_t get_range_angle() const;

	bool is_axially_constrained() const;
	void disable_axial_limits();
	void enable_axial_limits();
	void toggle_axial_limits();

	bool is_orientationally_constrained() const;
	void disable_orientational_limits();
	void enable_orientational_limits();
	void toggle_orientational_limits();

	bool is_enabled() const;
	void disable();
	void enable();

	void set_resistance(float p_resistance);
	float get_resistance() const;

	Vector3 _solve(const Vector3 &p_direction) const;
	Vector3 get_local_point_in_limits(Vector3 in_point, Vector<double> *in_bounds) const;
	Vector3 local_point_on_path_sequence(Vector3 p_in_point) const;

	void update_tangent_radii();

	void snap_to_orientation_limit(Ref<IKNode3D> bone_direction, Ref<IKNode3D> to_set, Ref<IKNode3D> limiting_axes, real_t p_dampening, real_t p_cos_half_angle_dampen);
	void set_snap_to_twist_limit(Ref<IKNode3D> p_bone_direction, Ref<IKNode3D> p_to_set, Ref<IKNode3D> p_constraint_axes, real_t p_dampening, real_t p_cos_half_dampen);

	static void get_swing_twist(Quaternion p_rotation, Vector3 p_axis, Quaternion &r_swing, Quaternion &r_twist);
	static Quaternion clamp_to_quadrance_angle(Quaternion p_rotation, double p_cos_half_angle);
	static Quaternion get_quaternion_axis_angle(const Vector3 &p_axis, real_t p_angle);

	// Backwards-compatibility wrappers expected by the EWBIK editor gizmo.
	// These forward to the new Kusudama-named APIs so the plugin can use the same calls.
	bool is_swing_orientationally_constrained() const { return is_orientationally_constrained(); }
	TypedArray<IKLimitCone3D> get_swing_open_cones() const { return get_open_cones(); }

	KusudamaBoneConstraint3D();
	~KusudamaBoneConstraint3D();
};
