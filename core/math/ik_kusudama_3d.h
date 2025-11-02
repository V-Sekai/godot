#pragma once

#include "core/io/resource.h"
#include "core/math/quaternion.h"
#include "core/object/ref_counted.h"
#include "core/variant/typed_array.h"

class IKNode3D;

class IKRay3D : public RefCounted {
	GDCLASS(IKRay3D, RefCounted);

	Vector3 point_1;
	Vector3 point_2;

protected:
	static void _bind_methods();

public:
	IKRay3D();
	IKRay3D(Vector3 p_point_one, Vector3 p_point_two);
	Vector3 get_heading();
	void set_heading(const Vector3 &p_new_head);
	real_t get_scaled_projection(const Vector3 p_input);
	void elongate(real_t p_amount);
	Vector3 get_intersects_plane(Vector3 p_vertex_a, Vector3 p_vertex_b, Vector3 p_vertex_c);
	int intersects_sphere(Vector3 p_sphere_center, real_t p_radius, Vector3 *r_first_intersection, Vector3 *r_second_intersection);
	void set_point_1(Vector3 p_point);
	void set_point_2(Vector3 p_point);
	Vector3 get_point_2();
	Vector3 get_point_1();
};

class IKKusudama3D;
class IKLimitCone3D : public Resource {
	GDCLASS(IKLimitCone3D, Resource);

	Vector3 control_point = Vector3(0, 1, 0);
	double radius = 0;
	double radius_cosine = 0;
	WeakRef parent_kusudama;

public:
	IKLimitCone3D() {}
	void set_attached_to(Ref<IKKusudama3D> p_attached_to);
	Ref<IKKusudama3D> get_attached_to();
	void update_tangent_handles(Ref<IKLimitCone3D> p_next);
	Vector3 get_on_great_tangent_triangle(Ref<IKLimitCone3D> next, Vector3 input) const;
	Vector3 closest_to_cone(Vector3 input, Vector<double> *in_bounds) const;
	Vector3 get_closest_path_point(Ref<IKLimitCone3D> next, Vector3 input) const;
	Vector3 get_control_point() const;
	void set_control_point(Vector3 p_control_point);
	double get_radius() const;
	void set_radius(double radius);
	static Vector3 get_orthogonal(Vector3 p_input);
};

class IKKusudama3D : public Resource {
	GDCLASS(IKKusudama3D, Resource);

	Vector<Ref<IKLimitCone3D>> open_cones;
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
	Ref<IKRay3D> bone_ray = Ref<IKRay3D>(memnew(IKRay3D()));
	Ref<IKRay3D> constrained_ray = Ref<IKRay3D>(memnew(IKRay3D()));
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
	void add_open_cone(Ref<IKLimitCone3D> p_open_cone);
	void remove_open_cone(Ref<IKLimitCone3D> limitCone);
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
