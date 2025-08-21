/**************************************************************************/
/*  iterate_ik_3d.h                                                       */
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

#include "scene/3d/chain_ik_3d.h"

class IterateIK3D : public ChainIK3D {
	GDCLASS(IterateIK3D, ChainIK3D);

public:
	struct IterateIK3DSetting : public ChainIK3DSetting {
		NodePath pole_node;
		NodePath target_node;
		int joint_size_half = -1;
		int chain_size_half = -1;
	};

protected:
	Vector<IterateIK3DSetting *> iterate_settings; // For caching.

	int max_iterations = 4;
	real_t min_distance = 0.01; // If distance between end joint and target is less than min_distance, finish iteration.
	real_t min_distance_squared = min_distance * min_distance; // For cache.
	real_t angular_delta_limit = Math::deg_to_rad(2.0); // If the delta is too large, the results before and after iterating can change significantly, and divergence of calculations can easily occur.

	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _validate_dynamic_prop(PropertyInfo &p_property) const;

	static void _bind_methods();

	virtual void _post_init_joints(int p_index) override;

	virtual void _process_ik(Skeleton3D *p_skeleton, double p_delta) override;
	void _process_joints(double p_delta, Skeleton3D *p_skeleton, IterateIK3DSetting *p_setting, Vector<ChainIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Vector3 &p_target_destination, const Vector3 &p_pole_destination, bool p_use_pole);
	virtual void _solve_iteration_with_pole(double p_delta, Skeleton3D *p_skeleton, IterateIK3DSetting *p_setting, Vector<ChainIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Vector3 &p_destination, int p_joint_size, int p_chain_size, const Vector3 &p_pole_destination, int p_joint_size_half, int p_chain_size_half);
	virtual void _solve_iteration(double p_delta, Skeleton3D *p_skeleton, IterateIK3DSetting *p_setting, Vector<ChainIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Vector3 &p_destination, int p_joint_size, int p_chain_size);

public:
	virtual void set_setting_count(int p_count) override {
		_set_setting_count<IterateIK3DSetting>(p_count);
		iterate_settings = _cast_settings<IterateIK3DSetting>();
		chain_settings = _cast_settings<ChainIK3DSetting>(); // Don't forget to sync super class settings.
	}
	virtual void clear_settings() override {
		_set_setting_count<IterateIK3DSetting>(0);
		iterate_settings.clear();
		chain_settings.clear(); // Don't forget to sync super class settings.
	}

	void set_max_iterations(int p_max_iterations);
	int get_max_iterations() const;
	void set_min_distance(real_t p_min_distance);
	real_t get_min_distance() const;
	void set_angular_delta_limit(real_t p_angular_delta_limit);
	real_t get_angular_delta_limit() const;

	// Setting.
	void set_target_node(int p_index, const NodePath &p_target_node);
	NodePath get_target_node(int p_index) const;
	void set_pole_node(int p_index, const NodePath &p_pole_node);
	NodePath get_pole_node(int p_index) const;

	~IterateIK3D();
};
