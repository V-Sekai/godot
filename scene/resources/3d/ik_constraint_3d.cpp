/**************************************************************************/
/*  ik_constraint_3d.cpp                                                  */
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

#include "ik_constraint_3d.h"

bool IKConstraint3D::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("holes/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, holes.size(), false);

		if (what == "axis") {
			set_axis(which, p_value);
		} else if (what == "radius_range") {
			set_radius_range(which, p_value);
		} else {
			return false;
		}
	}
	return true;
}

bool IKConstraint3D::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("holes/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, holes.size(), false);

		if (what == "axis") {
			r_ret = get_axis(which);
		} else if (what == "radius_range") {
			r_ret = get_radius_range(which);
		} else {
			return false;
		}
	}
	return true;
}

void IKConstraint3D::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < holes.size(); i++) {
		String path = "holes/" + itos(i) + "/";
		p_list->push_back(PropertyInfo(Variant::VECTOR3, path + "axis"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, path + "radius_range", PROPERTY_HINT_RANGE, "0,1,0.01"));
	}
}

void IKConstraint3D::set_axis(int p_hole, const Vector3 &p_axis) {
	ERR_FAIL_INDEX(p_hole, holes.size());
	ERR_FAIL_COND(p_axis.is_zero_approx());
	holes[p_hole]->axis = p_axis;
}

Vector3 IKConstraint3D::get_axis(int p_hole) const {
	ERR_FAIL_INDEX_V(p_hole, holes.size(), Vector3());
	return holes[p_hole]->axis;
}

void IKConstraint3D::set_radius_range(int p_hole, real_t p_range) {
	ERR_FAIL_INDEX(p_hole, holes.size());
	holes[p_hole]->radius_range = p_range;
}

real_t IKConstraint3D::get_radius_range(int p_hole) const {
	ERR_FAIL_INDEX_V(p_hole, holes.size(), 0);
	return holes[p_hole]->radius_range;
}

void IKConstraint3D::set_hole_count(int p_count) {
	ERR_FAIL_COND(p_count < 0);
	int delta = p_count - holes.size() + 1;
	holes.resize(p_count);
	if (delta > 1) {
		for (int i = 1; i < delta; i++) {
			holes.write[p_count - i] = memnew(IKConstraint3DHole);
		}
	}
	notify_property_list_changed();
}

int IKConstraint3D::get_hole_count() const {
	return holes.size();
}

void IKConstraint3D::clear_holes() {
	set_hole_count(0);
}

void IKConstraint3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_axis", "hole", "axis"), &IKConstraint3D::set_axis);
	ClassDB::bind_method(D_METHOD("get_axis", "hole"), &IKConstraint3D::get_axis);
	ClassDB::bind_method(D_METHOD("set_radius_range", "hole", "range"), &IKConstraint3D::set_radius_range);
	ClassDB::bind_method(D_METHOD("get_radius_range", "hole"), &IKConstraint3D::get_radius_range);

	ClassDB::bind_method(D_METHOD("set_hole_count", "count"), &IKConstraint3D::set_hole_count);
	ClassDB::bind_method(D_METHOD("get_hole_count"), &IKConstraint3D::get_hole_count);
	ClassDB::bind_method(D_METHOD("clear_holes"), &IKConstraint3D::clear_holes);

	ADD_ARRAY_COUNT("Holes", "hole_count", "set_hole_count", "get_hole_count", "holes/");
}

IKConstraint3D::Swing IKConstraint3D::solve(const Quaternion &p_rest, const Swing &p_swing) const {
	// TODO: Write constraint here.
	return Swing();
}

IKConstraint3D::IKConstraint3D() {
	//
}

IKConstraint3D::~IKConstraint3D() {
	//
}
