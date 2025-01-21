/**************************************************************************/
/*  joint_limitation_3d.h                                                 */
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

class JointLimitation3D : public Resource {
	GDCLASS(JointLimitation3D, Resource);

	struct JointLimitation3DHole {
		Vector3 axis; // Should be normalized.
		real_t radius_range = 0.1; // 0.0 = No hole, 1.0 = Hemisphere.
	};

	Vector<JointLimitation3DHole *> holes;

public:
	// TODO: coding which entry swing parameters.
	struct Swing {
		real_t foo;
		real_t bar;
	};

protected:
	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;

	static void _bind_methods();

public:
	Swing solve(const Quaternion &p_rest, const Swing &p_swing) const;

	void set_axis(int p_hole, const Vector3 &p_axis);
	Vector3 get_axis(int p_hole) const;
	void set_radius_range(int p_hole, real_t p_radius_range);
	real_t get_radius_range(int p_hole) const;

	void set_hole_count(int p_count);
	int get_hole_count() const;
	void clear_holes();

	JointLimitation3D();
	virtual ~JointLimitation3D();
};
