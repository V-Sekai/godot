/**************************************************************************/
/*  joint_limitation_kusudama_3d.h                                        */
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

#include "scene/resources/3d/joint_limitation_3d.h"
#include "core/math/projection.h"

class JointLimitationKusudama3D : public JointLimitation3D {
	GDCLASS(JointLimitationKusudama3D, JointLimitation3D);

	// Cones data: Each Projection is a 4x4 matrix representing a quad [cone1, tan1, tan2, cone2]
	// Each column is Vector4(x, y, z, radius) for that element
	// Column 0 = cone1, Column 1 = tan1, Column 2 = tan2, Column 3 = cone2
	Vector<Projection> cones;

	bool orientationally_constrained = true;

protected:
	static void _bind_methods();
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	virtual Vector3 _solve(const Vector3 &p_direction) const override;

public:
	void set_cone_count(int p_count);
	int get_cone_count() const;

	void set_cone_center(int p_index, const Vector3 &p_center);
	Vector3 get_cone_center(int p_index) const;

	void set_cone_radius(int p_index, real_t p_radius);
	real_t get_cone_radius(int p_index) const;

	void set_orientationally_constrained(bool p_constrained);
	bool is_orientationally_constrained() const;

#ifdef TOOLS_ENABLED
	virtual void draw_shape(Ref<SurfaceTool> &p_surface_tool, const Transform3D &p_transform, float p_bone_length, const Color &p_color) const override;
#endif // TOOLS_ENABLED
};
