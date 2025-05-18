/**************************************************************************/
/*  joint_limitation_cone_3d.cpp                                          */
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

#include "joint_limitation_cone_3d.h"


void JointLimitationCone3D::set_radius_range(real_t p_radius_range) {
	radius_range = p_radius_range;
}

real_t JointLimitationCone3D::get_radius_range() const {
	return radius_range;
}

void JointLimitationCone3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius_range", "radius_range"), &JointLimitationCone3D::set_radius_range);
	ClassDB::bind_method(D_METHOD("get_radius_range"), &JointLimitationCone3D::get_radius_range);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius_range", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_radius_range", "get_radius_range");
}

Vector3 JointLimitationCone3D::_solve(const Vector3 &p_direction) const {
	// Assume the central (forward of the cone) axis is the +Y.
	// This is based on the coordinate system set by JointLimitation3D::_make_space().
	Vector3 center_axis = Vector3(0, 1, 0);
	
	// Apply the limitation if the angle exceeds radius_range * PI.
	real_t angle = p_direction.angle_to(center_axis);
	real_t max_angle = radius_range * Math::PI;
	
	if (angle <= max_angle) {
		// If within the limitation range, return the new direction as is.
		return p_direction;
	} else {
		// If outside the limitation range, calculate the closest direction within the range.
		// Define a plane using the central axis and the new direction vector.
		Vector3 plane_normal;
		
		// Special handling for when the new direction vector is completely opposite to the central axis.
		if (Math::is_equal_approx((double)angle, Math::PI)) {
			// Select an arbitrary perpendicular axis
			plane_normal = center_axis.get_any_perpendicular();
		} else {
			plane_normal = center_axis.cross(p_direction).normalized();
		}
		
		// Calculate a vector rotated by the maximum angle from the central axis on the plane.
		Quaternion rotation = Quaternion(plane_normal, max_angle);
		Vector3 limited_dir = rotation.xform(center_axis);
		
		// Return the vector within the limitation range that is closest to p_direction.
		// This preserves the directionality of p_direction as much as possible.
		Vector3 projection = p_direction - center_axis * p_direction.dot(center_axis);
		if (projection.length_squared() > CMP_EPSILON) {
			Vector3 side_dir = projection.normalized();
			Quaternion side_rotation = Quaternion(center_axis.cross(side_dir).normalized(), max_angle);
			limited_dir = side_rotation.xform(center_axis);
		}
		
		return limited_dir.normalized();
	}
}
