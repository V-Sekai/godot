/**************************************************************************/
/*  renik_helper.cpp                                                      */
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

#include "renik_helper.h"

float RenIKHelper::safe_acos(float f) {
	if (f > 1) {
		f = 1;
	} else if (f < -1) {
		f = -1;
	}
	return acos(f);
}
float RenIKHelper::safe_asin(float f) {
	if (f > 1) {
		f = 1;
	} else if (f < -1) {
		f = -1;
	}
	return asin(f);
}

Vector3 RenIKHelper::get_perpendicular_vector(Vector3 v) {
	Vector3 perpendicular;
	if (v[0] != 0 && v[1] != 0) {
		perpendicular = Vector3(0, 0, 1).cross(v).normalized();
	} else {
		perpendicular = Vector3(1, 0, 0);
	}
	return perpendicular;
}

Vector3 RenIKHelper::vector_rejection(Vector3 v, Vector3 normal) {
	if (v.length_squared() == 0 || normal.length_squared() == 0) {
		return Vector3();
	}
	float normalLength = normal.length();
	Vector3 proj = (normal.dot(v) / normalLength) * (normal / normalLength);
	return v - proj;
}

Quaternion RenIKHelper::align_vectors(Vector3 a, Vector3 b, float influence) {
	if (a.length_squared() == 0 || b.length_squared() == 0) {
		return Quaternion();
	}
	a.normalize();
	b.normalize();
	if (a.length_squared() != 0 && b.length_squared() != 0) {
		// Find the axis perpendicular to both vectors and rotate along it by the
		// angular difference
		Vector3 perpendicular = a.cross(b);
		float angleDiff = a.angle_to(b) * influence;
		if (perpendicular.length_squared() == 0) {
			perpendicular = get_perpendicular_vector(a);
		}
		return Quaternion(perpendicular.normalized().normalized(), angleDiff)
				.normalized(); // lmao look at this double normalization bullshit
	} else {
		return Quaternion();
	}
}

float RenIKHelper::smoothCurve(float number, float modifier) {
	return number / (fabsf(number) + modifier);
}

Basis RenIKHelper::log_clamp(Basis basis, Basis target, float looseness) {
	return Basis(log_clamp(basis.get_rotation_quaternion(),
			target.get_rotation_quaternion(), looseness));
}

Quaternion RenIKHelper::log_clamp(Quaternion quat, Quaternion target,
		float looseness) {
	quat.x = log_clamp(quat.x, target.x, looseness);
	quat.y = log_clamp(quat.y, target.y, looseness);
	quat.z = log_clamp(quat.z, target.z, looseness);
	quat.w = log_clamp(quat.w, target.w, looseness);
	quat.normalize();
	return quat;
}
Vector3 RenIKHelper::log_clamp(Vector3 vector, Vector3 target,
		float looseness) {
	vector.x = log_clamp(vector.x, target.x, looseness);
	vector.y = log_clamp(vector.y, target.y, looseness);
	vector.z = log_clamp(vector.z, target.z, looseness);
	return vector;
}
float RenIKHelper::log_clamp(float value, float target, float looseness) {
	float difference = value - target;
	float effectiveLooseness = difference >= 0 ? looseness : looseness * -1;
	return target +
			effectiveLooseness * log(1 + (difference / effectiveLooseness));
}
