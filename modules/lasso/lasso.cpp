/**************************************************************************/
/*  lasso.cpp                                                             */
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

#include "lasso.h"
#include <core/math/math_defs.h>
#include <servers/xr_server.h>

LassoPoint::LassoPoint(){};

LassoPoint::~LassoPoint() {
	unregister_point();
}

void LassoPoint::_bind_methods() {
	ClassDB::bind_method(D_METHOD("enable_snapping", "on"),
			&LassoPoint::enable_snapping);
	ClassDB::bind_method(D_METHOD("get_snapping_enabled"),
			&LassoPoint::get_snapping_enabled);
	ClassDB::bind_method(D_METHOD("set_snap_locked", "p_enable"),
			&LassoPoint::set_snap_locked);
	ClassDB::bind_method(D_METHOD("get_snap_locked"),
			&LassoPoint::get_snap_locked);
	ClassDB::bind_method(D_METHOD("set_size", "p_size"), &LassoPoint::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &LassoPoint::get_size);
	ClassDB::bind_method(D_METHOD("set_snapping_power", "p_snapping_power"),
			&LassoPoint::set_snapping_power);
	ClassDB::bind_method(D_METHOD("get_snapping_power"),
			&LassoPoint::get_snapping_power);
	ClassDB::bind_method(D_METHOD("get_snap_score"), &LassoPoint::get_snap_score);
	ClassDB::bind_method(D_METHOD("get_origin"), &LassoPoint::get_origin);
	ClassDB::bind_method(D_METHOD("register_point", "p_database", "p_origin"),
			&LassoPoint::register_point);
	ClassDB::bind_method(D_METHOD("unregister_point"),
			&LassoPoint::unregister_point);
}

void LassoPoint::set_snap_locked(bool p_enable) {
	snap_locked = p_enable;
}

bool LassoPoint::get_snap_locked() {
	return snap_locked;
}

void LassoPoint::register_point(Ref<LassoDB> p_database, Node *p_origin) {
	database = p_database;
	origin = cast_to<Node3D>(p_origin);
	ERR_FAIL_NULL(origin);
	ERR_FAIL_COND(database.is_null());
	database->add_point(this);
};

void LassoPoint::unregister_point() {
	if (database.is_valid()) {
		database->remove_point(this);
	}
	database.unref();
}

float LassoPoint::get_snap_score() {
	return last_snap_score;
}

void LassoPoint::set_snap_score(float score) {
	last_snap_score = score;
}

Vector3 LassoPoint::get_origin_pos() {
	if (origin != nullptr) {
		return origin->get_global_transform().origin;
	}
	return Vector3();
}

bool LassoPoint::valid_origin() {
	return origin != nullptr && origin->is_visible_in_tree() && snapping_enabled;
}

bool LassoPoint::matching_origin(Node *p_origin) {
	return origin == p_origin;
}

Node *LassoPoint::get_origin() {
	return origin;
}
void LassoPoint::enable_snapping(bool on) {
	snapping_enabled = on;
}
bool LassoPoint::get_snapping_enabled() {
	return snapping_enabled;
}
void LassoPoint::set_size(float p_size) {
	size = p_size;
}
float LassoPoint::get_size() {
	return size;
}
void LassoPoint::set_snapping_power(float p_snapping_power) {
	snapping_power = p_snapping_power;
}
float LassoPoint::get_snapping_power() {
	return snapping_power;
}

LassoDB::LassoDB() {}

LassoDB::~LassoDB() {}

void LassoDB::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_point", "point"), &LassoDB::add_point);
	ClassDB::bind_method(D_METHOD("remove_point", "point"),
			&LassoDB::remove_point);
	ClassDB::bind_method(D_METHOD("calc_top_two_snapping_power", "source",
								 "current_snap",
								 "snap_max_power_increase",
								 "snap_increase_amount", "snap_lock"),
			&LassoDB::calc_top_two_snapping_power);
	ClassDB::bind_method(D_METHOD("calc_top_redirecting_power", "snapped_point",
								 "viewpoint", "redirection_direction"),
			&LassoDB::calc_top_redirecting_power);
}
void LassoDB::add_point(Ref<LassoPoint> point) {
	if (point.is_valid() && !points.has(point)) {
		points.append(point);
	}
}

void LassoDB::remove_point(Ref<LassoPoint> point) {
	int index = points.find(point);
	if (index >= 0) {
		points.remove_at(index);
	}
}

Array LassoDB::calc_top_two_snapping_power(Transform3D source, Node *current_snap,
		float snap_max_power_increase,
		float snap_increase_amount,
		bool snap_lock) {
	Array output;
	Ref<LassoPoint> first;
	Ref<LassoPoint> second;
	for (int i = 0; i < points.size(); i++) {
		Ref<LassoPoint> next = points[i];
		if (next.is_valid() && next->valid_origin() && next->snapping_enabled) {
			Vector3 point_local = source.xform_inv(next->get_origin_pos());
			float euclidian_dist = point_local.length();
			float angular_dist = point_local.angle_to(Vector3(0, 0, -1));
			float rejection_length =
					Vector3(point_local[0], point_local[1], 0).length();

			float snapping_power = 0;
			if (rejection_length <= next->size) {
				snapping_power = next->snapping_power / (1.0 + euclidian_dist) /
								 (0.01 + angular_dist);
			} else {
				snapping_power = next->snapping_power / (1.0 + euclidian_dist) /
								 (0.1 + angular_dist);
			}

			if (next->matching_origin(current_snap)) {
				snapping_power += snapping_power * pow(snap_increase_amount, 2) *
								  snap_max_power_increase;
				if (next->snap_locked && snap_lock) {
					next->set_snap_score(snapping_power);
					first = next;
					second.unref();
					break;
				}
			}
			next->set_snap_score(snapping_power);

			if (first.is_null() || first->get_snap_score() < next->get_snap_score()) {
				second = first;
				first = next;
			} else if (second.is_null() ||
					   second->get_snap_score() < next->get_snap_score()) {
				second = next;
			}
		}
	}
	output.push_back(first);
	output.push_back(second);
	return output;
}

Node *LassoDB::calc_top_redirecting_power(Node *snapped_origin,
		Transform3D viewpoint,
		Vector2 redirection_direction) {
	Node3D *snapped_origin_Node3D = cast_to<Node3D>(snapped_origin);
	ERR_FAIL_NULL_V(snapped_origin_Node3D, nullptr);

	Node *output = snapped_origin_Node3D;

	if (redirection_direction.length_squared() > 0) {
		// Caclculate the basis.
		Vector3 snapped_origin_position =
				snapped_origin_Node3D->get_global_transform().origin;
		Vector3 snapped_vector = viewpoint.origin - snapped_origin_position;
		Vector3 z_vector = snapped_vector.normalized();
		Vector3 up_vector = viewpoint.get_basis().get_column(1).normalized();
		Vector3 x_vector = z_vector.cross(up_vector).normalized();
		Vector3 y_vector = x_vector.cross(z_vector).normalized();
		Basis local_basis =
				Basis(x_vector, y_vector,
						z_vector); // ITS FUCKING TRANSPOSED BY DEFAULT. WHY!?
		Ref<LassoPoint> first;
		float redirect_power = INFINITY; // The lower is better.
		for (int i = 0; i < points.size(); i++) {
			Ref<LassoPoint> next = points[i];
			float next_power = 0;
			if (next.is_valid() && next->valid_origin() &&
					!next->matching_origin(snapped_origin_Node3D)) {
				Vector3 point_vector = viewpoint.origin - next->get_origin_pos();
				if (point_vector.angle_to(snapped_vector) < Math_PI / 4.0) {
					Vector3 point_xyz = local_basis.xform(point_vector);
					Vector2 point_xy = Vector2(point_xyz[0], -point_xyz[1]);

					if (Math::abs(redirection_direction.angle_to(point_xy)) >=
							Math_PI / 2) {
						continue;
						// Keep the redirect power at infinity if the joystick is more than
						// 90 degrees away from the point.
					} else if (redirection_direction[0] == 0 && point_xy[1] != 0) {
						// If you moved your joystick perfectly vertically calculating the
						// intersection of two lines breaks because of the y = mx + b
						// notation so instead let's calculate the y intercept of the line.
						float bisecting_slope =
								-point_xy[0] /
								point_xy[1]; // Rotated 90 because it's the bisecting line.
						float bisecting_y =
								point_xy[1] / 2; // Y value when the bisecting line intersects
						// the line to the point.
						next_power = pow(
								bisecting_slope * -(point_xy[0] / 2) + bisecting_y,
								2); // Squared just because. we're not actually calculating dist.
					} else if (point_xy[1] == 0) {
						// Point is on the x-axis which means the bisecting line would be
						// vertical and also undefined we calculate.
						float bisecting_x = point_xy[0] / 2;
						float slope = redirection_direction[1] /
									  redirection_direction[0]; // rise over run
						float intersect_x = bisecting_x;
						float intersect_y = bisecting_x * slope;
						next_power = pow(intersect_x, 2) +
									 pow(intersect_y, 2); // squared euclidian distance
					} else {
						// This is the most common case
						// equation taken from the internet.
						float a1 = -point_xy[0] / point_xy[1];
						float c1 = (1 - a1) * point_xy[0] / 2;
						float a2 = redirection_direction[1] / redirection_direction[0];
						float x_component = c1 / (a2 - a1);
						float y_component = (a2 * c1) / (a2 - a1);
						next_power = pow(x_component, 2) + pow(y_component, 2);
					}
					if (next_power < redirect_power) {
						redirect_power = next_power;
						first = next;
					}
				}
			}
		}
		if (first.is_valid() && first->valid_origin()) {
			output = first->get_origin();
		}
	}
	return output;
}
