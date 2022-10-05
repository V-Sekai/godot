/**************************************************************************/
/*  vr_hand.cpp                                                           */
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

#include "vr_hand.h"
#include "vr_collision.h"
#include "vr_editor_avatar.h"
#include "vr_keyboard.h"
#include "vr_window.h"

/* VRPoke */

void VRPoke::_set_ray_visible_length(float p_length) {
	if (ray_mesh.is_valid()) {
		ray_mesh->set_size(Vector3(0.001, 0.001, p_length));
	}
	if (cast) {
		cast->set_position(Vector3(0.0, 0.0, -0.5 * p_length));
	}
}

VRCollision::CollisionType VRPoke::check_for_collisions(VRCollision **r_collision, Vector3 &r_collision_position, float &r_collision_distance) {
	VRCollision::CollisionType collision_type = VRCollision::CollisionType::NONE;
	*r_collision = nullptr;
	r_collision_distance = 0.0;

	if (touch_enabled || ray_enabled) {
		Transform3D poke_transform = get_global_transform();
		Vector3 poke_position = poke_transform.origin;
		Vector3 poke_direction = -poke_transform.basis.get_column(2);

		Vector<VRCollision *> collisions = VRCollision::get_hit_tests(true, false);
		for (int i = 0; i < collisions.size(); i++) {
			VRCollision *test_collision = collisions[i];
			Vector3 test_position;
			float test_distance = 9999.99;

			if (touch_enabled && test_collision->within_sphere(poke_position, radius, test_position)) {
				// only simulate one button when we're touching
				test_distance = (test_position - poke_position).length_squared();
				collision_type = VRCollision::CollisionType::SPHERE;
			} else if (ray_enabled && test_collision->raycast(poke_position, poke_direction, test_position)) {
				test_distance = (test_position - poke_position).length_squared();
				collision_type = VRCollision::CollisionType::RAYCAST;
			} else {
				test_collision = nullptr;
			}

			// if we have a collision, check if we're closer
			if (test_collision && (!(*r_collision) || (test_distance < r_collision_distance))) {
				*r_collision = test_collision;
				r_collision_position = test_position;
				r_collision_distance = test_distance;
			}
		}
	}
	return collision_type;
}

void VRPoke::set_touch_enabled(bool p_enabled) {
	touch_enabled = p_enabled;

	if (sphere) {
		sphere->set_visible(touch_enabled);
	}
}

void VRPoke::set_ray_enabled(bool p_enabled) {
	ray_enabled = p_enabled;

	if (cast) {
		cast->set_visible(ray_enabled);
	}
}

void VRPoke::set_ray_collided(bool p_collided, float p_collision_distance) {
	if (p_collided) {
		_set_ray_visible_length(Math::sqrt(p_collision_distance));
		material->set_albedo(touch_color);
	} else {
		material->set_albedo(normal_color);
		_set_ray_visible_length(5.0);
	}
}

VRPoke::VRPoke() {
	material.instantiate();
	material->set_shading_mode(BaseMaterial3D::SHADING_MODE_UNSHADED);
	material->set_transparency(BaseMaterial3D::TRANSPARENCY_ALPHA);
	material->set_albedo(normal_color);

	Ref<SphereMesh> sphere_mesh;
	sphere_mesh.instantiate();
	sphere_mesh->set_radius(radius);
	sphere_mesh->set_height(radius * 2.0);
	sphere_mesh->set_radial_segments(16);
	sphere_mesh->set_rings(8);
	sphere_mesh->set_material(material);

	sphere = memnew(MeshInstance3D);
	sphere->set_mesh(sphere_mesh);
	sphere->set_visible(touch_enabled);
	add_child(sphere);

	ray_mesh.instantiate();
	ray_mesh->set_material(material);

	cast = memnew(MeshInstance3D);
	cast->set_mesh(ray_mesh);
	cast->set_visible(ray_enabled);
	_set_ray_visible_length(5.0);
	add_child(cast);

	set_process_internal(true);
}

VRPoke::~VRPoke() {
}

void VRHand::set_grab_radius(float p_radius) {
	grab_radius = p_radius;
}

Vector2 VRHand::get_scroll() {
	return get_vector2("scroll");
}

bool VRHand::is_left_click() {
	// Returns true if our left_click action is triggered.
	return is_button_pressed("left_click");
}

bool VRHand::is_middle_click() {
	// Returns true if the middle_click action is triggered.
	return is_button_pressed("middle_click");
}

bool VRHand::is_right_click() {
	// Returns true if our right_click action is triggered.
	return is_button_pressed("right_click");
}

void VRHand::_on_grab_pressed(VRCollision *collision, const Vector3 &collision_position) {
	grab_pressed = true;
	if (collision) {
		collision->on_grab_pressed(*this, collision_position);
	} else {
		VREditorAvatar *editor_avatar = Object::cast_to<VREditorAvatar>(get_parent());
		ERR_FAIL_NULL(editor_avatar);

		editor_avatar->on_grab_pressed(*this);
	}
}

void VRHand::_on_grab_released(VRCollision *collision) {
	grab_pressed = false;
	if (collision) {
		collision->on_grab_released(*this);
	} else {
		VREditorAvatar *editor_avatar = Object::cast_to<VREditorAvatar>(get_parent());
		ERR_FAIL_NULL(editor_avatar);

		editor_avatar->on_grab_released(*this);
	}
}

/* VRHand */

void VRHand::_notification(int p_notification) {
	switch (p_notification) {
		case NOTIFICATION_INTERNAL_PROCESS: {
			// Controllers that have a pressure sensor can have unreliable thresholds so we hardcode this.
			// Controllers that don't have a pressure sensor simply return 0.0 or 1.0
			float grab_value = get_float("grab");
			if (grab_pressed) {
				if (grab_value > 0.65) {
					// Grab is still pressed; since it takes priority, we abort.
					_on_grab_pressed(last_collision, last_position);
					return;
				}
				if (grab_value < 0.35) {
					_on_grab_released(last_collision);
				}
			}

			if (!poke) {
				return;
			}

			// Check for collisions
			VRCollision *collision = nullptr;
			Vector3 collision_position;
			float collision_distance;
			VRCollision::CollisionType collision_type = poke->check_for_collisions(&collision, collision_position, collision_distance);

			// Last collision no longer relevant?
			if (last_collision && last_collision != collision) {
				// Might want to leave first and forego on the releases?
				if (last_was_pressed[2]) {
					last_collision->_on_interact_released(last_position, MouseButton::MIDDLE);
					last_was_pressed[2] = false;
				}
				if (last_was_pressed[1]) {
					last_collision->_on_interact_released(last_position, MouseButton::RIGHT);
					last_was_pressed[1] = false;
				}
				if (last_was_pressed[0]) {
					last_collision->_on_interact_released(last_position, MouseButton::LEFT);
					last_was_pressed[0] = false;
				}
				last_collision->_on_interact_leave(last_position);

				// reset
				last_collision = nullptr;
			}

			if (grab_value > 0.65) {
				_on_grab_pressed(collision, collision_position);
			}

			const Vector2 scroll = get_scroll();
			// Hit something?
			if (collision_type != VRCollision::NONE) {
				ERR_FAIL_NULL(collision);
				poke->set_ray_collided(true, collision_distance);

				if (last_collision != collision) {
					last_collision = collision;
					last_position = collision_position; // We don't want to trigger a move event here.
					collision->_on_interact_enter(collision_position);
				}

				bool col_is_pressed[3] = { false, false, false };
				float col_pressure[3] = { 0.0, 0.0, 0.0 };
				if (collision_type == VRCollision::SPHERE) {
					col_is_pressed[0] = true;
					col_pressure[0] = CLAMP(1.0 - (Math::sqrt(collision_distance) / poke->get_touch_radius()), 0.0, 1.0);
				} else {
					col_is_pressed[0] = is_left_click();
					col_is_pressed[1] = is_right_click();
					col_is_pressed[2] = is_middle_click();

					for (int i = 0; i < 3; i++) {
						col_pressure[i] = col_is_pressed[i] ? 1.0 : 0.0;
					}
				}

				for (int i = 0; i < 3; i++) {
					if (col_is_pressed[i] && !last_was_pressed[i]) {
						collision->_on_interact_pressed(collision_position, MouseButton(i + 1));
					} else if (!col_is_pressed[i] && last_was_pressed[i]) {
						collision->_on_interact_released(collision_position, MouseButton(i + 1));
					}
				}

				if (last_position != collision_position) {
					collision->_on_interact_moved(collision_position, col_pressure[0]);
				}

				last_position = collision_position;
				for (int i = 0; i < 3; i++) {
					last_was_pressed[i] = col_is_pressed[i];
				}

				// Forward scroll values
				collision->_on_interact_scrolled(collision_position, scroll);
			} else {
				poke->set_ray_collided(false, collision_distance);

				VREditorAvatar *editor_avatar = Object::cast_to<VREditorAvatar>(get_parent());
				ERR_FAIL_NULL(editor_avatar);

				editor_avatar->on_scroll(*this, scroll);
			}
		} break;
		default: {
			// ignore
		} break;
	}
}

VRHand::VRHand(XRPositionalTracker::TrackerHand p_hand) {
	// We need to make something nicer for our hands
	// maybe use hand tracking if available and see if we can support all fingers.
	// But for now just fingers on our default position is fine.

	set_name(p_hand == XRPositionalTracker::TRACKER_HAND_LEFT ? "left_hand" : "right_hand");
	set_tracker(p_hand == XRPositionalTracker::TRACKER_HAND_LEFT ? "left_hand" : "right_hand");
	set_pose_name(SNAME("tool_pose"));

	poke = memnew(VRPoke);
	poke->set_position(Vector3(0.0, 0.0, -0.01));
	add_child(poke);

	if (ClassDB::class_exists("OpenXRFbHandTrackingMesh")) {
		Node3D *hand_tracking_mesh = Object::cast_to<Node3D>(ClassDB::instantiate("OpenXRFbHandTrackingMesh"));
		if (hand_tracking_mesh) {
			hand_modifier = memnew(XRHandModifier3D);
			hand_modifier->set_hand_tracker(p_hand == XRPositionalTracker::TRACKER_HAND_LEFT ? "/user/left" : "/user/right");
			hand_modifier->add_child(hand_tracking_mesh);
		}
	}

	set_process_internal(true);
}

VRHand::~VRHand() {
}
