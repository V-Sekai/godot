/**************************************************************************/
/*  vr_editor_avatar.cpp                                                  */
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

#include "editor/settings/editor_settings.h"

#include "vr_editor_avatar.h"
#include "vr_util.h"

void VREditorAvatar::set_ray_active_on_hand(XRPositionalTracker::TrackerHand p_hand) {
	ray_active_on_hand = p_hand;
	left_hand->set_ray_enabled(p_hand == XRPositionalTracker::TRACKER_HAND_LEFT);
	right_hand->set_ray_enabled(p_hand == XRPositionalTracker::TRACKER_HAND_RIGHT);
}

void VREditorAvatar::_on_button_pressed_on_hand(const String p_action, int p_hand) {
	if (p_action == "left_click") {
		set_ray_active_on_hand(XRPositionalTracker::TrackerHand(p_hand));
	}
}

void VREditorAvatar::set_hud_offset(real_t p_offset) {
	hud_offset = p_offset;

	Vector3 position = hud_pivot->get_position();
	position.y = hud_offset;
	hud_pivot->set_position(position);
}

void VREditorAvatar::set_hud_distance(real_t p_distance) {
	hud_distance = p_distance;

	Vector3 position = hud_root->get_position();
	position.z = -hud_distance;
	hud_root->set_position(position);
}

void VREditorAvatar::set_camera_cull_layers(uint32_t p_layers) {
	ERR_FAIL_NULL(camera);

	camera->set_cull_mask(p_layers);
}

void VREditorAvatar::on_scroll(const VRHand &hand, const Vector2 &scroll) {
	const real_t world_scale = get_world_scale();
	if (hand.get_tracker_hand() == XRPositionalTracker::TRACKER_HAND_LEFT) {
		// Scroll values are used to move the avatar on the X / Z axis
		real_t x_value = Math::abs(scroll.x) >= scroll_translation_deadzone ? scroll.x * scroll_translation_speed_factor : 0;
		real_t z_value = Math::abs(scroll.y) >= scroll_translation_deadzone ? -scroll.y * scroll_translation_speed_factor : 0;

		Vector3 translation_offset = Vector3(x_value, 0, z_value) / world_scale;
		translate(translation_offset);
	} else {
		// Pick the axis with the higher magnitude
		if (Math::abs(scroll.y) >= Math::abs(scroll.x)) {
			// Scroll values are used to move the avatar on the Y axis and pivot it
			real_t y_value = Math::abs(scroll.y) >= scroll_translation_deadzone ? scroll.y * scroll_translation_speed_factor : 0;
			Vector3 translation_offset = Vector3(0, y_value, 0) / world_scale;
			translate(translation_offset);
		} else {
			real_t rotation_angle = -scroll.x * (rotation_snap_angle_in_degrees * Math::PI / 180);
			rotate_y(rotation_angle);
		}
	}
}

void VREditorAvatar::_handle_grab_pressed(const VRHand &hand, bool &r_grab_in_progress,
		Vector3 &r_grab_initial_local_position,
		Vector3 &r_grab_current_local_position,
		Vector3 &r_scale_last_global_position,
		Vector3 &r_scale_current_global_position) {
	if (!r_grab_in_progress) {
		// Record the starting grab position
		r_grab_initial_local_position = hand.get_position();
		r_scale_current_global_position = hand.get_global_position();
	}

	r_scale_last_global_position = r_scale_current_global_position;
	r_scale_current_global_position = hand.get_global_position();

	r_grab_current_local_position = hand.get_position();
	r_grab_in_progress = true;
}

void VREditorAvatar::on_grab_pressed(const VRHand &hand) {
	if (hand.get_tracker_hand() == XRPositionalTracker::TRACKER_HAND_LEFT) {
		_handle_grab_pressed(hand, left_hand_grab_in_progress,
				left_hand_grab_initial_local_position,
				left_hand_grab_current_local_position,
				left_hand_scale_last_global_position,
				left_hand_scale_current_global_position);
	} else {
		_handle_grab_pressed(hand, right_hand_grab_in_progress,
				right_hand_grab_initial_local_position,
				right_hand_grab_current_local_position,
				right_hand_scale_last_global_position,
				right_hand_scale_current_global_position);
	}
}

void VREditorAvatar::on_grab_released(const VRHand &hand) {
	if (hand.get_tracker_hand() == XRPositionalTracker::TRACKER_HAND_LEFT) {
		left_hand_grab_in_progress = false;
	} else {
		right_hand_grab_in_progress = false;
	}
}

void VREditorAvatar::_process_grab() {
	if (!right_hand_grab_in_progress && !left_hand_grab_in_progress) {
		scaling_in_progress = false;
		return;
	}

	if (right_hand_grab_in_progress && left_hand_grab_in_progress) {
		// Scaling is in progress
		scaling_in_progress = true;
		Vector3 last_scale_vector = right_hand_scale_last_global_position - left_hand_scale_last_global_position;
		Vector3 current_scale_vector = right_hand_scale_current_global_position - left_hand_scale_current_global_position;
		if (last_scale_vector.is_zero_approx() || current_scale_vector.is_zero_approx() || last_scale_vector.is_equal_approx(current_scale_vector)) {
			return;
		}
		const real_t last_scale_vector_distance = last_scale_vector.length();
		const real_t current_scale_vector_distance = current_scale_vector.length();
		if (Math::is_zero_approx(last_scale_vector_distance)) {
			return;
		}

		const real_t scale_factor = ((current_scale_vector_distance / last_scale_vector_distance) - 1);
		real_t world_scale = get_world_scale();
		real_t updated_world_scale = CLAMP(world_scale - scale_factor, 0.8, 5);
		if (Math::is_equal_approx(world_scale, updated_world_scale)) {
			return;
		}

		set_world_scale(updated_world_scale);
		print_verbose(vformat("XR world scale updated to: %f", updated_world_scale));
	} else if (!scaling_in_progress) {
		// Translation via grab is in progress
		Vector3 initial_grab_position = right_hand_grab_in_progress ? right_hand_grab_initial_local_position : left_hand_grab_initial_local_position;
		Vector3 current_grab_position = right_hand_grab_in_progress ? right_hand_grab_current_local_position : left_hand_grab_current_local_position;

		if (initial_grab_position.is_equal_approx(current_grab_position)) {
			return;
		}

		Vector3 translation_offset = (current_grab_position - initial_grab_position);
		translation_offset = Vector3(Math::abs(translation_offset.x) > grab_translation_deadzone ? translation_offset.x : 0,
				Math::abs(translation_offset.y) > grab_translation_deadzone ? translation_offset.y : 0,
				Math::abs(translation_offset.z) > grab_translation_deadzone ? translation_offset.z : 0);
		if (translation_offset.is_zero_approx()) {
			return;
		}
		translation_offset *= grab_translation_speed_factor;
		translate(translation_offset);
	}
}

VREditorAvatar::VREditorAvatar() {
	// TODO once https://github.com/godotengine/godot/pull/63607 is merged we need to add an enhancement
	// to make this node the "current" XROrigin3D node.
	// For now this will be the current node but if a VR project is loaded things could go haywire.

	_refresh_editor_settings();

	camera = memnew(XRCamera3D);
	camera->set_name("camera");
	add_child(camera);

	// Our hud pivot will follow our camera around at a constant height.
	// TODO add a button press or other mechanism to rotate our hud pivot
	// so our hud is recentered in front of our player.
	hud_pivot = memnew(Node3D);
	hud_pivot->set_name("hud_pivot");
	hud_pivot->set_position(Vector3(0.0, 1.6 + hud_offset, 0.0)); // we don't know our eye height yet
	add_child(hud_pivot);

	// Our hud root extends our hud outwards to a certain distance away
	// from our player.
	hud_root = memnew(Node3D);
	hud_root->set_name("hud_root");
	hud_root->set_position(Vector3(0.0, 0.0, -hud_distance));
	hud_pivot->add_child(hud_root);

	// Add our hands
	left_hand = memnew(VRHand(XRPositionalTracker::TRACKER_HAND_LEFT));
	left_hand->connect("button_pressed", callable_mp(this, &VREditorAvatar::_on_button_pressed_on_hand).bind(int(XRPositionalTracker::TRACKER_HAND_LEFT)));
	add_child(left_hand);

	if (left_hand->get_hand_modifier() != nullptr) {
		add_child(left_hand->get_hand_modifier());
	}

	right_hand = memnew(VRHand(XRPositionalTracker::TRACKER_HAND_RIGHT));
	right_hand->connect("button_pressed", callable_mp(this, &VREditorAvatar::_on_button_pressed_on_hand).bind(int(XRPositionalTracker::TRACKER_HAND_RIGHT)));
	add_child(right_hand);

	if (right_hand->get_hand_modifier() != nullptr) {
		add_child(right_hand->get_hand_modifier());
	}

	// TODO add callback for left_click so we can activate ray on last used hand

	set_ray_active_on_hand(ray_active_on_hand);

#ifndef ANDROID_ENABLED
	// Add virtual keyboard
	keyboard = memnew(VRKeyboard);
	keyboard->set_name("VRKeyboard");
	keyboard->set_rotation(Vector3(20.0 * Math_PI / 180.0, 0.0, 0.0)); // put at a slight angle for comfort
	keyboard->set_position(Vector3(0.0, -0.4, 0.2)); // should make this a setting or something we can change
	hud_root->add_child(keyboard);
#endif

	set_process(true);

	// Our default transform logic in XROrigin3D is disabled in editor mode,
	// this should be our only active XROrigin3D node in our VR editor
	set_notify_local_transform(true);
	set_notify_transform(true);
}

VREditorAvatar::~VREditorAvatar() {
}

void VREditorAvatar::_notification(int p_notification) {
	switch (p_notification) {
		case NOTIFICATION_ENTER_TREE: {
			Ref<XRInterface> xr_interface = XRServer::get_singleton()->find_interface("OpenXR");
			update_passthrough_mode(xr_interface, get_viewport());
		} break;

		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			_refresh_editor_settings();

			Ref<XRInterface> xr_interface = XRServer::get_singleton()->find_interface("OpenXR");
			update_passthrough_mode(xr_interface, get_viewport());
		} break;

		case NOTIFICATION_PROCESS: {
			_update_hud_position();
			_process_grab();
		} break;

		case NOTIFICATION_LOCAL_TRANSFORM_CHANGED:
		case NOTIFICATION_TRANSFORM_CHANGED: {
			XRServer::get_singleton()->set_world_origin(get_global_transform());
		} break;
	}
}

void VREditorAvatar::_refresh_editor_settings() {
	grab_translation_deadzone = EDITOR_GET("interface/xr_editor/grab_translation_deadzone");
	grab_translation_speed_factor = EDITOR_GET("interface/xr_editor/grab_translation_speed_factor");

	scroll_translation_deadzone = EDITOR_GET("interface/xr_editor/scroll_translation_deadzone");
	scroll_translation_speed_factor = EDITOR_GET("interface/xr_editor/scroll_translation_speed_factor");

	rotation_snap_angle_in_degrees = EDITOR_GET("interface/xr_editor/rotation_snap_angle_in_degrees");
}

void VREditorAvatar::_update_hud_position() {
	double delta = get_process_delta_time();

	XRPose::TrackingConfidence confidence = camera->get_tracking_confidence();
	if (confidence == XRPose::XR_TRACKING_CONFIDENCE_NONE) {
		if (camera_is_tracking) {
			// We are not tracking so keep things where they are, user is likely not wearing the headset
			camera_is_tracking = false;

			print_line("HMD is no longer tracking");
		}
	} else {
		// Center our hud on our camera, start by calculating our desired location
		Vector3 desired_location = camera->get_position();
		desired_location.y = MAX(0.5, desired_location.y + hud_offset);

		if (!camera_is_tracking) {
			// If we weren't tracking, reposition our HUD right away, user likely just put on their headset
			camera_is_tracking = true;
			print_line("HMD is now tracking");

			hud_pivot->set_position(desired_location);
		} else {
			// If we were tracking, have HUD follow head movement, this prevents motion sickness

			// Now update our transform.
			bool update_location = false;
			Vector3 hud_location = hud_pivot->get_position();

			// If our desired location is more then 10cm away,
			// we start moving until our hud is within 1 cm of
			// the desired location.
			if ((desired_location - hud_location).length() > (hud_moving ? 0.01 : 0.2)) {
				hud_location = hud_location.lerp(desired_location, delta);
				hud_moving = true;
				update_location = true;
			} else {
				hud_moving = false;
			}

			if (update_location) {
				hud_pivot->set_position(hud_location);
			}
		}
	}
}
