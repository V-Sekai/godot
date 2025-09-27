/**************************************************************************/
/*  vr_editor_avatar.h                                                    */
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

#include "vr_collision.h"
#include "vr_hand.h"
#ifndef ANDROID_ENABLED
#include "vr_keyboard.h"
#endif

#include "scene/3d/label_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/xr/xr_nodes.h"
#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/material.h"

class VREditorAvatar : public XROrigin3D {
	GDCLASS(VREditorAvatar, XROrigin3D);

private:
	XRCamera3D *camera = nullptr;
	bool camera_is_tracking = false;

	Node3D *hud_pivot = nullptr;
	Node3D *hud_root = nullptr;
	real_t hud_offset = 0.0; // offset from eye height of our hud
	real_t hud_distance = 0.5; // desired distance of our hud to our player
	bool hud_moving = true; // we are adjusting the position of the hud

	float grab_translation_speed_factor;
	float grab_translation_deadzone;
	float scroll_translation_deadzone;
	float scroll_translation_speed_factor;
	int rotation_snap_angle_in_degrees;

	bool right_hand_grab_in_progress = false;
	Vector3 right_hand_grab_initial_local_position;
	Vector3 right_hand_grab_current_local_position;
	Vector3 right_hand_scale_last_global_position;
	Vector3 right_hand_scale_current_global_position;

	bool left_hand_grab_in_progress = false;
	Vector3 left_hand_grab_initial_local_position;
	Vector3 left_hand_grab_current_local_position;
	Vector3 left_hand_scale_last_global_position;
	Vector3 left_hand_scale_current_global_position;

	bool scaling_in_progress = false;

	XRPositionalTracker::TrackerHand ray_active_on_hand = XRPositionalTracker::TRACKER_HAND_RIGHT;
	VRHand *left_hand = nullptr;
	VRHand *right_hand = nullptr;
	void set_ray_active_on_hand(XRPositionalTracker::TrackerHand p_hand);
	void _on_button_pressed_on_hand(const String p_action, int p_hand);

	void _refresh_editor_settings();
	void _update_hud_position();
	void _handle_grab_pressed(const VRHand &hand, bool &r_grab_in_progress,
			Vector3 &r_grab_initial_local_position,
			Vector3 &r_grab_current_local_position,
			Vector3 &r_scale_last_global_position,
			Vector3 &r_scale_current_global_position);
	void _process_grab();

#ifndef ANDROID_ENABLED
	VRKeyboard *keyboard = nullptr;
#endif

	float time_since_last_print = 0.0f;

protected:
	void _notification(int p_notification);

public:
	Node3D *get_hud_root() const { return hud_root; }

	real_t get_hud_offset() const { return hud_offset; }
	void set_hud_offset(real_t p_offset);

	real_t get_hud_distance() const { return hud_distance; }
	void set_hud_distance(real_t p_distance);

	void set_camera_cull_layers(uint32_t p_layers);
	Transform3D get_camera_transform() const { return camera->get_global_transform(); }

	void on_scroll(const VRHand &hand, const Vector2 &scroll);
	void on_grab_pressed(const VRHand &hand);
	void on_grab_released(const VRHand &hand);

	VREditorAvatar();
	~VREditorAvatar();
};
