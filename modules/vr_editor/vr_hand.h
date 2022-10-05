/**************************************************************************/
/*  vr_hand.h                                                             */
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

#include "core/disabled_classes.gen.h"
#include "core/object/callable_method_pointer.h"
#include "core/object/method_bind.h"
#include "core/object/object.h"
#include "core/string/print_string.h"
#include "core/templates/hash_set.h"
#include "scene/3d/label_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/xr/xr_hand_modifier_3d.h"
#include "scene/3d/xr/xr_nodes.h"
#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/material.h"
#include "vr_collision.h"
#include "vr_keyboard.h"
#include <type_traits>

class VRHand;

// VRPoke is our main UI device, it supports a raycast to interact with UI elements at distance, and supports close range touch detection
class VRPoke : public Node3D {
	GDCLASS(VRPoke, Node3D);

	// TODO add class for doing hits, now that we do raycast we can't do what we currently do.

private:
	float radius = 0.01;
	bool touch_enabled = true;
	bool ray_enabled = true;

	Color normal_color = Color(0.4, 0.4, 1.0, 0.5);
	Color touch_color = Color(0.8, 0.8, 1.0, 0.75);

	Ref<StandardMaterial3D> material;
	MeshInstance3D *sphere = nullptr;
	Ref<BoxMesh> ray_mesh;
	MeshInstance3D *cast = nullptr;

	void _set_ray_visible_length(float p_length);

public:
	void set_touch_enabled(bool p_enabled);
	bool get_touch_enabled() const { return touch_enabled; }

	void set_ray_enabled(bool p_enabled);
	bool get_ray_enabled() const { return ray_enabled; }

	float get_touch_radius() const { return radius; }

	VRCollision::CollisionType check_for_collisions(VRCollision **r_collision, Vector3 &r_collision_position, float &r_collision_distance);

	void set_ray_collided(bool p_collided, float p_collision_distance);

	VRPoke();
	~VRPoke();
};

class VRHand : public XRController3D {
	GDCLASS(VRHand, XRController3D);

protected:
	void _notification(int p_notification);

public:
	VRPoke *get_poke() const { return poke; }
	XRHandModifier3D *get_hand_modifier() const { return hand_modifier; }
	void set_ray_enabled(bool p_enabled) { poke->set_ray_enabled(p_enabled); }

	void set_grab_radius(float radius);
	float get_grab_radius() const { return grab_radius; }

	Vector2 get_scroll();
	bool is_left_click();
	bool is_middle_click();
	bool is_right_click();

	void _on_grab_pressed(VRCollision *collision, const Vector3 &collision_position);
	void _on_grab_released(VRCollision *collision);

	VRHand(XRPositionalTracker::TrackerHand p_hand);
	~VRHand();

private:
	VRPoke *poke = nullptr;
	XRHandModifier3D *hand_modifier = nullptr;

	bool grab_pressed = false;
	float grab_radius = 0.3;

	VRCollision *last_collision = nullptr;
	Vector3 last_position;
	bool last_was_pressed[3] = { false, false, false };
};
