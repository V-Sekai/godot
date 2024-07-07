/**************************************************************************/
/*  thrust.cpp                                                            */
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

#include "Spatial.h"
#include "Transform.h"
#include "keyboard.h"
#include "math_defs.h"
#include "wasgo\wasgo.h"
#include <stdio.h>
#include <stdlib.h>

float velocity = 0;
int frame_counter = 0;
float thrust = 0;
float thrust_limit = 50;
float thrust_delta = 0;
float gravity_influence = 100;
float smooth = 0.9;

void _ready() {
	thrust_limit = WasGo::get_property_float("thrust_limit");
	gravity_influence = WasGo::get_property_float("gravity_influence");
	smooth = WasGo::get_property_float("smoothness");
}

void _unhandled_key_input(InputEventKey p_key_event) {
	p_key_event.get_scancode();
	switch (p_key_event.get_scancode()) {
		case KEY_W: {
			if (p_key_event.is_pressed()) {
				thrust_delta = 10;
			} else {
				thrust_delta = 0;
			}
		} break;
		case KEY_S: {
			if (p_key_event.is_pressed()) {
				thrust_delta = -10;
			} else {
				thrust_delta = 0;
			}
		} break;

		default:
			break;
	}
}

void _physics_process(float delta) {
	float y_ang = 0;
	Spatial plane;
	Vector3 direction;
	Vector3 target_vector;
	Transform t;
	float target_speed = 0;

	Node this_node = WasGo::this_node();
	Variant path = WasGo::get_property_nodepath("other_node");
	plane = (Spatial)this_node.get_node(path);

	Basis b = plane.get_transform().get_basis();
	direction = b.get_axis(2);
	y_ang = direction.angle_to(Vector3(0, 1, 0));
	if ((thrust <= thrust_limit && thrust_delta > 0) || (thrust >= 0 && thrust_delta < 0)) {
		thrust += thrust_delta;
	}
	target_speed = thrust + (y_ang / Math_PI - 0.5) * gravity_influence; //we subtract 0.5 from the angle over pi because gravity's influence should be zero at 90 degrees or when the angle over pi is 0.5
	velocity = (velocity * smooth + target_speed * (1 - smooth)) / 2;
	target_vector = Vector3(0, 0, velocity);
	t = plane.get_transform();
	t.translate(target_vector * delta);
	plane.set_transform(t);

	//only print every 120 frames to prevent log spam
	frame_counter = (frame_counter + 1) % 120;
	if (frame_counter == 0) {
		printf("thrust: %d\n", int(thrust * 1000));
		printf("y_ang: %d\n", int(y_ang * 1000));
		printf("velocity: %d\n", int(velocity * 1000));
	}
}
