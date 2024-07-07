/**************************************************************************/
/*  turning.cpp                                                           */
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

#include "Input.h"
#include "InputEvent.h"
#include "InputEventMouseMotion.h"
#include "Spatial.h"
#include "wasgo/wasgo.h"
#include <stdio.h>

Vector2 mouse_speed;
Spatial plane;
float sensitivity;

void _ready() {
	// Input::set_mouse_mode(Input::MOUSE_MODE_CAPTURED);
	plane = (Spatial)WasGo::this_node().get_node((WasGo::get_property_nodepath("plane_node")));
	sensitivity = WasGo::get_property_float("sensitivity");
}

void _physics_process(float delta) {
	if (plane) {
		Input::set_mouse_mode(Input::MOUSE_MODE_CAPTURED);
		plane.rotate_object_local(Vector3(1, 0, 0), mouse_speed.y);
		plane.rotate_object_local(Vector3(0, 0, 1), mouse_speed.x);
	} else {
		plane = (Spatial)WasGo::this_node().get_node((WasGo::get_property_nodepath("plane_node")));
	}
}

void _input(InputEvent event) {
	// InputEventMouse mouse_event = (InputEventMouse)event;
	InputEventMouseMotion mouse_motion = InputEventMouseMotion(InputEventMouse(InputEventWithModifiers(event)));
	mouse_speed = mouse_speed.linear_interpolate(mouse_motion.get_relative() / 1000, sensitivity);
	// printf("mouse speed: %d %d\n", int(mouse_speed.x * 1000), int(mouse_speed.y * 1000));
}
