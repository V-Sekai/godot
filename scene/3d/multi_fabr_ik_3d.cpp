/**************************************************************************/
/*  multi_fabr_ik_3d.cpp                                                  */
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

#include "multi_fabr_ik_3d.h"

void MultiFABRIK3D::_process_joints(double p_delta, Skeleton3D *p_skeleton) {
	for (int i = 0; i < settings.size(); i++) {
		Node3D *target = Object::cast_to<Node3D>(get_node_or_null(settings[i]->target_node));
		if (!target || settings[i]->joints.is_empty()) {
			continue; // Abort.
		}
		Node3D *pole = Object::cast_to<Node3D>(get_node_or_null(settings[i]->pole_node));
		Vector3 destination = settings[i]->cached_space.affine_inverse().xform(target->get_global_position());
		_process_joints_forward(p_delta, p_skeleton, settings[i], settings[i]->joints, settings[i]->chain, destination, !pole ? destination : settings[i]->cached_space.affine_inverse().xform(pole->get_global_position()), !!pole);
	}
	for (int i = 0; i < settings.size(); i++) {
		Node3D *target = Object::cast_to<Node3D>(get_node_or_null(settings[i]->target_node));
		if (!target || settings[i]->joints.is_empty()) {
			continue; // Abort.
		}
		Node3D *pole = Object::cast_to<Node3D>(get_node_or_null(settings[i]->pole_node));
		Vector3 destination = settings[i]->cached_space.affine_inverse().xform(target->get_global_position());
		_process_joints_backward(p_delta, p_skeleton, settings[i], settings[i]->joints, settings[i]->chain, destination, !pole ? destination : settings[i]->cached_space.affine_inverse().xform(pole->get_global_position()), !!pole);
	}
}
