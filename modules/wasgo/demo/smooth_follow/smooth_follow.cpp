/**************************************************************************/
/*  smooth_follow.cpp                                                     */
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
#include "wasgo/wasgo.h"

Spatial parent;
Spatial child;
float smoothness;
Transform last_transform;

void _ready() {
	child = (Spatial)WasGo::this_node().get_node(WasGo::get_property_nodepath("child_node"));
	parent = (Spatial)WasGo::this_node().get_node(WasGo::get_property_nodepath("parent_node"));
	smoothness = WasGo::get_property_float("smoothness");
}

void _process(float delta) {
	if (child && parent) {
		Transform parent_transform = parent.get_transform();
		last_transform = parent_transform.interpolate_with(last_transform, smoothness);
		last_transform.set_basis(parent_transform.get_basis().slerp(last_transform.get_basis(), smoothness / 2));
		child.set_transform(last_transform);
		// child.set_rotation(parent.get_rotation() / 0.5);
	} else {
		child = (Spatial)WasGo::this_node().get_node(WasGo::get_property_nodepath("child_node"));
		parent = (Spatial)WasGo::this_node().get_node(WasGo::get_property_nodepath("parent_node"));
		last_transform = child.get_transform();
	}
}