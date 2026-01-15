/**************************************************************************/
/*  csg_sculpted_cylinder.cpp                                             */
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

#include "csg_sculpted_cylinder.h"
#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/mesh.h"

void CSGSculptedCylinder3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &CSGSculptedCylinder3D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &CSGSculptedCylinder3D::get_radius);

	ClassDB::bind_method(D_METHOD("set_height", "height"), &CSGSculptedCylinder3D::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &CSGSculptedCylinder3D::get_height);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height"), "set_height", "get_height");
}

CSGSculptedCylinder3D::CSGSculptedCylinder3D() {
	profile_curve = PROFILE_CURVE_CIRCLE;
	path_curve = PATH_CURVE_LINE;
}

void CSGSculptedCylinder3D::set_radius(const real_t p_radius) {
	radius = p_radius;
	_make_dirty();
}

real_t CSGSculptedCylinder3D::get_radius() const {
	return radius;
}

void CSGSculptedCylinder3D::set_height(const real_t p_height) {
	height = p_height;
	_make_dirty();
}

real_t CSGSculptedCylinder3D::get_height() const {
	return height;
}

CSGBrush *CSGSculptedCylinder3D::_build_brush() {
	return memnew(CSGBrush);
}
