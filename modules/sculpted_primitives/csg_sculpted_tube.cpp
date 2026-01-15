/**************************************************************************/
/*  csg_sculpted_tube.cpp                                                 */
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

#include "csg_sculpted_tube.h"

#include "core/math/geometry_3d.h"

// Helper function to generate profile points based on curve type

// Helper function to apply path transformations

void CSGSculptedTube3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_inner_radius", "inner_radius"), &CSGSculptedTube3D::set_inner_radius);
	ClassDB::bind_method(D_METHOD("get_inner_radius"), &CSGSculptedTube3D::get_inner_radius);

	ClassDB::bind_method(D_METHOD("set_outer_radius", "outer_radius"), &CSGSculptedTube3D::set_outer_radius);
	ClassDB::bind_method(D_METHOD("get_outer_radius"), &CSGSculptedTube3D::get_outer_radius);

	ClassDB::bind_method(D_METHOD("set_height", "height"), &CSGSculptedTube3D::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &CSGSculptedTube3D::get_height);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "inner_radius"), "set_inner_radius", "get_inner_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "outer_radius"), "set_outer_radius", "get_outer_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height"), "set_height", "get_height");
}

CSGSculptedTube3D::CSGSculptedTube3D() {
	profile_curve = PROFILE_CURVE_CIRCLE;
	path_curve = PATH_CURVE_LINE;
}

void CSGSculptedTube3D::set_inner_radius(const real_t p_inner_radius) {
	inner_radius = p_inner_radius;
	_make_dirty();
}

real_t CSGSculptedTube3D::get_inner_radius() const {
	return inner_radius;
}

void CSGSculptedTube3D::set_outer_radius(const real_t p_outer_radius) {
	outer_radius = p_outer_radius;
	_make_dirty();
}

real_t CSGSculptedTube3D::get_outer_radius() const {
	return outer_radius;
}

void CSGSculptedTube3D::set_height(const real_t p_height) {
	height = p_height;
	_make_dirty();
}

real_t CSGSculptedTube3D::get_height() const {
	return height;
}

CSGBrush *CSGSculptedTube3D::_build_brush() {
	return memnew(CSGBrush);
}
