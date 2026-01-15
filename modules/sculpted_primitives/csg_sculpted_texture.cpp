/**************************************************************************/
/*  csg_sculpted_texture.cpp                                              */
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

#include "csg_sculpted_texture.h"

#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/mesh.h"

void CSGSculptedTexture3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_sculpt_texture", "texture"), &CSGSculptedTexture3D::set_sculpt_texture);
	ClassDB::bind_method(D_METHOD("get_sculpt_texture"), &CSGSculptedTexture3D::get_sculpt_texture);

	ClassDB::bind_method(D_METHOD("set_mirror", "mirror"), &CSGSculptedTexture3D::set_mirror);
	ClassDB::bind_method(D_METHOD("get_mirror"), &CSGSculptedTexture3D::get_mirror);

	ClassDB::bind_method(D_METHOD("set_invert", "invert"), &CSGSculptedTexture3D::set_invert);
	ClassDB::bind_method(D_METHOD("get_invert"), &CSGSculptedTexture3D::get_invert);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "sculpt_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_sculpt_texture", "get_sculpt_texture");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "mirror"), "set_mirror", "get_mirror");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "invert"), "set_invert", "get_invert");
}

void CSGSculptedTexture3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (sculpt_texture.is_valid()) {
				sculpt_texture->connect("changed", callable_mp(this, &CSGSculptedTexture3D::_texture_changed));
			}
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (sculpt_texture.is_valid()) {
				sculpt_texture->disconnect("changed", callable_mp(this, &CSGSculptedTexture3D::_texture_changed));
			}
		} break;
	}
}

CSGSculptedTexture3D::CSGSculptedTexture3D() {
	profile_curve = PROFILE_CURVE_CIRCLE;
	path_curve = PATH_CURVE_LINE;
}

void CSGSculptedTexture3D::_texture_changed() {
	_make_dirty();
}

void CSGSculptedTexture3D::set_sculpt_texture(const Ref<Texture2D> &p_texture) {
	if (sculpt_texture.is_valid()) {
		if (is_inside_tree()) {
			sculpt_texture->disconnect("changed", callable_mp(this, &CSGSculptedTexture3D::_texture_changed));
		}
	}
	sculpt_texture = p_texture;
	if (sculpt_texture.is_valid()) {
		if (is_inside_tree()) {
			sculpt_texture->connect("changed", callable_mp(this, &CSGSculptedTexture3D::_texture_changed));
		}
	}
	_make_dirty();
}

Ref<Texture2D> CSGSculptedTexture3D::get_sculpt_texture() const {
	return sculpt_texture;
}

void CSGSculptedTexture3D::set_mirror(bool p_mirror) {
	mirror = p_mirror;
	_make_dirty();
}

bool CSGSculptedTexture3D::get_mirror() const {
	return mirror;
}

void CSGSculptedTexture3D::set_invert(bool p_invert) {
	invert = p_invert;
	_make_dirty();
}

bool CSGSculptedTexture3D::get_invert() const {
	return invert;
}

CSGBrush *CSGSculptedTexture3D::_build_brush() {
	return memnew(CSGBrush);
}
