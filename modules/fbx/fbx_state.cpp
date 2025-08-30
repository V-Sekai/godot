/**************************************************************************/
/*  fbx_state.cpp                                                         */
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

#include "fbx_state.h"

void FBXState::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_allow_geometry_helper_nodes"), &FBXState::get_allow_geometry_helper_nodes);
	ClassDB::bind_method(D_METHOD("set_allow_geometry_helper_nodes", "allow"), &FBXState::set_allow_geometry_helper_nodes);

	// Export options API bindings
	ClassDB::bind_method(D_METHOD("set_export_ascii_format", "ascii_format"), &FBXState::set_export_ascii_format);
	ClassDB::bind_method(D_METHOD("get_export_ascii_format"), &FBXState::get_export_ascii_format);
	ClassDB::bind_method(D_METHOD("set_export_embed_textures", "embed_textures"), &FBXState::set_export_embed_textures);
	ClassDB::bind_method(D_METHOD("get_export_embed_textures"), &FBXState::get_export_embed_textures);
	ClassDB::bind_method(D_METHOD("set_export_animations", "export_animations"), &FBXState::set_export_animations);
	ClassDB::bind_method(D_METHOD("get_export_animations"), &FBXState::get_export_animations);
	ClassDB::bind_method(D_METHOD("set_export_materials", "export_materials"), &FBXState::set_export_materials);
	ClassDB::bind_method(D_METHOD("get_export_materials"), &FBXState::get_export_materials);
	ClassDB::bind_method(D_METHOD("set_export_fbx_version", "fbx_version"), &FBXState::set_export_fbx_version);
	ClassDB::bind_method(D_METHOD("get_export_fbx_version"), &FBXState::get_export_fbx_version);
	ClassDB::bind_method(D_METHOD("set_export_coordinate_system", "coordinate_system"), &FBXState::set_export_coordinate_system);
	ClassDB::bind_method(D_METHOD("get_export_coordinate_system"), &FBXState::get_export_coordinate_system);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "allow_geometry_helper_nodes"), "set_allow_geometry_helper_nodes", "get_allow_geometry_helper_nodes");

	BIND_ENUM_CONSTANT(COORDINATE_SYSTEM_Y_UP);
	BIND_ENUM_CONSTANT(COORDINATE_SYSTEM_Z_UP);
}

bool FBXState::get_allow_geometry_helper_nodes() {
	return allow_geometry_helper_nodes;
}

void FBXState::set_allow_geometry_helper_nodes(bool p_allow_geometry_helper_nodes) {
	allow_geometry_helper_nodes = p_allow_geometry_helper_nodes;
}

// Export options implementation
void FBXState::set_export_ascii_format(bool p_ascii_format) {
	export_opts.ascii_format = p_ascii_format;
}

bool FBXState::get_export_ascii_format() const {
	return export_opts.ascii_format;
}

void FBXState::set_export_embed_textures(bool p_embed_textures) {
	export_opts.embed_textures = p_embed_textures;
}

bool FBXState::get_export_embed_textures() const {
	return export_opts.embed_textures;
}

void FBXState::set_export_animations(bool p_export_animations) {
	export_opts.export_animations = p_export_animations;
}

bool FBXState::get_export_animations() const {
	return export_opts.export_animations;
}

void FBXState::set_export_materials(bool p_export_materials) {
	export_opts.export_materials = p_export_materials;
}

bool FBXState::get_export_materials() const {
	return export_opts.export_materials;
}

void FBXState::set_export_fbx_version(int p_fbx_version) {
	export_opts.fbx_version = p_fbx_version;
}

int FBXState::get_export_fbx_version() const {
	return export_opts.fbx_version;
}

void FBXState::set_export_coordinate_system(CoordinateSystem p_coordinate_system) {
	switch (p_coordinate_system) {
		case COORDINATE_SYSTEM_Y_UP:
			export_opts.axes = ufbx_axes_right_handed_y_up;
			break;
		case COORDINATE_SYSTEM_Z_UP:
			export_opts.axes = ufbx_axes_right_handed_z_up;
			break;
	}
	export_opts.unit_meters = 1.0f;
}

FBXState::CoordinateSystem FBXState::get_export_coordinate_system() const {
	// Simple comparison based on up axis
	if (export_opts.axes.up == UFBX_COORDINATE_AXIS_POSITIVE_Y) {
		return COORDINATE_SYSTEM_Y_UP;
	}
	return COORDINATE_SYSTEM_Z_UP;
}
