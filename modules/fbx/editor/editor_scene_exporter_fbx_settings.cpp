/**************************************************************************/
/*  editor_scene_exporter_fbx_settings.cpp                                */
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

#include "editor_scene_exporter_fbx_settings.h"

bool EditorSceneExporterFBXSettings::_set(const StringName &p_name, const Variant &p_value) {
	if (p_name == StringName("copyright")) {
		set_copyright(p_value);
		return true;
	}
	if (p_name == StringName("bake_fps")) {
		set_bake_fps(p_value);
		return true;
	}
	if (p_name == StringName("ascii_format")) {
		set_ascii_format(p_value);
		emit_signal(CoreStringName(property_list_changed));
		return true;
	}
	if (p_name == StringName("embed_textures")) {
		set_embed_textures(p_value);
		return true;
	}
	if (p_name == StringName("export_animations")) {
		set_export_animations(p_value);
		return true;
	}
	if (p_name == StringName("export_materials")) {
		set_export_materials(p_value);
		return true;
	}
	if (p_name == StringName("export_skinning")) {
		set_export_skinning(p_value);
		return true;
	}
	if (p_name == StringName("export_morph_targets")) {
		set_export_morph_targets(p_value);
		return true;
	}
	if (p_name == StringName("optimize_skin_weights")) {
		set_optimize_skin_weights(p_value);
		return true;
	}
	if (p_name == StringName("max_skin_weights_per_vertex")) {
		set_max_skin_weights_per_vertex(p_value);
		return true;
	}
	if (p_name == StringName("fbx_version")) {
		set_fbx_version(p_value);
		return true;
	}
	if (p_name == StringName("coordinate_system")) {
		set_coordinate_system((FBXState::CoordinateSystem)(int)p_value);
		return true;
	}
	return false;
}

bool EditorSceneExporterFBXSettings::_get(const StringName &p_name, Variant &r_ret) const {
	if (p_name == StringName("copyright")) {
		r_ret = get_copyright();
		return true;
	}
	if (p_name == StringName("bake_fps")) {
		r_ret = get_bake_fps();
		return true;
	}
	if (p_name == StringName("ascii_format")) {
		r_ret = get_ascii_format();
		return true;
	}
	if (p_name == StringName("embed_textures")) {
		r_ret = get_embed_textures();
		return true;
	}
	if (p_name == StringName("export_animations")) {
		r_ret = get_export_animations();
		return true;
	}
	if (p_name == StringName("export_materials")) {
		r_ret = get_export_materials();
		return true;
	}
	if (p_name == StringName("export_skinning")) {
		r_ret = get_export_skinning();
		return true;
	}
	if (p_name == StringName("export_morph_targets")) {
		r_ret = get_export_morph_targets();
		return true;
	}
	if (p_name == StringName("optimize_skin_weights")) {
		r_ret = get_optimize_skin_weights();
		return true;
	}
	if (p_name == StringName("max_skin_weights_per_vertex")) {
		r_ret = get_max_skin_weights_per_vertex();
		return true;
	}
	if (p_name == StringName("fbx_version")) {
		r_ret = get_fbx_version();
		return true;
	}
	if (p_name == StringName("coordinate_system")) {
		r_ret = get_coordinate_system();
		return true;
	}
	return false;
}

void EditorSceneExporterFBXSettings::_get_property_list(List<PropertyInfo> *p_list) const {
	for (PropertyInfo prop : _property_list) {
		p_list->push_back(prop);
	}
}

void EditorSceneExporterFBXSettings::generate_property_list(Ref<FBXDocument> p_document, Node *p_root) {
	_property_list.clear();
	_document = p_document;

	// Add all FBX export properties to dynamic property list
	PropertyInfo copyright_prop = PropertyInfo(Variant::STRING, "copyright", PROPERTY_HINT_PLACEHOLDER_TEXT, "Example: 2024 Your Company");
	_property_list.push_back(copyright_prop);

	PropertyInfo bake_fps_prop = PropertyInfo(Variant::FLOAT, "bake_fps", PROPERTY_HINT_NONE, "");
	_property_list.push_back(bake_fps_prop);

	PropertyInfo ascii_format_prop = PropertyInfo(Variant::BOOL, "ascii_format", PROPERTY_HINT_NONE, "");
	_property_list.push_back(ascii_format_prop);

	PropertyInfo fbx_version_prop = PropertyInfo(Variant::INT, "fbx_version", PROPERTY_HINT_ENUM, "FBX 7.3 (7300),FBX 7.4 (7400),FBX 7.5 (7500)");
	_property_list.push_back(fbx_version_prop);

	PropertyInfo coordinate_system_prop = PropertyInfo(Variant::INT, "coordinate_system", PROPERTY_HINT_ENUM, "Y-Up,Z-Up");
	_property_list.push_back(coordinate_system_prop);

	PropertyInfo embed_textures_prop = PropertyInfo(Variant::BOOL, "embed_textures", PROPERTY_HINT_NONE, "");
	_property_list.push_back(embed_textures_prop);

	PropertyInfo export_animations_prop = PropertyInfo(Variant::BOOL, "export_animations", PROPERTY_HINT_NONE, "");
	_property_list.push_back(export_animations_prop);

	PropertyInfo export_materials_prop = PropertyInfo(Variant::BOOL, "export_materials", PROPERTY_HINT_NONE, "");
	_property_list.push_back(export_materials_prop);

	PropertyInfo export_skinning_prop = PropertyInfo(Variant::BOOL, "export_skinning", PROPERTY_HINT_NONE, "");
	_property_list.push_back(export_skinning_prop);

	PropertyInfo export_morph_targets_prop = PropertyInfo(Variant::BOOL, "export_morph_targets", PROPERTY_HINT_NONE, "");
	_property_list.push_back(export_morph_targets_prop);

	PropertyInfo optimize_skin_weights_prop = PropertyInfo(Variant::BOOL, "optimize_skin_weights", PROPERTY_HINT_NONE, "");
	_property_list.push_back(optimize_skin_weights_prop);

	PropertyInfo max_skin_weights_prop = PropertyInfo(Variant::INT, "max_skin_weights_per_vertex", PROPERTY_HINT_RANGE, "1,8,1");
	_property_list.push_back(max_skin_weights_prop);
}

String EditorSceneExporterFBXSettings::get_copyright() const {
	return _copyright;
}

void EditorSceneExporterFBXSettings::set_copyright(const String &p_copyright) {
	_copyright = p_copyright;
}

double EditorSceneExporterFBXSettings::get_bake_fps() const {
	return _bake_fps;
}

void EditorSceneExporterFBXSettings::set_bake_fps(const double p_bake_fps) {
	_bake_fps = p_bake_fps;
}

bool EditorSceneExporterFBXSettings::get_ascii_format() const {
	return _ascii_format;
}

void EditorSceneExporterFBXSettings::set_ascii_format(const bool p_ascii_format) {
	_ascii_format = p_ascii_format;
}

bool EditorSceneExporterFBXSettings::get_embed_textures() const {
	return _embed_textures;
}

void EditorSceneExporterFBXSettings::set_embed_textures(const bool p_embed_textures) {
	_embed_textures = p_embed_textures;
}

bool EditorSceneExporterFBXSettings::get_export_animations() const {
	return _export_animations;
}

void EditorSceneExporterFBXSettings::set_export_animations(const bool p_export_animations) {
	_export_animations = p_export_animations;
}

bool EditorSceneExporterFBXSettings::get_export_materials() const {
	return _export_materials;
}

void EditorSceneExporterFBXSettings::set_export_materials(const bool p_export_materials) {
	_export_materials = p_export_materials;
}

int EditorSceneExporterFBXSettings::get_fbx_version() const {
	return _fbx_version;
}

void EditorSceneExporterFBXSettings::set_fbx_version(const int p_fbx_version) {
	_fbx_version = p_fbx_version;
}

FBXState::CoordinateSystem EditorSceneExporterFBXSettings::get_coordinate_system() const {
	return _coordinate_system;
}

void EditorSceneExporterFBXSettings::set_coordinate_system(const FBXState::CoordinateSystem p_coordinate_system) {
	_coordinate_system = p_coordinate_system;
}

bool EditorSceneExporterFBXSettings::get_export_skinning() const {
	return _export_skinning;
}

void EditorSceneExporterFBXSettings::set_export_skinning(const bool p_export_skinning) {
	_export_skinning = p_export_skinning;
}

bool EditorSceneExporterFBXSettings::get_export_morph_targets() const {
	return _export_morph_targets;
}

void EditorSceneExporterFBXSettings::set_export_morph_targets(const bool p_export_morph_targets) {
	_export_morph_targets = p_export_morph_targets;
}

bool EditorSceneExporterFBXSettings::get_optimize_skin_weights() const {
	return _optimize_skin_weights;
}

void EditorSceneExporterFBXSettings::set_optimize_skin_weights(const bool p_optimize_skin_weights) {
	_optimize_skin_weights = p_optimize_skin_weights;
}

int EditorSceneExporterFBXSettings::get_max_skin_weights_per_vertex() const {
	return _max_skin_weights_per_vertex;
}

void EditorSceneExporterFBXSettings::set_max_skin_weights_per_vertex(const int p_max_skin_weights_per_vertex) {
	_max_skin_weights_per_vertex = p_max_skin_weights_per_vertex;
}

void EditorSceneExporterFBXSettings::_bind_methods() {
	// No static property binding - using dynamic property list only
}
