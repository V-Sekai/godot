/**************************************************************************/
/*  editor_scene_exporter_fbx_settings.h                                  */
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

#include "../fbx_document.h"

class EditorSceneExporterFBXSettings : public RefCounted {
	GDCLASS(EditorSceneExporterFBXSettings, RefCounted);
	List<PropertyInfo> _property_list;
	Ref<FBXDocument> _document;

	String _copyright;
	double _bake_fps = 30.0;
	bool _ascii_format = false;
	bool _embed_textures = true;
	bool _export_animations = true;
	bool _export_materials = true;
	int _fbx_version = 7400; // FBX 7.4
	FBXState::CoordinateSystem _coordinate_system = FBXState::COORDINATE_SYSTEM_Y_UP;

protected:
	static void _bind_methods();
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	void generate_property_list(Ref<FBXDocument> p_document, Node *p_root = nullptr);

	String get_copyright() const;
	void set_copyright(const String &p_copyright);

	double get_bake_fps() const;
	void set_bake_fps(const double p_bake_fps);

	bool get_ascii_format() const;
	void set_ascii_format(const bool p_ascii_format);

	bool get_embed_textures() const;
	void set_embed_textures(const bool p_embed_textures);

	bool get_export_animations() const;
	void set_export_animations(const bool p_export_animations);

	bool get_export_materials() const;
	void set_export_materials(const bool p_export_materials);

	int get_fbx_version() const;
	void set_fbx_version(const int p_fbx_version);

	FBXState::CoordinateSystem get_coordinate_system() const;
	void set_coordinate_system(const FBXState::CoordinateSystem p_coordinate_system);
};
