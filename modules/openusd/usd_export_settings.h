/**************************************************************************/
/*  usd_export_settings.h                                                 */
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

#include "core/io/resource.h"

class UsdDocument;

class UsdExportSettings : public Resource {
	GDCLASS(UsdExportSettings, Resource);

private:
	// Export options
	bool _export_materials;
	bool _export_animations;
	bool _export_cameras;
	bool _export_lights;
	bool _export_meshes;
	bool _export_textures;
	String _copyright;
	float _bake_fps;

	// USD-specific options
	bool _use_binary_format; // .usdc vs .usda
	bool _flatten_stage;
	bool _export_as_single_layer;
	bool _export_with_references;

protected:
	static void _bind_methods();

public:
	UsdExportSettings();

	// Getters and setters
	void set_export_materials(bool p_enabled);
	bool get_export_materials() const;

	void set_export_animations(bool p_enabled);
	bool get_export_animations() const;

	void set_export_cameras(bool p_enabled);
	bool get_export_cameras() const;

	void set_export_lights(bool p_enabled);
	bool get_export_lights() const;

	void set_export_meshes(bool p_enabled);
	bool get_export_meshes() const;

	void set_export_textures(bool p_enabled);
	bool get_export_textures() const;

	void set_copyright(const String &p_copyright);
	String get_copyright() const;

	void set_bake_fps(float p_fps);
	float get_bake_fps() const;

	void set_use_binary_format(bool p_enabled);
	bool get_use_binary_format() const;

	void set_flatten_stage(bool p_enabled);
	bool get_flatten_stage() const;

	void set_export_as_single_layer(bool p_enabled);
	bool get_export_as_single_layer() const;

	void set_export_with_references(bool p_enabled);
	bool get_export_with_references() const;

	// Generate property list for the inspector
	void generate_property_list(Ref<UsdDocument> p_document, Node *p_root = nullptr);
};
