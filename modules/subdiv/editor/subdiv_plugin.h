/**************************************************************************/
/*  subdiv_plugin.h                                                       */
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

#ifndef SUBDIV_PLUGIN_H
#define SUBDIV_PLUGIN_H

#include "editor/import/editor_import_plugin.h"
#include "editor/plugins/editor_plugin.h"
#include "modules/gltf/gltf_document.h"

#include "subdiv_converter.h"

class GodotSubdivImporter;

class SubdivEditorPlugin : public EditorPlugin {
	GDCLASS(SubdivEditorPlugin, EditorPlugin);

protected:
	Ref<GodotSubdivImporter> resource_import_plugin;

public:
	SubdivEditorPlugin() {
		resource_import_plugin.instantiate();
		add_import_plugin(resource_import_plugin);
	}
};

class GodotSubdivImporter : public EditorImportPlugin {
	GDCLASS(GodotSubdivImporter, EditorImportPlugin);

protected:
	static void _bind_methods();

public:
	String get_importer_name() const override;
	String get_visible_name() const override;
	virtual void get_recognized_extensions(List<String> *p_extensions) const override;

	String get_save_extension() const override;
	String get_resource_type() const override;
	int get_preset_count() const override;
	String get_preset_name(int preset) const override;
	float get_priority() const override;
	int get_import_order() const override;
	virtual bool get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const override;
	virtual void get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const override;

	virtual Error import(ResourceUID::ID p_source_id, const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata = nullptr) override;
	GodotSubdivImporter();
};

#endif // SUBDIV_PLUGIN_H
