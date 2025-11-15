/**************************************************************************/
/*  editor_scene_importer_usd.cpp                                         */
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

#include "editor_scene_importer_usd.h"

#include "../usd_document.h"
#include "../usd_state.h"

#include "core/config/project_settings.h"

void EditorSceneFormatImporterUSD::get_extensions(List<String> *r_extensions) const {
	r_extensions->push_back("usd");
	r_extensions->push_back("usda");
	r_extensions->push_back("usdc");
}

Node *EditorSceneFormatImporterUSD::import_scene(const String &p_path, uint32_t p_flags,
		const HashMap<StringName, Variant> &p_options,
		List<String> *r_missing_deps, Error *r_err) {
	Ref<USDDocument> usd;
	usd.instantiate();
	Ref<USDState> state;
	state.instantiate();
	print_verbose(vformat("USD path: %s", p_path));
	String path = ProjectSettings::get_singleton()->globalize_path(p_path);
	if (p_options.has("usd/naming_version")) {
		int naming_version = p_options["usd/naming_version"];
		usd->set_naming_version(naming_version);
	}
	if (p_options.has("usd/embedded_image_handling")) {
		const int32_t enum_option = p_options["usd/embedded_image_handling"];
		state->set_handle_binary_image_mode((GLTFState::HandleBinaryImageMode)enum_option);
	}
	if (p_options.has(SNAME("nodes/import_as_skeleton_bones")) ? (bool)p_options[SNAME("nodes/import_as_skeleton_bones")] : false) {
		state->set_import_as_skeleton_bones(true);
	}
	p_flags |= EditorSceneFormatImporter::IMPORT_USE_NAMED_SKIN_BINDS;
	state->set_bake_fps(p_options["animation/fps"]);
	Error err = usd->append_from_file(path, state, p_flags, p_path.get_base_dir());
	if (err != OK) {
		if (r_err) {
			*r_err = FAILED;
		}
		return nullptr;
	}
	return usd->generate_scene(state, state->get_bake_fps(), (bool)p_options["animation/trimming"], false);
}

Variant EditorSceneFormatImporterUSD::get_option_visibility(const String &p_path, const String &p_scene_import_type,
		const String &p_option, const HashMap<StringName, Variant> &p_options) {
	return true;
}

void EditorSceneFormatImporterUSD::get_import_options(const String &p_path,
		List<ResourceImporter::ImportOption> *r_options) {
	// Returns all the options when path is empty because that means it's for the Project Settings.
	String file_extension = p_path.get_extension().to_lower();
	if (p_path.is_empty() || file_extension == "usd" || file_extension == "usda" || file_extension == "usdc") {
		r_options->push_back(ResourceImporterScene::ImportOption(PropertyInfo(Variant::INT, "usd/embedded_image_handling", PROPERTY_HINT_ENUM, "Discard All Textures,Extract Textures,Embed as Basis Universal,Embed as Uncompressed", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), GLTFState::HANDLE_BINARY_IMAGE_MODE_EXTRACT_TEXTURES));
		r_options->push_back(ResourceImporterScene::ImportOption(PropertyInfo(Variant::INT, "usd/naming_version", PROPERTY_HINT_ENUM, "Godot 4.0 or 4.1,Godot 4.2 to 4.4,Godot 4.5 or later"), 2));
	}
}

void EditorSceneFormatImporterUSD::handle_compatibility_options(HashMap<StringName, Variant> &p_import_params) const {
	if (!p_import_params.has("usd/naming_version")) {
		// If a .usd's existing import file is missing the USD
		// naming compatibility version, we need to use version 1.
		// Version 1 is the behavior before this option was added.
		p_import_params["usd/naming_version"] = 1;
	}
}

