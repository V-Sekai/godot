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

#include "../gltf_defines.h"
#include "../gltf_document.h"
#include "editor_blender_utils.h"
#include "editor_import_blend_runner.h"

#include "core/config/project_settings.h"
#include "core/os/os.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "main/main.h"
#include "scene/gui/button.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/main/window.h"
#include "servers/display/display_server.h"

#ifdef WINDOWS_ENABLED
#include <shlwapi.h>
#endif

void EditorSceneFormatImporterUSD::get_extensions(List<String> *r_extensions) const {
	r_extensions->push_back("usd");
	r_extensions->push_back("usda");
	r_extensions->push_back("usdc");
	r_extensions->push_back("usdz");
}

Node *EditorSceneFormatImporterUSD::import_scene(const String &p_path, uint32_t p_flags,
		const HashMap<StringName, Variant> &p_options,
		List<String> *r_missing_deps, Error *r_err) {
	String blender_path = EDITOR_GET("filesystem/import/blender/blender_path");

	ERR_FAIL_COND_V_MSG(blender_path.is_empty(), nullptr, "Blender path is empty, check your Editor Settings.");
	ERR_FAIL_COND_V_MSG(!FileAccess::exists(blender_path), nullptr, vformat("Invalid Blender path: %s, check your Editor Settings.", blender_path));

	if (blender_major_version == -1 || blender_minor_version == -1 || last_tested_blender_path != blender_path) {
		String error;
		if (!EditorBlenderUtils::get_blender_version(blender_path, blender_major_version, blender_minor_version, &error)) {
			ERR_FAIL_V_MSG(nullptr, error);
		}
		last_tested_blender_path = blender_path;
	}

	// Get global paths for source and sink.
	// Escape paths to be valid Python strings to embed in the script.
	String source_global = ProjectSettings::get_singleton()->globalize_path(p_path);
	source_global = EditorBlenderUtils::fix_windows_network_share_path(source_global);

	const String usd_basename = p_path.get_file().get_basename();
	const String sink = ProjectSettings::get_singleton()->get_imported_files_path().path_join(
			vformat("%s-%s.gltf", usd_basename, p_path.md5_text()));
	const String sink_global = ProjectSettings::get_singleton()->globalize_path(sink);

	// Handle configuration options.
	Dictionary request_options;
	Dictionary usd_import_options;
	Dictionary gltf_export_options;

	// USD import options
	// Only include parameters that are documented and supported by Blender's USD import operator
	// Based on testing with Blender 4.5.3, many parameters that seem logical are not actually
	// supported by bpy.ops.wm.usd_import. The Blender Python API documentation doesn't clearly
	// list all supported parameters, so we use only those confirmed to work through testing.
	usd_import_options["filepath"] = source_global;

	// Parameters that appear to be supported (tested):
	// Only add import_subdiv for Blender < 5.0
	if (blender_major_version < 5 && p_options.has(SNAME("usd/import_subdiv"))) {
		usd_import_options["import_subdiv"] = (bool)p_options[SNAME("usd/import_subdiv")];
	}
	if (p_options.has(SNAME("usd/import_usd_preview"))) {
		usd_import_options["import_usd_preview"] = (bool)p_options[SNAME("usd/import_usd_preview")];
	}
	if (p_options.has(SNAME("usd/import_set_frame_range"))) {
		usd_import_options["set_frame_range"] = (bool)p_options[SNAME("usd/import_set_frame_range")];
	}
	if (p_options.has(SNAME("usd/import_materials"))) {
		usd_import_options["import_materials"] = (bool)p_options[SNAME("usd/import_materials")];
	}

	// Parameters confirmed NOT supported by Blender's USD import (tested with Blender 4.5.3):
	// - import_uv_coordinates (TypeError: keyword "import_uv_coordinates" unrecognized)
	// - import_normals (TypeError: keyword "import_normals" unrecognized)
	// - import_colors (likely not supported - not tested but similar parameters fail)
	// - import_cameras (likely not supported - UI option exists but Python param may differ)
	// - import_lights (likely not supported - UI option exists but Python param may differ)
	// - import_skins (likely not supported - UI option exists but Python param may differ)
	//
	// These settings are controlled through the glTF export options instead, which is appropriate
	// since we convert USD -> glTF. The glTF export options handle: normals, colors, cameras, lights, skins.

	// glTF export options (similar to Blender importer)
	gltf_export_options["filepath"] = sink_global;
	gltf_export_options["export_format"] = "GLTF_SEPARATE";
	gltf_export_options["export_yup"] = true;
	gltf_export_options["export_import_convert_lighting_mode"] = "COMPAT";

	if (p_options.has(SNAME("usd/materials/export_materials"))) {
		int32_t exports = p_options[SNAME("usd/materials/export_materials")];
		switch (exports) {
			case USD_MATERIAL_EXPORT_PLACEHOLDER: {
				gltf_export_options["export_materials"] = "PLACEHOLDER";
			} break;
			case USD_MATERIAL_EXPORT_EXPORT: {
				gltf_export_options["export_materials"] = "EXPORT";
			} break;
			case USD_MATERIAL_EXPORT_NAMED_PLACEHOLDER: {
				gltf_export_options["export_materials"] = "EXPORT";
				gltf_export_options["export_image_format"] = "NONE";
			} break;
		}
	} else {
		gltf_export_options["export_materials"] = "PLACEHOLDER";
	}

	if (p_options.has(SNAME("usd/nodes/cameras")) && p_options[SNAME("usd/nodes/cameras")]) {
		gltf_export_options["export_cameras"] = true;
	} else {
		gltf_export_options["export_cameras"] = false;
	}
	if (p_options.has(SNAME("usd/nodes/punctual_lights")) && p_options[SNAME("usd/nodes/punctual_lights")]) {
		gltf_export_options["export_lights"] = true;
	} else {
		gltf_export_options["export_lights"] = false;
	}

	if (p_options.has(SNAME("usd/meshes/skins"))) {
		int32_t skins = p_options["usd/meshes/skins"];
		if (skins == 0) { // NONE
			gltf_export_options["export_skins"] = false;
		} else if (skins == 1) { // COMPATIBLE
			gltf_export_options["export_skins"] = true;
			gltf_export_options["export_all_influences"] = false;
		} else if (skins == 2) { // ALL
			gltf_export_options["export_skins"] = true;
			gltf_export_options["export_all_influences"] = true;
		}
	} else {
		gltf_export_options["export_skins"] = false;
	}

	if (p_options.has(SNAME("usd/meshes/uvs")) && p_options[SNAME("usd/meshes/uvs")]) {
		gltf_export_options["export_texcoords"] = true;
	} else {
		gltf_export_options["export_texcoords"] = false;
	}
	if (p_options.has(SNAME("usd/meshes/normals")) && p_options[SNAME("usd/meshes/normals")]) {
		gltf_export_options["export_normals"] = true;
	} else {
		gltf_export_options["export_normals"] = false;
	}
	// Blender 4.2+ uses export_vertex_color instead of export_colors
	if (blender_major_version > 4 || (blender_major_version == 4 && blender_minor_version >= 2)) {
		if (p_options.has(SNAME("usd/meshes/colors")) && p_options[SNAME("usd/meshes/colors")]) {
			gltf_export_options["export_vertex_color"] = "MATERIAL";
		} else {
			gltf_export_options["export_vertex_color"] = "NONE";
		}
	} else {
		if (p_options.has(SNAME("usd/meshes/colors")) && p_options[SNAME("usd/meshes/colors")]) {
			gltf_export_options["export_colors"] = true;
		} else {
			gltf_export_options["export_colors"] = false;
		}
	}

	request_options["usd_import_options"] = usd_import_options;
	request_options["gltf_options"] = gltf_export_options;

	// Use the Blender runner to execute the import/export
	Error err = EditorImportBlendRunner::get_singleton()->do_import_usd(request_options);
	if (err != OK) {
		if (r_err) {
			*r_err = err;
		}
		return nullptr;
	}

	// Load the generated glTF file
	Ref<GLTFDocument> gltf;
	gltf.instantiate();
	Ref<GLTFState> state;
	state.instantiate();
	state->set_scene_name(usd_basename);
	state->set_extract_path(p_path.get_base_dir());
	state->set_extract_prefix(usd_basename);
	err = gltf->append_from_file(sink.get_basename() + ".gltf", state, p_flags, sink.get_base_dir());
	if (err != OK) {
		if (r_err) {
			*r_err = err;
		}
		return nullptr;
	}

	if (p_options.has("animation/import")) {
		state->set_create_animations(bool(p_options["animation/import"]));
	}

	ERR_FAIL_COND_V(!p_options.has("animation/fps"), nullptr);

#ifndef DISABLE_DEPRECATED
	bool trimming = p_options.has("animation/trimming") ? (bool)p_options["animation/trimming"] : false;
	Node *root = gltf->generate_scene(state, (float)p_options["animation/fps"], trimming, false);
#else
	Node *root = gltf->generate_scene(state, (float)p_options["animation/fps"], (bool)p_options["animation/trimming"], false);
#endif
	if (root) {
		root->set_name(usd_basename);
	}
	return root;
}

void EditorSceneFormatImporterUSD::get_import_options(const String &p_path,
		List<ResourceImporter::ImportOption> *r_options) {
	// USD import options (only parameters that actually work in Blender)
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "usd/import_subdiv"), true));
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "usd/import_usd_preview"), true));
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "usd/import_set_frame_range"), true));
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "usd/import_materials"), true));

	// Note: The following USD import options are NOT supported by Blender's USD import operator:
	// - usd/import_uv_coordinates (causes error)
	// - usd/import_normals (causes error)
	// - usd/import_colors (not supported)
	// - usd/import_cameras (not supported)
	// - usd/import_lights (not supported)
	// - usd/import_skins (not supported)
	// These are controlled via glTF export options below instead.

	// glTF export options (control what gets exported from USD -> glTF conversion)
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::INT, "usd/materials/export_materials", PROPERTY_HINT_ENUM, "Placeholder,Export,Named Placeholder"), USD_MATERIAL_EXPORT_EXPORT));
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "usd/nodes/cameras"), true));
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "usd/nodes/punctual_lights"), true));
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::INT, "usd/meshes/skins", PROPERTY_HINT_ENUM, "None,Compatible,All"), 1));
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "usd/meshes/uvs"), true));
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "usd/meshes/normals"), true));
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "usd/meshes/colors"), true));

	// Animation options (shared with glTF)
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "animation/import"), true));
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::INT, "animation/fps", PROPERTY_HINT_RANGE, "1,120,1"), 30));
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "animation/trimming"), true));
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "animation/remove_immutable_tracks"), true));
}

Variant EditorSceneFormatImporterUSD::get_option_visibility(const String &p_path, const String &p_scene_import_type, const String &p_option,
		const HashMap<StringName, Variant> &p_options) {
	// Check if this is a USD file
	String extension = p_path.get_extension().to_lower();
	bool is_usd_file = (extension == "usd" || extension == "usda" || extension == "usdc" || extension == "usdz");

	if (!is_usd_file) {
		// For non-USD files, hide only USD-specific options
		if (p_option.begins_with("usd/")) {
			return false;
		}
		return true;
	}

	// For USD files, show all options (match FBX importer behavior)
	return true;
}

void EditorSceneFormatImporterUSD::handle_compatibility_options(HashMap<StringName, Variant> &p_import_params) const {
	// Handle any compatibility options if needed
}

static bool _test_blender_path_usd(const String &p_path, String *r_err = nullptr) {
	int major, minor;
	return EditorBlenderUtils::get_blender_version(p_path, major, minor, r_err);
}

void EditorFileSystemImportFormatSupportQueryUSD::_validate_path(String p_path) {
	String error;
	bool success = false;
	if (p_path == "") {
		error = TTR("Path is empty.");
	} else {
		if (_test_blender_path_usd(p_path, &error)) {
			success = true;
			if (auto_detected_path == p_path) {
				error = TTR("Path to Blender executable is valid (Autodetected).");
			} else {
				error = TTR("Path to Blender executable is valid.");
			}
		}
	}

	path_status->set_text(error);

	if (success) {
		path_status->add_theme_color_override(SceneStringName(font_color), path_status->get_theme_color(SNAME("success_color"), EditorStringName(Editor)));
		configure_blender_dialog->get_ok_button()->set_disabled(false);
	} else {
		path_status->add_theme_color_override(SceneStringName(font_color), path_status->get_theme_color(SNAME("error_color"), EditorStringName(Editor)));
		configure_blender_dialog->get_ok_button()->set_disabled(true);
	}
}

bool EditorFileSystemImportFormatSupportQueryUSD::_autodetect_path() {
	// Autodetect - reuse same logic as Blend importer
	auto_detected_path = "";

#if defined(MACOS_ENABLED)
	Vector<String> find_paths = {
		"/opt/homebrew/bin/blender",
		"/opt/local/bin/blender",
		"/usr/local/bin/blender",
		"/usr/local/opt/blender",
		"/Applications/Blender.app/Contents/MacOS/Blender",
	};
	{
		List<String> mdfind_args;
		mdfind_args.push_back("kMDItemCFBundleIdentifier=org.blenderfoundation.blender");

		String output;
		Error err = OS::get_singleton()->execute("mdfind", mdfind_args, &output);
		if (err == OK) {
			for (const String &find_path : output.split("\n")) {
				find_paths.push_back(find_path.path_join("Contents/MacOS/Blender"));
			}
		}
	}
#elif defined(WINDOWS_ENABLED)
	Vector<String> find_paths = {
		"C:\\Program Files\\Blender Foundation\\blender.exe",
		"C:\\Program Files (x86)\\Blender Foundation\\blender.exe",
	};
	{
		char blender_opener_path[MAX_PATH];
		DWORD path_len = MAX_PATH;
		HRESULT res = AssocQueryString(0, ASSOCSTR_EXECUTABLE, ".blend", "open", blender_opener_path, &path_len);
		if (res == S_OK) {
			find_paths.push_back(String(blender_opener_path).get_base_dir().path_join("blender.exe"));
		}
	}

#elif defined(UNIX_ENABLED)
	Vector<String> find_paths = {
		"/usr/bin/blender",
		"/usr/local/bin/blender",
		"/opt/blender/bin/blender",
	};
#endif

	for (const String &find_path : find_paths) {
		if (_test_blender_path_usd(find_path)) {
			auto_detected_path = find_path;
			return true;
		}
	}

	return false;
}

void EditorFileSystemImportFormatSupportQueryUSD::_path_confirmed() {
	confirmed = true;
}

void EditorFileSystemImportFormatSupportQueryUSD::_select_install(String p_path) {
	blender_path->set_text(p_path);
	_validate_path(p_path);
}

void EditorFileSystemImportFormatSupportQueryUSD::_browse_install() {
	if (blender_path->get_text() != String()) {
		browse_dialog->set_current_file(blender_path->get_text());
	}

	browse_dialog->popup_centered_ratio();
}

void EditorFileSystemImportFormatSupportQueryUSD::_update_icons() {
	blender_path_browse->set_button_icon(blender_path_browse->get_editor_theme_icon(SNAME("FolderBrowse")));
}

bool EditorFileSystemImportFormatSupportQueryUSD::is_active() const {
	bool blend_enabled = GLOBAL_GET_CACHED(bool, "filesystem/import/blender/enabled");

	if (blend_enabled && !_test_blender_path_usd(EDITOR_GET("filesystem/import/blender/blender_path").operator String())) {
		// Intending to import USD, but Blender not configured.
		return true;
	}

	return false;
}

Vector<String> EditorFileSystemImportFormatSupportQueryUSD::get_file_extensions() const {
	Vector<String> ret;
	ret.push_back("usd");
	ret.push_back("usda");
	ret.push_back("usdc");
	ret.push_back("usdz");
	return ret;
}

bool EditorFileSystemImportFormatSupportQueryUSD::query() {
	ERR_FAIL_COND_V_MSG(DisplayServer::get_singleton()->get_name() == "headless", true, "Blender path is invalid or not set, check your Editor Settings. Cannot configure blender path in headless mode.");

	if (!configure_blender_dialog) {
		configure_blender_dialog = memnew(ConfirmationDialog);
		configure_blender_dialog->set_title(TTR("Configure Blender for USD Import"));
		configure_blender_dialog->set_flag(Window::FLAG_BORDERLESS, true); // Avoid closing accidentally.
		configure_blender_dialog->set_close_on_escape(false);

		String select_exec_label = TTR("Blender 3.0+ is required to import USD files.\nPlease provide a valid path to a Blender executable.");
#ifdef MACOS_ENABLED
		select_exec_label += "\n" + TTR("On macOS, this should be the `Contents/MacOS/blender` file within the Blender `.app` folder.");
#endif
		VBoxContainer *vb = memnew(VBoxContainer);
		vb->add_child(memnew(Label(select_exec_label)));

		HBoxContainer *hb = memnew(HBoxContainer);

		blender_path = memnew(LineEdit);
		blender_path->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		blender_path->set_accessibility_name(TTRC("Path"));
		hb->add_child(blender_path);

		blender_path_browse = memnew(Button);
		blender_path_browse->set_text(TTR("Browse"));
		blender_path_browse->connect(SceneStringName(pressed), callable_mp(this, &EditorFileSystemImportFormatSupportQueryUSD::_browse_install));
		hb->add_child(blender_path_browse);

		hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		hb->set_custom_minimum_size(Size2(400 * EDSCALE, 0));

		vb->add_child(hb);

		path_status = memnew(Label);
		path_status->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
		vb->add_child(path_status);

		configure_blender_dialog->add_child(vb);

		blender_path->connect(SceneStringName(text_changed), callable_mp(this, &EditorFileSystemImportFormatSupportQueryUSD::_validate_path));

		EditorNode::get_singleton()->get_gui_base()->add_child(configure_blender_dialog);

		configure_blender_dialog->set_ok_button_text(TTR("Confirm Path"));
		configure_blender_dialog->set_cancel_button_text(TTR("Disable USD Import"));
		configure_blender_dialog->get_cancel_button()->set_tooltip_text(TTR("Disables USD files import for this project. Can be re-enabled in Project Settings."));
		configure_blender_dialog->connect(SceneStringName(confirmed), callable_mp(this, &EditorFileSystemImportFormatSupportQueryUSD::_path_confirmed));

		browse_dialog = memnew(EditorFileDialog);
		browse_dialog->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
		browse_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
		browse_dialog->connect("file_selected", callable_mp(this, &EditorFileSystemImportFormatSupportQueryUSD::_select_install));

		EditorNode::get_singleton()->get_gui_base()->add_child(browse_dialog);

		// Update icons.
		callable_mp(this, &EditorFileSystemImportFormatSupportQueryUSD::_update_icons).call_deferred();
	}

	String path = EDITOR_GET("filesystem/import/blender/blender_path");

	if (path.is_empty() && _autodetect_path()) {
		path = auto_detected_path;
	}

	blender_path->set_text(path);

	_validate_path(path);

	configure_blender_dialog->popup_centered();
	confirmed = false;

	while (true) {
		DisplayServer::get_singleton()->process_events();
		Main::iteration();
		if (!configure_blender_dialog->is_visible() || confirmed) {
			break;
		}
	}

	if (confirmed) {
		// Can only confirm a valid path.
		EditorSettings::get_singleton()->set("filesystem/import/blender/blender_path", blender_path->get_text());
		EditorSettings::get_singleton()->save();
	} else {
		// USD import uses the same Blender setting, so we don't disable it
		// Just return false to indicate configuration was cancelled
		return false;
	}

	return false;
}
