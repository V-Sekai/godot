/**************************************************************************/
/*  editor_scene_exporter_usd_plugin.cpp                                  */
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

#include "editor_scene_exporter_usd_plugin.h"

#include "editor_blender_utils.h"
#include "editor_import_blend_runner.h"
#include "editor_scene_exporter_gltf_settings.h"

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/os/os.h"
#include "editor/editor_node.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/import/3d/scene_import_settings.h"
#include "editor/inspector/editor_inspector.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/dialogs.h"

String SceneExporterUSDPlugin::get_plugin_name() const {
	return "ConvertUSD";
}

bool SceneExporterUSDPlugin::has_main_screen() const {
	return false;
}

SceneExporterUSDPlugin::SceneExporterUSDPlugin() {
	_gltf_document.instantiate();
	// Set up the file dialog.
	_file_dialog = memnew(EditorFileDialog);
	_file_dialog->connect("file_selected", callable_mp(this, &SceneExporterUSDPlugin::_popup_usd_settings_dialog));
	_file_dialog->set_title(TTR("Export Scene to USD File"));
	_file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	_file_dialog->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	_file_dialog->clear_filters();
	_file_dialog->add_filter("*.usd");
	_file_dialog->add_filter("*.usda");
	_file_dialog->add_filter("*.usdc");
	_file_dialog->add_filter("*.usdz");
	EditorNode::get_singleton()->get_gui_base()->add_child(_file_dialog);
	// Set up the export settings menu.
	_config_dialog = memnew(ConfirmationDialog);
	_config_dialog->set_title(TTRC("Export Settings"));
	EditorNode::get_singleton()->get_gui_base()->add_child(_config_dialog);
	_config_dialog->connect(SceneStringName(confirmed), callable_mp(this, &SceneExporterUSDPlugin::_export_scene_as_usd));

	_export_settings.instantiate();
	_export_settings->generate_property_list(_gltf_document);
	_settings_inspector = memnew(EditorInspector);
	_settings_inspector->set_custom_minimum_size(Size2(350, 300) * EDSCALE);
	_config_dialog->add_child(_settings_inspector);
	// Add a button to the Scene -> Export menu to pop up the settings dialog.
	PopupMenu *menu = get_export_as_menu();
	int idx = menu->get_item_count();
	menu->add_item(TTRC("USD Scene..."));
	menu->set_item_metadata(idx, callable_mp(this, &SceneExporterUSDPlugin::_popup_usd_export_dialog));
}

void SceneExporterUSDPlugin::_popup_usd_settings_dialog(const String &p_selected_path) {
	export_path = p_selected_path;

	Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
	ERR_FAIL_NULL(root);
	// Generate and refresh the export settings.
	_export_settings->generate_property_list(_gltf_document, root);
	_settings_inspector->edit(nullptr);
	_settings_inspector->edit(_export_settings.ptr());
	// Show the config dialog.
	_config_dialog->popup_centered();
}

void SceneExporterUSDPlugin::_popup_usd_export_dialog() {
	Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
	if (!root) {
		EditorNode::get_singleton()->show_warning(TTR("This operation can't be done without a scene."));
		return;
	}
	// Set the file dialog's file name to the scene name.
	String filename = String(root->get_scene_file_path().get_file().get_basename());
	if (filename.is_empty()) {
		filename = root->get_name();
	}
	_file_dialog->set_current_file(filename + String(".usd"));
	// Show the file dialog.
	_file_dialog->popup_file_dialog();
}

void SceneExporterUSDPlugin::_export_scene_as_usd() {
	Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
	ERR_FAIL_NULL(root);

	// Check if Blender is available
	String blender_path = EDITOR_GET("filesystem/import/blender/blender_path");
	if (blender_path.is_empty() || !FileAccess::exists(blender_path)) {
		EditorNode::get_singleton()->show_warning(TTR("Blender 3.0+ is required for USD export. Please configure Blender path in Editor Settings."));
		return;
	}

	// Step 1: Convert Godot scene to GLTF
	List<String> deps;
	Ref<GLTFState> state;
	state.instantiate();
	state->set_copyright(_export_settings->get_copyright());
	int32_t flags = 0;
	flags |= EditorSceneFormatImporter::IMPORT_USE_NAMED_SKIN_BINDS;
	state->set_bake_fps(_export_settings->get_bake_fps());
	Error err = _gltf_document->append_from_scene(root, state, flags);
	if (err != OK) {
		ERR_PRINT(vformat("USD export: Failed to convert scene to GLTF: %s.", itos(err)));
		EditorNode::get_singleton()->show_warning(TTR("Failed to convert scene to GLTF for USD export."));
		return;
	}

	// Step 2: Export GLTF to temporary file
	String temp_gltf_path = OS::get_singleton()->get_cache_path().path_join("usd_export_temp_" + itos(OS::get_singleton()->get_ticks_msec()) + ".gltf");
	err = _gltf_document->write_to_filesystem(state, temp_gltf_path);
	if (err != OK) {
		ERR_PRINT(vformat("USD export: Failed to write temporary GLTF file: %s.", itos(err)));
		EditorNode::get_singleton()->show_warning(TTR("Failed to write temporary GLTF file for USD export."));
		return;
	}

	// Step 3: Use Blender to convert GLTF to USD
	Dictionary export_options;
	Dictionary gltf_import_options;
	Dictionary usd_export_options;

	// GLTF import options (minimal, just import the file)
	String temp_gltf_global = ProjectSettings::get_singleton()->globalize_path(temp_gltf_path);
	// Normalize path to forward slashes for Blender
	temp_gltf_global = temp_gltf_global.replace("\\", "/");
	gltf_import_options["filepath"] = temp_gltf_global;

	// USD export options
	String export_path_global = ProjectSettings::get_singleton()->globalize_path(export_path);
	export_path_global = EditorBlenderUtils::fix_windows_network_share_path(export_path_global);

	// Normalize path to forward slashes for Blender (Blender expects forward slashes on all platforms)
	export_path_global = export_path_global.replace("\\", "/");

	// Ensure the output directory exists
	String export_dir = export_path.get_base_dir();
	if (!export_dir.is_empty()) {
		Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		if (da.is_valid()) {
			da->make_dir_recursive(export_dir);
		}
	}

	// Use only well-documented USD export parameters
	// Reference: https://docs.blender.org/manual/en/latest/files/import_export/usd.html
	usd_export_options["filepath"] = export_path_global;
	usd_export_options["selected_objects_only"] = false;
	usd_export_options["visible_objects_only"] = false;
	usd_export_options["export_animation"] = true;
	usd_export_options["export_hair"] = false;
	usd_export_options["export_uvmaps"] = true;
	usd_export_options["export_normals"] = true;
	usd_export_options["export_materials"] = true;
	usd_export_options["use_instancing"] = true;
	// Note: Some parameters may not exist in all Blender versions
	// Only include essential parameters to avoid errors

	export_options["gltf_import_options"] = gltf_import_options;
	export_options["usd_export_options"] = usd_export_options;

	err = EditorImportBlendRunner::get_singleton()->do_export_usd(export_options);
	if (err != OK) {
		ERR_PRINT(vformat("USD export: Failed to convert GLTF to USD: %s.", itos(err)));
		EditorNode::get_singleton()->show_warning(TTR("Failed to convert GLTF to USD. Check Blender version (3.0+ required)."));
		// Clean up temp file
		if (FileAccess::exists(temp_gltf_path)) {
			Ref<DirAccess> da = DirAccess::open(temp_gltf_path.get_base_dir());
			if (da.is_valid()) {
				da->remove(temp_gltf_path.get_file());
			}
		}
		return;
	}

	// Step 4: Verify the exported file exists
	// Check both the relative path and the globalized path
	bool file_exists = FileAccess::exists(export_path);
	if (!file_exists) {
		// Also check the globalized path in case there's a path format issue
		file_exists = FileAccess::exists(export_path_global);
		if (file_exists) {
			ERR_PRINT(vformat("USD export: File found at globalized path but not relative path. Global: %s, Relative: %s", export_path_global, export_path));
		}
	}
	if (!file_exists) {
		ERR_PRINT(vformat("USD export: Export reported success but file was not created. Expected at: %s (globalized: %s)", export_path, export_path_global));
		EditorNode::get_singleton()->show_warning(vformat(TTR("USD export completed but file was not found at: %s"), export_path));
		// Clean up temp file
		if (FileAccess::exists(temp_gltf_path)) {
			Ref<DirAccess> da = DirAccess::open(temp_gltf_path.get_base_dir());
			if (da.is_valid()) {
				da->remove(temp_gltf_path.get_file());
			}
		}
		return;
	}

	// Step 5: Clean up temporary GLTF file
	if (FileAccess::exists(temp_gltf_path)) {
		Ref<DirAccess> da = DirAccess::open(temp_gltf_path.get_base_dir());
		if (da.is_valid()) {
			da->remove(temp_gltf_path.get_file());
			// Also remove .bin file if it exists
			String temp_bin_path = temp_gltf_path.get_basename() + ".bin";
			if (FileAccess::exists(temp_bin_path)) {
				da->remove(temp_bin_path.get_file());
			}
		}
	}

	EditorFileSystem::get_singleton()->scan_changes();
	// Note: show_warning is used for both warnings and info messages in Godot
	EditorNode::get_singleton()->show_warning(TTR("USD export completed successfully."));
}
