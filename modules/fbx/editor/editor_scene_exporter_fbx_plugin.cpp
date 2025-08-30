/**************************************************************************/
/*  editor_scene_exporter_fbx_plugin.cpp                                  */
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

#include "editor_scene_exporter_fbx_plugin.h"

#include "editor_scene_exporter_fbx_settings.h"

#include "editor/editor_node.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/import/3d/resource_importer_scene.h"
#include "editor/import/3d/scene_import_settings.h"
#include "editor/inspector/editor_inspector.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/popup_menu.h"

String SceneExporterFBXPlugin::get_plugin_name() const {
	return "ConvertFBX";
}

bool SceneExporterFBXPlugin::has_main_screen() const {
	return false;
}

SceneExporterFBXPlugin::SceneExporterFBXPlugin() {
	_fbx_document.instantiate();
	// Set up the file dialog.
	_file_dialog = memnew(EditorFileDialog);
	_file_dialog->connect("file_selected", callable_mp(this, &SceneExporterFBXPlugin::_export_scene_as_fbx));
	_file_dialog->set_title(TTR("Export Library"));
	_file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	_file_dialog->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	_file_dialog->clear_filters();
	_file_dialog->add_filter("*.fbx");
	_file_dialog->set_title(TTR("Export Scene to FBX File"));
	EditorNode::get_singleton()->get_gui_base()->add_child(_file_dialog);
	// Set up the export settings menu.
	_export_settings.instantiate();
	_export_settings->generate_property_list(_fbx_document);
	_settings_inspector = memnew(EditorInspector);
	_settings_inspector->set_custom_minimum_size(Size2(350, 300) * EDSCALE);
	_file_dialog->add_side_menu(_settings_inspector, TTR("Export Settings:"));
	// Add a button to the Scene -> Export menu to pop up the settings dialog.
	PopupMenu *menu = get_export_as_menu();
	int idx = menu->get_item_count();
	menu->add_item(TTRC("FBX Scene..."));
	menu->set_item_metadata(idx, callable_mp(this, &SceneExporterFBXPlugin::_popup_fbx_export_dialog));
}

void SceneExporterFBXPlugin::_popup_fbx_export_dialog() {
	Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
	if (!root) {
		EditorNode::get_singleton()->show_accept(TTR("This operation can't be done without a scene."), TTR("OK"));
		return;
	}
	// Set the file dialog's file name to the scene name.
	String filename = String(root->get_scene_file_path().get_file().get_basename());
	if (filename.is_empty()) {
		filename = root->get_name();
	}
	_file_dialog->set_current_file(filename + String(".fbx"));
	// Generate and refresh the export settings.
	_export_settings->generate_property_list(_fbx_document, root);
	_settings_inspector->edit(nullptr);
	_settings_inspector->edit(_export_settings.ptr());
	// Show the file dialog.
	_file_dialog->popup_centered_ratio();
}

void SceneExporterFBXPlugin::_export_scene_as_fbx(const String &p_file_path) {
	Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
	if (!root) {
		EditorNode::get_singleton()->show_accept(TTR("This operation can't be done without a scene."), TTR("OK"));
		return;
	}

	// Create FBX state and configure export settings
	Ref<FBXState> state;
	state.instantiate();
	state->set_copyright(_export_settings->get_copyright());
	state->set_bake_fps(_export_settings->get_bake_fps());

	// Configure export options in FBXState using proper API
	state->set_export_ascii_format(_export_settings->get_ascii_format());
	state->set_export_embed_textures(_export_settings->get_embed_textures());
	state->set_export_animations(_export_settings->get_export_animations());
	state->set_export_materials(_export_settings->get_export_materials());
	state->set_export_fbx_version(_export_settings->get_fbx_version());
	state->set_export_coordinate_system(_export_settings->get_coordinate_system());

	// Set up export flags
	int32_t flags = 0;
	flags |= EditorSceneFormatImporter::IMPORT_USE_NAMED_SKIN_BINDS;

	// Convert scene to FBX format
	Error err = _fbx_document->append_from_scene(root, state, flags);
	if (err != OK) {
		ERR_PRINT(vformat("FBX save scene error %s.", itos(err)));
		EditorNode::get_singleton()->show_accept(TTR("Failed to convert scene to FBX format."), TTR("OK"));
		return;
	}

	// Export to file using existing write_to_filesystem method
	err = _fbx_document->write_to_filesystem(state, p_file_path);
	if (err != OK) {
		ERR_PRINT(vformat("FBX save scene error %s.", itos(err)));
		EditorNode::get_singleton()->show_accept(TTR("Failed to write FBX file."), TTR("OK"));
		return;
	}

	// Refresh file system
	EditorFileSystem::get_singleton()->scan_changes();

	// Show success message
	EditorNode::get_singleton()->show_accept(TTR("FBX export completed successfully."), TTR("OK"));
}
