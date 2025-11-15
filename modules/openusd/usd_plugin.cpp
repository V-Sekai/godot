/**************************************************************************/
/*  usd_plugin.cpp                                                        */
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

#include "usd_plugin.h"
#include "usd_document.h"
#include "usd_import_state.h"
#include "usd_export_settings.h"

#include "core/io/resource_saver.h"
#include "editor/editor_interface.h"
#include "scene/3d/mesh_instance_3d.h"
#include "core/config/project_settings.h"
#include "scene/resources/mesh.h"
#include "scene/resources/packed_scene.h"
#include "scene/gui/popup.h"

// For setenv
#if defined(_WIN32)
#include <stdlib.h>
#define setenv(name, value, overwrite) _putenv_s(name, value)
#else
#include <stdlib.h>
#endif

void USDPlugin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_on_hello_button_pressed"), &USDPlugin::_on_hello_button_pressed);
	ClassDB::bind_method(D_METHOD("_popup_usd_export_dialog"), &USDPlugin::_popup_usd_export_dialog);
	ClassDB::bind_method(D_METHOD("_export_scene_as_usd", "file_path"), &USDPlugin::_export_scene_as_usd);
	ClassDB::bind_method(D_METHOD("_popup_usd_import_dialog"), &USDPlugin::_popup_usd_import_dialog);
	ClassDB::bind_method(D_METHOD("_import_usd_file", "file_path"), &USDPlugin::_import_usd_file);
}

USDPlugin::USDPlugin() {
	// Constructor
	_usd_document.instantiate();
	_export_settings.instantiate();

	// Generate the property list for the export settings
	_export_settings->generate_property_list(_usd_document);

	// Set the PXR_PLUGINPATH_NAME environment variable to include the Godot scene's root directory
	// We'll set this in _enter_tree when we have access to the editor interface
}

USDPlugin::~USDPlugin() {
	// Destructor
	if (_file_dialog) {
		_file_dialog->queue_free();
		_file_dialog = nullptr;
	}

	if (_settings_inspector) {
		_settings_inspector->queue_free();
		_settings_inspector = nullptr;
	}

	if (_import_file_dialog) {
		_import_file_dialog->queue_free();
		_import_file_dialog = nullptr;
	}
}

void USDPlugin::_on_hello_button_pressed() {
	print_line("Hello USD button pressed!");

	// Get the editor interface
	EditorInterface *editor = EditorInterface::get_singleton();
	if (!editor) {
		ERR_PRINT("Failed to get EditorInterface singleton");
		return;
	}

	// Get the edited scene root
	Node *edited_scene_root = editor->get_edited_scene_root();
	if (!edited_scene_root) {
		ERR_PRINT("No scene is currently being edited");
		return;
	}

	// TODO: Add example USD functionality here
	print_line("Hello USD button pressed!");
}

void USDPlugin::_popup_usd_export_dialog() {
	// This method is called when the user selects "USD Scene..." from the Scene -> Export menu

	// Get the editor interface
	EditorInterface *editor = EditorInterface::get_singleton();
	if (!editor) {
		ERR_PRINT("USD Export: Failed to get EditorInterface singleton");
		return;
	}

	// Get the edited scene root
	Node *edited_scene_root = editor->get_edited_scene_root();
	if (!edited_scene_root) {
		print_line("USD Export: No scene is currently being edited");
		editor->get_base_control()->add_child(_file_dialog);
		editor->get_base_control()->remove_child(_file_dialog);
		return;
	}

	// Set the file dialog's file name to the scene name
	String filename = String(edited_scene_root->get_scene_file_path().get_file().get_basename());
	if (filename.is_empty()) {
		filename = edited_scene_root->get_name();
	}
	_file_dialog->set_current_file(filename + String(".usd"));

	// Generate and refresh the export settings
	_export_settings->generate_property_list(_usd_document, edited_scene_root);

	// Note: In a full implementation, we would update the inspector to show the export settings
	// But for now, we'll just print the settings
	print_line("USD Export: Using export settings:");
	print_line("  - Export materials: ", _export_settings->get_export_materials());
	print_line("  - Export animations: ", _export_settings->get_export_animations());
	print_line("  - Export cameras: ", _export_settings->get_export_cameras());
	print_line("  - Export lights: ", _export_settings->get_export_lights());
	print_line("  - Export meshes: ", _export_settings->get_export_meshes());
	print_line("  - Export textures: ", _export_settings->get_export_textures());
	print_line("  - Copyright: ", _export_settings->get_copyright());
	print_line("  - Bake FPS: ", _export_settings->get_bake_fps());
	print_line("  - Use binary format: ", _export_settings->get_use_binary_format());
	print_line("  - Flatten stage: ", _export_settings->get_flatten_stage());
	print_line("  - Export as single layer: ", _export_settings->get_export_as_single_layer());
	print_line("  - Export with references: ", _export_settings->get_export_with_references());

	// Show the file dialog
	_file_dialog->popup_centered_ratio();

	print_line("USD Export: Showing export dialog");
}

void USDPlugin::_export_scene_as_usd(const String &p_file_path) {
	// This method is called when the user selects a file path in the export dialog

	// Get the editor interface
	EditorInterface *editor = EditorInterface::get_singleton();
	if (!editor) {
		ERR_PRINT("USD Export: Failed to get EditorInterface singleton");
		return;
	}

	// Get the edited scene root
	Node *edited_scene_root = editor->get_edited_scene_root();
	if (!edited_scene_root) {
		ERR_PRINT("USD Export: No scene is currently being edited");
		return;
	}

	print_line("USD Export: Exporting scene to ", p_file_path);

	// Create a new USD state
	Ref<USDState> state;
	state.instantiate();
	// TODO: Add copyright and bake_fps to USDState if needed for export
	// For now, export functionality is stubbed out

	// Export the scene
	Error err = _usd_document->export_from_scene(edited_scene_root, state);
	if (err != OK) {
		ERR_PRINT("USD Export: Failed to export scene to USD document");
		return;
	}

	// Write the USD file
	err = _usd_document->write_to_filesystem(state, p_file_path);
	if (err != OK) {
		ERR_PRINT("USD Export: Failed to write USD document to filesystem");
		return;
	}

	print_line("USD Export: Successfully exported scene to ", p_file_path);
}

void USDPlugin::_popup_usd_import_dialog() {
	// This method is called when the user selects "USD Scene..." from the File -> Import menu

	// Get the editor interface
	EditorInterface *editor = EditorInterface::get_singleton();
	if (!editor) {
		ERR_PRINT("USD Import: Failed to get EditorInterface singleton");
		return;
	}

	// Show the file dialog
	_import_file_dialog->popup_centered_ratio();

	print_line("USD Import: Showing import dialog");
}

void USDPlugin::_import_usd_file(const String &p_file_path) {
	// This method is called when the user selects a file path in the import dialog

	// Get the editor interface
	EditorInterface *editor = EditorInterface::get_singleton();
	if (!editor) {
		ERR_PRINT("USD Import: Failed to get EditorInterface singleton");
		return;
	}

	// TODO: This old import method is deprecated - use EditorSceneFormatImporterUSD instead
	// The GLTF-based import system handles USD import through the editor importer
	ERR_PRINT("USD Import: _import_usd_file is deprecated - use EditorSceneFormatImporterUSD (GLTF-based import system)");
	return;
}

// Helper method to extract and apply transform from a USD prim to a Godot node
// TODO: Update to use TinyUSDZ API
bool USDPlugin::_apply_transform_from_usd_prim(const tinyusdz::Prim &p_prim, Node3D *p_node) {
	ERR_PRINT("USD Plugin: _apply_transform_from_usd_prim not yet implemented with TinyUSDZ");
	return false;
}

// Helper method to convert a USD prim to a Godot node
// TODO: Update to use TinyUSDZ API
Node *USDPlugin::_convert_prim_to_node(const tinyusdz::Prim &p_prim, Node *p_parent, Node *p_scene_root) {
	ERR_PRINT("USD Plugin: _convert_prim_to_node not yet implemented with TinyUSDZ");
	return nullptr;
}

// Helper method to print the prim hierarchy
// TODO: Update to use TinyUSDZ API
void USDPlugin::_print_prim_hierarchy(const tinyusdz::Prim &p_prim, int p_indent) {
	ERR_PRINT("USD Plugin: _print_prim_hierarchy not yet implemented with TinyUSDZ");
}

// Helper method to print the node hierarchy
void USDPlugin::_print_node_hierarchy(Node *p_node, int p_indent) {
	if (!p_node) {
		return;
	}

	// Create an indentation string
	String indent = "";
	for (int i = 0; i < p_indent; i++) {
		indent += "  ";
	}

	// Print the node name, class, and owner
	String owner_info = p_node->get_owner() ? String(" (owner: ") + p_node->get_owner()->get_name() + ")" : " (no owner)";
	print_line(indent, "- ", p_node->get_name(), " [", p_node->get_class(), "]", owner_info);

	// Recursively print children
	for (int i = 0; i < p_node->get_child_count(); i++) {
		_print_node_hierarchy(p_node->get_child(i), p_indent + 1);
	}
}

bool USDPlugin::has_main_screen() const {
	// Return true if the plugin needs a main screen
	return false;
}

String USDPlugin::get_plugin_name() const {
	// Return the name of the plugin
	return "USD";
}

void USDPlugin::_notification(int p_what) {
    switch (p_what) {
        case NOTIFICATION_ENTER_TREE: {
            print_line("USD Plugin: Enter Tree");

            // Get the project root directory
            String project_root = ProjectSettings::get_singleton()->globalize_path("res://");
            if (!project_root.ends_with("/")) {
                project_root += "/";
            }

            // TODO: Set up TinyUSDZ plugin paths if needed
            // TinyUSDZ doesn't use the same plugin system as OpenUSD

            // Create the "Hello USD" button
            hello_button = memnew(Button);
            hello_button->set_text("Hello USD");
            hello_button->set_tooltip_text("Create a 'Hello USD' text node in the scene");
            hello_button->connect("pressed", Callable(this, "_on_hello_button_pressed"));
            add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, hello_button);
            print_line("USD Plugin: Added Hello USD button to toolbar");

            // Add the USD export menu item to the Scene -> Export menu
            PopupMenu *export_menu = get_export_as_menu();
            if (export_menu) {
                int idx = export_menu->get_item_count();
                export_menu->add_item("USD Scene...");
                export_menu->set_item_metadata(idx, Callable(this, "_popup_usd_export_dialog"));
                print_line("USD Plugin: Added USD Scene export menu item");
            } else {
                ERR_PRINT("USD Plugin: Failed to get export menu");
            }

            // Create and add the "Import USD" button to the toolbar
            Button *import_button = memnew(Button);
            import_button->set_text("Import USD");
            import_button->set_tooltip_text("Import a USD file into the current scene");
            import_button->connect("pressed", Callable(this, "_popup_usd_import_dialog"));
            add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, import_button);
            print_line("USD Plugin: Added Import USD button to toolbar");

            // Set up the export file dialog if not already created
            if (!_file_dialog) {
                _file_dialog = memnew(EditorFileDialog);
                _file_dialog->connect("file_selected", Callable(this, "_export_scene_as_usd"));
                _file_dialog->set_title("Export Scene to USD");
                _file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
                _file_dialog->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
                _file_dialog->clear_filters();
                _file_dialog->add_filter("*.usd");
                _file_dialog->add_filter("*.usda");
                _file_dialog->add_filter("*.usdc");

                // Set up the export settings pane
                _settings_inspector = memnew(EditorInspector);
                _settings_inspector->set_custom_minimum_size(Size2(350, 300));
                _file_dialog->add_side_menu(_settings_inspector, "Export Settings:");
            }

            // Set up the import file dialog if not already created
            if (!_import_file_dialog) {
                _import_file_dialog = memnew(EditorFileDialog);
                _import_file_dialog->connect("file_selected", Callable(this, "_import_usd_file"));
                _import_file_dialog->set_title("Import USD Scene");
                _import_file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
                _import_file_dialog->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
                _import_file_dialog->clear_filters();
                _import_file_dialog->add_filter("*.usd");
                _import_file_dialog->add_filter("*.usda");
                _import_file_dialog->add_filter("*.usdc");
            }

            // Add the file dialogs to the editor's base control
            EditorInterface *editor = EditorInterface::get_singleton();
            if (editor) {
                editor->get_base_control()->add_child(_file_dialog);
                editor->get_base_control()->add_child(_import_file_dialog);
            }
            break;
        }
        case NOTIFICATION_EXIT_TREE: {
            // Clean up the "Hello USD" button
            if (hello_button) {
                hello_button->queue_free();
                hello_button = nullptr;
            }

            // Remove the export menu item
            PopupMenu *export_menu = get_export_as_menu();
            if (export_menu) {
                for (int i = 0; i < export_menu->get_item_count(); i++) {
                    Variant metadata = export_menu->get_item_metadata(i);
                    if (metadata.get_type() == Variant::CALLABLE) {
                        Callable callable = metadata;
                        if (callable.get_object() == this) {
                            export_menu->remove_item(i);
                            break;
                        }
                    }
                }
            }
            print_line("USD Plugin: Exit Tree");
            break;
        }
        default:
            break;
    }
}
