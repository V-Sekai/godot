/**************************************************************************/
/*  editor_scene_exporter_blend_plugin.cpp                                */
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

#include "editor_scene_exporter_blend_plugin.h"

#ifdef TOOLS_ENABLED

#include "core/config/project_settings.h"
#include "editor/editor_file_system.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/import/3d/resource_importer_scene.h"
#include "modules/gltf/gltf_document.h"
#include "scene/gui/popup_menu.h"

String SceneExporterBlendPlugin::get_name() const {
	return "ConvertGLTF2";
}

bool SceneExporterBlendPlugin::has_main_screen() const {
	return false;
}

SceneExporterBlendPlugin::SceneExporterBlendPlugin() {
	file_export_lib = memnew(EditorFileDialog);
	EditorNode::get_singleton()->get_gui_base()->add_child(file_export_lib);
	file_export_lib->connect("file_selected", callable_mp(this, &SceneExporterBlendPlugin::_gltf2_dialog_action));
	file_export_lib->set_title(TTR("Export Library"));
	file_export_lib->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	file_export_lib->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	file_export_lib->clear_filters();
	file_export_lib->add_filter("*.blend");
	file_export_lib->set_title(TTR("Export Scene to Blend File"));

	PopupMenu *menu = get_export_as_menu();
	int idx = menu->get_item_count();
	menu->add_item(TTR("Blend Scene..."));
	menu->set_item_metadata(idx, callable_mp(this, &SceneExporterBlendPlugin::convert_scene_to_blend));
}

void SceneExporterBlendPlugin::convert_scene_to_blend() {
	Node *root = get_tree()->get_edited_scene_root();
	if (!root) {
		EditorNode::get_singleton()->show_accept(TTR("This operation can't be done without a scene."), TTR("OK"));
		return;
	}
	String filename = String(root->get_scene_file_path().get_file().get_basename());
	if (filename.is_empty()) {
		filename = root->get_name();
	}
	file_export_lib->set_current_file(filename + String(".blend"));
	file_export_lib->popup_centered_ratio();
}

void SceneExporterBlendPlugin::_gltf2_dialog_action(String p_file) {
	Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
	if (!root) {
		EditorNode::get_singleton()->show_accept(TTR("This operation can't be done without a scene."), TTR("OK"));
		return;
	}
	List<String> deps;
	Ref<GLTFDocument> doc;
	doc.instantiate();
	String blend_to_scene_setting = "import bpy, os, sys;bpy.ops.wm.read_homefile(use_empty=True);bpy.context.scene.render.fps=30;bpy.ops.import_scene.gltf(filepath='GODOT_SOURCE', bone_heuristic='BLENDER');bpy.ops.wm.save_mainfile(filepath='GODOT_SINK');";
	String blender_script = blend_to_scene_setting;
	String source = "res://.godot/imported/" + p_file.get_file().get_basename().to_camel_case() + "-" + p_file.md5_text() + ".glb";
	Ref<GLTFState> gltf_state;
	gltf_state.instantiate();
	int32_t flags = EditorSceneFormatImporter::IMPORT_USE_NAMED_SKIN_BINDS;
	Error err = doc->append_from_scene(root, gltf_state, flags);
	if (err != OK) {
		ERR_PRINT(vformat("Append from scene error %s.", itos(err)));
	}
	err = doc->write_to_filesystem(gltf_state, source);
	if (err != OK) {
		ERR_PRINT(vformat("Blend save gltf scene error %s.", itos(err)));
	}
	String source_global = ProjectSettings::get_singleton()->globalize_path(source);
	source_global = source_global.c_escape();
	blender_script = blender_script.replace("GODOT_SOURCE", source_global);
	String sink = ProjectSettings::get_singleton()->globalize_path(p_file);
	String sink_global = ProjectSettings::get_singleton()->globalize_path(sink);
	sink_global = sink_global.c_escape();
	blender_script = blender_script.replace("GODOT_SINK", sink_global);
	String standard_out;
	List<String> args;
	args.push_back("--background");
	args.push_back("--python-expr");
	print_line(blender_script);
	args.push_back(blender_script);
	String addon_path;
	if (OS::get_singleton()->get_name() == "Windows") {
		addon_path = EditorSettings::get_singleton()->get_setting("filesystem/import/blender/blender3_path");
		addon_path += "/blender.exe";
	} else {
		addon_path = EditorSettings::get_singleton()->get_setting("filesystem/import/blender/blender3_path");
		addon_path += "/blender";
	}
	int32_t ret = OS::get_singleton()->execute(addon_path, args, &standard_out);
	if (ret != OK) {
		print_line(standard_out);
		print_error("Blender returned " + itos(ret));
		return;
	}
	const Vector<String> p_files = { sink_global };
	EditorFileSystem::get_singleton()->reimport_files(p_files);
}

#endif // TOOLS_ENABLED
