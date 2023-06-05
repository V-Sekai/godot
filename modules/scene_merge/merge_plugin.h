#pragma once

#ifdef TOOLS_ENABLED

#include "editor/editor_file_dialog.h"
#include "editor/editor_file_system.h"
#include "editor/editor_node.h"
#include "editor/editor_plugin.h"

#include "scene/resources/packed_scene.h"

#include "merge.h"

class SceneMergePlugin : public EditorPlugin {

	GDCLASS(SceneMergePlugin, EditorPlugin);
	CheckBox *file_export_lib_merge = memnew(CheckBox);
	EditorFileDialog *file_export_lib = memnew(EditorFileDialog);
	Ref<SceneMerge> scene_optimize;
	void _dialog_action(String p_file);
	void merge();

protected:
	static void _bind_methods();

public:
	SceneMergePlugin();
	~SceneMergePlugin() {
		EditorNode::get_singleton()->remove_tool_menu_item("Merge Scene");
	}
	void _notification(int notification);
};

void SceneMergePlugin::merge() {
	file_export_lib_merge->set_pressed(false);
	List<String> extensions;
	extensions.push_back("tscn");
	extensions.push_back("scn");
	file_export_lib->clear_filters();
	for (int extension_i = 0; extension_i < extensions.size(); extension_i++) {
		file_export_lib->add_filter("*." + extensions[extension_i] + " ; " + extensions[extension_i].to_upper());
	}
	file_export_lib->popup_centered_ratio();
	Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
	ERR_FAIL_NULL(root);
	String filename = String(root->get_scene_file_path().get_file().get_basename());
	if (filename.is_empty()) {
		filename = root->get_name();
	}
	file_export_lib->set_current_file(filename + String(".scn"));
}

void SceneMergePlugin::_dialog_action(String p_file) {
	Node *node = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
	if (!node) {
		EditorNode::get_singleton()->show_accept(TTR("This operation can't be done without a scene."), TTR("OK"));
		return;
	}
	if (FileAccess::exists(p_file) && file_export_lib_merge->is_pressed()) {
		Ref<PackedScene> scene = ResourceLoader::load(p_file, "PackedScene");
		if (scene.is_null()) {
			EditorNode::get_singleton()->show_accept(TTR("Can't load scene for merging!"), TTR("OK"));
			return;
		} else {
			node->add_child(scene->instantiate(), true);
		}
	}
	scene_optimize->merge(p_file, node);
	EditorFileSystem::get_singleton()->scan_changes();
}

void SceneMergePlugin::_bind_methods() {
	ClassDB::bind_method("_dialog_action", &SceneMergePlugin::_dialog_action);
	ClassDB::bind_method(D_METHOD("merge"), &SceneMergePlugin::merge);
}

void SceneMergePlugin::_notification(int notification) {
}

SceneMergePlugin::SceneMergePlugin() {
	file_export_lib->set_title(TTR("Export Library"));
	file_export_lib->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	file_export_lib->connect("file_selected", callable_mp(this, &SceneMergePlugin::_dialog_action));
	file_export_lib_merge->set_text(TTR("Merge With Existing"));
	file_export_lib->get_vbox()->add_child(file_export_lib_merge, true);
	EditorNode::get_singleton()->get_gui_base()->add_child(file_export_lib, true);
	file_export_lib->set_title(TTR("Merge Scene"));
	EditorNode::get_singleton()->add_tool_menu_item("Merge Scene", callable_mp(this, &SceneMergePlugin::merge));
}
#endif
