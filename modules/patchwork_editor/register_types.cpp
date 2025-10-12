/*************************************************************************/
/*  register_types.cpp                                                   */
/*************************************************************************/

#include "register_types.h"
#include "editor/diff_inspector.h"
#include "editor/editor_node.h"
#include "editor/patchwork_editor.h"
#include "editor/text_diff.h"

void patchwork_editor_init_callback() {
    EditorNode *editor = EditorNode::get_singleton();
    PatchworkEditor *plugin = memnew(PatchworkEditor(editor));
    editor->add_editor_plugin(plugin);
    PatchworkEditor::singleton = plugin;
}

void initialize_patchwork_editor_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		ClassDB::register_class<DiffInspector>();
		ClassDB::register_class<PatchworkEditor>();
		ClassDB::register_class<DiffInspectorProperty>();
		ClassDB::register_class<EditorInspectorSection>();
		ClassDB::register_class<DiffInspectorSection>();
		ClassDB::register_class<TextDiffer>();
	}
	if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
		EditorNode::add_init_callback(&patchwork_editor_init_callback);
	}
}

void uninitialize_patchwork_editor_module(ModuleInitializationLevel p_level) {
}
