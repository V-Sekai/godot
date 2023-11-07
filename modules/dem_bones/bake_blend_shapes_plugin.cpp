/*************************************************************************/
/*  editor_scene_exporter_gltf_plugin.cpp                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifdef TOOLS_ENABLED
#include "bake_blend_shapes_plugin.h"
#include "core/config/project_settings.h"
#include "core/error/error_list.h"
#include "core/object/object.h"
#include "core/templates/vector.h"
#include "dem_bones.h"
#include "editor/editor_file_dialog.h"
#include "editor/editor_file_system.h"
#include "editor/editor_node.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/gui/check_box.h"
#include "scene/main/node.h"


String BakeBlendShapesPlugin::get_name() const {
	return "bake_blend_shapes";
}

bool BakeBlendShapesPlugin::has_main_screen() const {
	return false;
}

BakeBlendShapesPlugin::BakeBlendShapesPlugin() {
	file_export_lib = memnew(FileDialog);
	EditorNode::get_singleton()->get_gui_base()->add_child(file_export_lib);
	file_export_lib->set_title(RTR("Export Library"));
	file_export_lib->set_file_mode(FileDialog::FILE_MODE_SAVE_FILE);
	file_export_lib->set_access(FileDialog::ACCESS_FILESYSTEM);
	file_export_lib->clear_filters();
	file_export_lib->add_filter("*.glb");
	file_export_lib->add_filter("*.gltf");
	file_export_lib->set_title(RTR("Bake Blend Shapes..."));
	String bake_blend_shapes_name = RTR("Bake Blend Shapes");
	add_tool_menu_item(bake_blend_shapes_name, callable_mp(this, &BakeBlendShapesPlugin::convert_scene_to_bake_blend_shapes));
}

void BakeBlendShapesPlugin::convert_scene_to_bake_blend_shapes() {
	Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
	if (!root) {
		EditorNode::get_singleton()->show_accept(RTR("This operation can't be done without a scene."), RTR("OK"));
		return;
	}

	List<String> deps;
	Ref<BlendShapeBake> bake;
	bake.instantiate();
	Error err = bake->convert_scene(root);
	if (err != OK) {
		ERR_PRINT(vformat("Bake scene error %s.", itos(err)));
	}
}

#endif // TOOLS_ENABLED
