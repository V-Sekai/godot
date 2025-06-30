/**************************************************************************/
/*  voxel_terrain_editor_plugin.h                                         */
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

#pragma once

#include "../../engine/ids.h"
#include "../../util/containers/fixed_array.h"
#include "../../util/godot/classes/editor_plugin.h"
#include "../../util/godot/macros.h"
#include "../../util/godot/object_weak_ref.h"
#include "voxel_terrain_editor_inspector_plugin.h"

// When compiling with GodotCpp, it isn't possible to forward-declare these, due to how virtual methods are implemented.
#include "../../util/godot/classes/camera_3d.h"
#include "../../util/godot/classes/input_event.h"

ZN_GODOT_FORWARD_DECLARE(class MenuButton)
ZN_GODOT_FORWARD_DECLARE(class EditorFileDialog)

namespace zylann::voxel {

class VoxelAboutWindow;
class VoxelNode;
class VoxelTerrainEditorTaskIndicator;

class VoxelTerrainEditorPlugin : public zylann::godot::ZN_EditorPlugin {
	GDCLASS(VoxelTerrainEditorPlugin, zylann::godot::ZN_EditorPlugin)
public:
	VoxelTerrainEditorPlugin();

protected:
	bool _zn_handles(const Object *p_object) const override;
	void _zn_edit(Object *p_object) override;
	void _zn_make_visible(bool visible) override;
	EditorPlugin::AfterGUIInput _zn_forward_3d_gui_input(Camera3D *p_camera, const Ref<InputEvent> &p_event) override;

	void _notification(int p_what);

private:
	void init();
	void set_voxel_node(VoxelNode *node);
	void generate_menu_items(MenuButton *menu_button, bool is_lod_terrain);

	void _on_menu_item_selected(int id);
	void _on_save_file_dialog_file_selected(String fpath);

	static void _bind_methods();

	enum MenuID { //
		MENU_RESTART_STREAM,
		MENU_REMESH,

		MENU_STREAM_FOLLOW_CAMERA,
		MENU_ENABLE_EDITOR_VIEWER,

		MENU_DUMP_AS_SCENE,
		MENU_ABOUT,

		MENU_COUNT
	};

	zylann::godot::ObjectWeakRef<VoxelNode> _terrain_node;

	ViewerID _editor_viewer_id;
	bool _editor_viewer_enabled = true;
	Vector3 _editor_camera_last_position;
	bool _editor_viewer_follows_camera = false;

	MenuButton *_menu_button = nullptr;
	VoxelAboutWindow *_about_window = nullptr;
	VoxelTerrainEditorTaskIndicator *_task_indicator = nullptr;
	Ref<VoxelTerrainEditorInspectorPlugin> _inspector_plugin;
	EditorFileDialog *_save_file_dialog = nullptr;
};

} // namespace zylann::voxel
