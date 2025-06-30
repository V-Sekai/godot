/**************************************************************************/
/*  voxel_instance_library_multimesh_item_editor_plugin.h                 */
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

#include "../../terrain/instancing/voxel_instance_library_multimesh_item.h"
#include "../../util/godot/classes/editor_plugin.h"
#include "voxel_instance_library_multimesh_item_inspector_plugin.h"

ZN_GODOT_FORWARD_DECLARE(class EditorFileDialog)

namespace zylann::voxel {

class VoxelInstanceLibraryMultiMeshItemEditorPlugin : public zylann::godot::ZN_EditorPlugin {
	GDCLASS(VoxelInstanceLibraryMultiMeshItemEditorPlugin, zylann::godot::ZN_EditorPlugin)
public:
	VoxelInstanceLibraryMultiMeshItemEditorPlugin();

#if defined(ZN_GODOT)
	void _on_update_from_scene_button_pressed(VoxelInstanceLibraryMultiMeshItem *item);
#elif defined(ZN_GODOT_EXTENSION)
	void _on_update_from_scene_button_pressed(Object *item_o);
#endif

protected:
	bool _zn_handles(const Object *p_object) const override;
	void _zn_edit(Object *p_object) override;
	void _zn_make_visible(bool visible) override;

private:
	void init();
	void _notification(int p_what);

	void _on_open_scene_dialog_file_selected(String fpath);

	static void _bind_methods();

	EditorFileDialog *_open_scene_dialog = nullptr;
	Ref<VoxelInstanceLibraryMultiMeshItem> _item;
	Ref<VoxelInstanceLibraryMultiMeshItemInspectorPlugin> _inspector_plugin;
};

} // namespace zylann::voxel
