/**************************************************************************/
/*  voxel_blocky_type_library_editor_inspector_plugin.h                   */
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

#include "../../../meshers/blocky/types/voxel_blocky_type_library.h"
#include "../../../util/godot/classes/editor_inspector_plugin.h"

namespace zylann::voxel {

class VoxelBlockyTypeLibraryIDSDialog;

class VoxelBlockyTypeLibraryEditorInspectorPlugin : public zylann::godot::ZN_EditorInspectorPlugin {
	GDCLASS(VoxelBlockyTypeLibraryEditorInspectorPlugin, zylann::godot::ZN_EditorInspectorPlugin)
public:
	void set_ids_dialog(VoxelBlockyTypeLibraryIDSDialog *ids_dialog);

protected:
	bool _zn_can_handle(const Object *p_object) const override;
	void _zn_parse_end(Object *p_object) override;

private:
	void _on_inspect_ids_button_pressed(Ref<VoxelBlockyTypeLibrary> library);

	static void _bind_methods();

	VoxelBlockyTypeLibraryIDSDialog *_ids_dialog = nullptr;
};

} // namespace zylann::voxel
