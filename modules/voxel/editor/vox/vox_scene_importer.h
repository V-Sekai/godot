/**************************************************************************/
/*  vox_scene_importer.h                                                  */
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

#include "../../util/containers/std_vector.h"
#include "../../util/godot/classes/editor_import_plugin.h"

namespace zylann::voxel::magica {

// Imports a vox file as a scene, where the internal scene layout is preserved as nodes
class VoxelVoxSceneImporter : public zylann::godot::ZN_EditorImportPlugin {
	GDCLASS(VoxelVoxSceneImporter, zylann::godot::ZN_EditorImportPlugin)
public:
	String _zn_get_importer_name() const override;
	String _zn_get_visible_name() const override;
	PackedStringArray _zn_get_recognized_extensions() const override;
	String _zn_get_preset_name(int p_idx) const override;
	int _zn_get_preset_count() const override;
	String _zn_get_save_extension() const override;
	String _zn_get_resource_type() const override;
	float _zn_get_priority() const override;
	int _zn_get_import_order() const override;

	void _zn_get_import_options(
			StdVector<zylann::godot::ImportOptionWrapper> &p_out_options,
			const String &p_path,
			int p_preset_index
	) const override;

	bool _zn_get_option_visibility(
			const String &p_path,
			const StringName &p_option_name,
			const zylann::godot::KeyValueWrapper p_options
	) const override;

	Error _zn_import(
			const String &p_source_file,
			const String &p_save_path,
			const zylann::godot::KeyValueWrapper p_options,
			zylann::godot::StringListWrapper p_out_platform_variants,
			zylann::godot::StringListWrapper p_out_gen_files
	) const override;

	bool _zn_can_import_threaded() const override;

private:
	// When compiling with GodotCpp, `_bind_methods` is not optional.
	static void _bind_methods() {}
};

} // namespace zylann::voxel::magica
