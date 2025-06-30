/**************************************************************************/
/*  voxel_blocky_library.h                                                */
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
#include "voxel_blocky_library_base.h"
#include "voxel_blocky_model.h"

namespace zylann::voxel {

// Library exposing every model in a simple array. Indices in the array correspond to voxel data.
// Rotations and variants have to be setup manually as separate models. You may use this library if your models are
// simple, or if you want to organize your own system of voxel types.
//
// Should have been named `VoxelBlockyModelLibrary` or `VoxelBlockyLibrarySimple`, but the name was kept for
// compatibility with previous versions.
class VoxelBlockyLibrary : public VoxelBlockyLibraryBase {
	GDCLASS(VoxelBlockyLibrary, VoxelBlockyLibraryBase)

public:
	VoxelBlockyLibrary();
	~VoxelBlockyLibrary();

	void load_default() override;
	void clear() override;

	void bake() override;

	int get_model_index_from_resource_name(String resource_name) const;

	// Convenience method that returns the index of the added model
	int add_model(Ref<VoxelBlockyModel> model);

	//-------------------------
	// Internal use

	// inline bool has_model(unsigned int id) const {
	// 	return id < _voxel_models.size() && _voxel_models[id].is_valid();
	// }

	// unsigned int get_model_count() const;

	// inline const VoxelBlockyModel &get_model_const(unsigned int id) const {
	// 	const Ref<VoxelBlockyModel> &model = _voxel_models[id];
	// 	ZN_ASSERT(model.is_valid());
	// 	return **model;
	// }

#ifdef TOOLS_ENABLED
	void get_configuration_warnings(PackedStringArray &out_warnings) const override;
#endif

private:
	Ref<VoxelBlockyModel> _b_get_model(unsigned int id) const;

	TypedArray<VoxelBlockyModel> _b_get_models() const;
	void _b_set_models(TypedArray<VoxelBlockyModel> models);

	bool _set(const StringName &p_name, const Variant &p_value);

	static void _bind_methods();

private:
	// Indices matter, they correspond to voxel data
	StdVector<Ref<VoxelBlockyModel>> _voxel_models;
};

} // namespace zylann::voxel
