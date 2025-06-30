/**************************************************************************/
/*  voxel_modifier_gd.h                                                   */
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

#include "../../storage/voxel_data.h"
#include "../../terrain/variable_lod/voxel_lod_terrain.h"
#include "../../util/godot/classes/node_3d.h"
#include "../voxel_modifier.h"

#ifdef TOOLS_ENABLED
#include "../../util/godot/core/version.h"
#endif

namespace zylann::voxel::godot {

class VoxelModifier : public Node3D {
	GDCLASS(VoxelModifier, Node3D)
public:
	enum Operation { //
		OPERATION_ADD,
		OPERATION_REMOVE,
		OPERATION_COUNT
	};

	VoxelModifier();

	void set_operation(Operation op);
	Operation get_operation() const;

	void set_smoothness(float s);
	float get_smoothness() const;

#ifdef TOOLS_ENABLED
#if defined(ZN_GODOT)
	PackedStringArray get_configuration_warnings() const override;
#elif defined(ZN_GODOT_EXTENSION)
	PackedStringArray _get_configuration_warnings() const override;
#endif
	virtual void get_configuration_warnings(PackedStringArray &warnings) const;
#endif

protected:
	virtual zylann::voxel::VoxelModifier *create(zylann::voxel::VoxelModifierStack &modifiers, uint32_t id);
	void _notification(int p_what);

	VoxelLodTerrain *_volume = nullptr;
	uint32_t _modifier_id = 0;

private:
	static void _bind_methods();

	Operation _operation = OPERATION_ADD;
	float _smoothness = 0.f;
};

// Helpers

void post_edit_modifier(VoxelLodTerrain &volume, AABB aabb);

template <typename T>
T *get_modifier(VoxelLodTerrain &volume, uint32_t id, zylann::voxel::VoxelModifier::Type type) {
	VoxelData &data = volume.get_storage();
	VoxelModifierStack &modifiers = data.get_modifiers();
	zylann::voxel::VoxelModifier *modifier = modifiers.get_modifier(id);
	ZN_ASSERT_RETURN_V(modifier != nullptr, nullptr);
	ZN_ASSERT_RETURN_V(modifier->get_type() == type, nullptr);
	return static_cast<T *>(modifier);
}

} // namespace zylann::voxel::godot

VARIANT_ENUM_CAST(zylann::voxel::godot::VoxelModifier::Operation);
