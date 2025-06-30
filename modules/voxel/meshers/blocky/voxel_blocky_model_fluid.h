/**************************************************************************/
/*  voxel_blocky_model_fluid.h                                            */
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

#include "voxel_blocky_fluid.h"
#include "voxel_blocky_model.h"

namespace zylann::voxel {

// Minecraft-style fluid model for a specific level.
class VoxelBlockyModelFluid : public VoxelBlockyModel {
	GDCLASS(VoxelBlockyModelFluid, VoxelBlockyModel)
public:
	static const int MAX_LEVELS = 256;

	VoxelBlockyModelFluid();

	void set_fluid(Ref<VoxelBlockyFluid> fluid);
	Ref<VoxelBlockyFluid> get_fluid() const;

	void set_level(int level);
	int get_level() const;

	bool is_empty() const override;

	void bake(blocky::ModelBakingContext &ctx) const override;

	Ref<Mesh> get_preview_mesh() const override;

#ifdef TOOLS_ENABLED
	void get_configuration_warnings(PackedStringArray &warnings) const override;
#endif

private:
	void _on_fluid_changed();

	static void _bind_methods();

	Ref<VoxelBlockyFluid> _fluid;
	unsigned int _level = 0;
};

} // namespace zylann::voxel
