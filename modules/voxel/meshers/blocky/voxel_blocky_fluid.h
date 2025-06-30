/**************************************************************************/
/*  voxel_blocky_fluid.h                                                  */
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

#include "../../constants/cube_tables.h"
#include "../../util/containers/fixed_array.h"
#include "../../util/containers/std_vector.h"
#include "../../util/godot/classes/resource.h"
#include "../../util/math/vector3f.h"
#include "blocky_baked_library.h"
#include <cstdint>

ZN_GODOT_FORWARD_DECLARE(class Material);

namespace zylann::voxel {

namespace blocky {
struct MaterialIndexer;
}

// Minecraft-style fluid common configuration.
// Fluids are a bit special compared to regular models. Rendering them with precalculated models would require way too
// many of them. So instead, they are procedurally generated during meshing.
// They only require separate models to represent their level, or other states such as falling.
class VoxelBlockyFluid : public Resource {
	GDCLASS(VoxelBlockyFluid, Resource)
public:
	enum FlowState : uint8_t {
		// o---x
		// |
		// z
		// Values are proportional to an angle, and named after a top-down OpenGL coordinate system.
		FLOW_STRAIGHT_POSITIVE_X,
		FLOW_DIAGONAL_POSITIVE_X_NEGATIVE_Z,
		FLOW_STRAIGHT_NEGATIVE_Z,
		FLOW_DIAGONAL_NEGATIVE_X_NEGATIVE_Z,
		FLOW_STRAIGHT_NEGATIVE_X,
		FLOW_DIAGONAL_NEGATIVE_X_POSITIVE_Z,
		FLOW_STRAIGHT_POSITIVE_Z,
		FLOW_DIAGONAL_POSITIVE_X_POSITIVE_Z,
		FLOW_IDLE,
		FLOW_STATE_COUNT
	};

	VoxelBlockyFluid();

	void set_material(Ref<Material> material);
	Ref<Material> get_material() const;

	void set_dip_when_flowing_down(bool enable);
	bool get_dip_when_flowing_down() const;

	void bake(blocky::BakedFluid &baked_fluid, blocky::MaterialIndexer &materials) const;

private:
	static void _bind_methods();

	Ref<Material> _material;
	bool _dip_when_flowing_down = false;
};

} // namespace zylann::voxel
