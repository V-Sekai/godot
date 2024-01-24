/**************************************************************************/
/*  voxel_mesher_blocky.h                                                 */
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

#ifndef VOXEL_MESHER_BLOCKY_H
#define VOXEL_MESHER_BLOCKY_H

#include "../voxel_mesher.h"
#include "core/object/ref_counted.h"
#include "voxel_library.h"
#include <scene/resources/mesh.h>
#include <vector>

// TODO Rename VoxelMesherModelBatch

// Interprets voxel values as indexes to models in a VoxelLibrary, and batches
// them together. Overlapping faces are removed from the final mesh.
class VoxelMesherBlocky : public VoxelMesher {
	GDCLASS(VoxelMesherBlocky, VoxelMesher)

public:
	static const unsigned int MAX_MATERIALS = 8; // Arbitrary. Tweak if needed.
	static const int PADDING = 1;

	VoxelMesherBlocky();
	~VoxelMesherBlocky();

	void set_library(Ref<VoxelLibrary> library);
	Ref<VoxelLibrary> get_library() const;

	void set_occlusion_darkness(float darkness);
	float get_occlusion_darkness() const;

	void set_occlusion_enabled(bool enable);
	bool get_occlusion_enabled() const;

	void build(VoxelMesher::Output &output,
			const VoxelMesher::Input &input) override;

	Ref<Resource> duplicate(bool p_subresources = false) const override;
	int get_used_channels_mask() const override;

	bool supports_lod() const override { return false; }

	// Using std::vector because they make this mesher twice as fast than Godot
	// Vectors. See why: https://github.com/godotengine/godot/issues/24731
	struct Arrays {
		std::vector<Vector3> positions;
		std::vector<Vector3> normals;
		std::vector<Vector2> uvs;
		std::vector<Color> colors;
		std::vector<int> indices;
		std::vector<float> tangents;

		void clear() {
			positions.clear();
			normals.clear();
			uvs.clear();
			colors.clear();
			indices.clear();
			tangents.clear();
		}
	};

protected:
	static void _bind_methods();

private:
	struct Parameters {
		float baked_occlusion_darkness = 0.8;
		bool bake_occlusion = true;
		Ref<VoxelLibrary> library;
	};

	struct Cache {
		FixedArray<Arrays, MAX_MATERIALS> arrays_per_material;
	};

	// Parameters
	Parameters _parameters;
	RWLock _parameters_lock;

	// Work cache
	static thread_local Cache _cache;
};

#endif // VOXEL_MESHER_BLOCKY_H
