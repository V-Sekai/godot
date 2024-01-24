/**************************************************************************/
/*  voxel_mesher_cubes.h                                                  */
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

#ifndef VOXEL_MESHER_CUBES_H
#define VOXEL_MESHER_CUBES_H

#include "../voxel_mesher.h"
#include "voxel_color_palette.h"
#include <vector>

// A super simple mesher only producing colored cubes
class VoxelMesherCubes : public VoxelMesher {
	GDCLASS(VoxelMesherCubes, VoxelMesher)
public:
	static const unsigned int PADDING = 1;

	enum Materials { MATERIAL_OPAQUE = 0,
		MATERIAL_TRANSPARENT,
		MATERIAL_COUNT };

	enum ColorMode {
		// The voxel value will be treated as an RGBA color with components of equal
		// bit depth
		COLOR_RAW = 0,
		// The voxel value will map to a 32bit color in the palette specified on
		// this mesher
		COLOR_MESHER_PALETTE,
		// The voxel value will be copied directly to the vertex array,
		// so the proper color can be selected by a shader.
		// LIMITATION: only one material can be used in this mode at the moment.
		COLOR_SHADER_PALETTE,

		COLOR_MODE_COUNT
	};

	VoxelMesherCubes();
	~VoxelMesherCubes();

	void build(VoxelMesher::Output &output,
			const VoxelMesher::Input &input) override;

	void set_greedy_meshing_enabled(bool enable);
	bool is_greedy_meshing_enabled() const;

	void set_color_mode(ColorMode mode);
	ColorMode get_color_mode() const;

	void set_palette(Ref<VoxelColorPalette> palette);
	Ref<VoxelColorPalette> get_palette() const;

	Ref<Resource> duplicate(bool p_subresources = false) const override;
	int get_used_channels_mask() const override;

	void set_store_colors_in_texture(bool enable);
	bool get_store_colors_in_texture() const;

	bool supports_lod() const override { return false; }

	// Using std::vector because they make this mesher twice as fast than Godot
	// Vectors. See why: https://github.com/godotengine/godot/issues/24731
	struct Arrays {
		std::vector<Vector3> positions;
		std::vector<Vector3> normals;
		std::vector<Color> colors;
		std::vector<Vector2> uvs;
		std::vector<int> indices;

		void clear() {
			positions.clear();
			normals.clear();
			colors.clear();
			uvs.clear();
			indices.clear();
		}
	};

	struct GreedyAtlasData {
		struct ImageInfo {
			unsigned int first_color_index;
			unsigned int first_vertex_index; // From a quad
			unsigned int size_x;
			unsigned int size_y;
			unsigned int surface_index;
		};
		std::vector<Color8> colors;
		std::vector<ImageInfo> images;

		void clear() {
			colors.clear();
			images.clear();
		}
	};

protected:
	static void _bind_methods();

private:
	struct Parameters {
		ColorMode color_mode = COLOR_RAW;
		Ref<VoxelColorPalette> palette;
		bool greedy_meshing = true;
		bool store_colors_in_texture = false;
	};

	struct Cache {
		FixedArray<Arrays, MATERIAL_COUNT> arrays_per_material;
		std::vector<uint8_t> mask_memory_pool;
		GreedyAtlasData greedy_atlas_data;
	};

	// Parameters
	Parameters _parameters;
	RWLock _parameters_lock;

	// Work cache
	static thread_local Cache _cache;
};

VARIANT_ENUM_CAST(VoxelMesherCubes::ColorMode);
VARIANT_ENUM_CAST(VoxelMesherCubes::Materials);

#endif // VOXEL_MESHER_CUBES_H
