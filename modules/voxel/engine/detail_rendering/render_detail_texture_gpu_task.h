/**************************************************************************/
/*  render_detail_texture_gpu_task.h                                      */
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
#include "../../util/math/vector3f.h"
#include "../../util/math/vector4f.h"
#include "../gpu/gpu_storage_buffer_pool.h"
#include "../gpu/gpu_task_runner.h"
#include "../ids.h"
#include "detail_rendering.h"

#ifdef VOXEL_ENABLE_MODIFIERS
#include "../../modifiers/voxel_modifier.h"
#endif

namespace zylann::voxel {

class ComputeShader;
struct ComputeShaderParameters;

class RenderDetailTextureGPUTask : public IGPUTask {
public:
	uint16_t texture_width;
	uint16_t texture_height;

	// Using 4-component vectors to match alignment rules.
	// See https://stackoverflow.com/a/38172697
	StdVector<Vector4f> mesh_vertices;
	StdVector<int32_t> mesh_indices;
	StdVector<int32_t> cell_triangles;

	struct TileData {
		uint8_t cell_x = 0;
		uint8_t cell_y = 0;
		uint8_t cell_z = 0;
		uint8_t _pad = 0;
		// aaaaaaaa aaaaaaaa aaaaaaaa 0bbb00cc
		// a: 24-bit index into `u_cell_tris.data` array.
		// b: 3-bit number of triangles.
		// c: 2-bit projection direction (0:X, 1:Y, 2:Z)
		uint32_t data = 0;
	};

	// Should fit as an `ivec2[]`
	StdVector<TileData> tile_data;

	struct Params {
		Vector3f block_origin_world;
		float pixel_world_step;
		float max_deviation_cosine;
		float max_deviation_sine;
		int32_t tile_size_pixels;
		int32_t tiles_x;
		int32_t tiles_y;
	};

	Params params;

	// Base modifier (generator)
	std::shared_ptr<ComputeShader> shader;
	std::shared_ptr<ComputeShaderParameters> shader_params;

#ifdef VOXEL_ENABLE_MODIFIERS
	StdVector<VoxelModifier::ShaderData> modifiers;
#endif

	// Stuff to carry over for the second CPU pass
	std::shared_ptr<DetailTextureOutput> output;
	DetailTextureData edited_tiles_texture_data;
	Vector3i block_position;
	Vector3i block_size;
	VolumeID volume_id;
	uint8_t lod_index;
#ifdef VOXEL_TESTS
	PackedByteArray *testing_output = nullptr;
#endif

	void prepare(GPUTaskContext &ctx) override;
	void collect(GPUTaskContext &ctx) override;

	// Exposed for testing
	PackedByteArray collect_texture_and_cleanup(RenderingDevice &rd, GPUStorageBufferPool &storage_buffer_pool);

private:
	RID _normalmap_texture0_rid;
	RID _normalmap_texture1_rid;

	RID _gather_hits_pipeline_rid;
	RID _detail_generator_pipeline_rid;
	RID _detail_normalmap_pipeline_rid;
	RID _normalmap_dilation_pipeline_rid;
	StdVector<RID> _detail_modifier_pipelines;
	StdVector<RID> _uniform_sets_to_free;

	GPUStorageBuffer _mesh_vertices_sb;
	GPUStorageBuffer _mesh_indices_sb;
	GPUStorageBuffer _cell_triangles_sb;
	GPUStorageBuffer _tile_data_sb;
	GPUStorageBuffer _gather_hits_params_sb;
	RID _dilation_params_rid;
	GPUStorageBuffer _hit_positions_buffer_sb;
	GPUStorageBuffer _generator_params_sb;
	GPUStorageBuffer _sd_buffer0_sb;
	GPUStorageBuffer _sd_buffer1_sb;
	GPUStorageBuffer _normalmap_params_sb;
};

} // namespace zylann::voxel
