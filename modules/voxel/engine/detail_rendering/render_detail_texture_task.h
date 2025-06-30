/**************************************************************************/
/*  render_detail_texture_task.h                                          */
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

#include "../../generators/voxel_generator.h"
#include "../../meshers/voxel_mesher.h"
#include "../../util/containers/std_vector.h"
#include "../../util/memory/memory.h"
#include "../../util/tasks/threaded_task.h"
#include "../ids.h"
#include "../priority_dependency.h"
#include "detail_rendering.h"

namespace zylann::voxel {

class RenderDetailTextureGPUTask;

// Renders textures providing extra details to far away voxel meshes.
// This is separate from the meshing task because it takes significantly longer to complete. It has different priority
// so most of the time we can get the mesh earlier and affine later with the results.
class RenderDetailTextureTask : public IThreadedTask {
public:
	// Input

	UniquePtr<ICellIterator> cell_iterator;
	// TODO Optimize: perhaps we could find a way to not copy mesh data? The only reason is because Godot wants a
	// slightly different data structure potentially taking unnecessary doubles because it uses `Vector3`...
	StdVector<Vector3f> mesh_vertices;
	StdVector<Vector3f> mesh_normals;
	StdVector<int> mesh_indices;
	Ref<VoxelGenerator> generator;
	std::shared_ptr<VoxelData> voxel_data;
	Vector3i mesh_block_size;
	uint8_t lod_index;
	bool use_gpu = false;
	DetailRenderingSettings detail_texture_settings;

	// Output (to be assigned so it can be populated)
	std::shared_ptr<DetailTextureOutput> output_textures;

	// Identification
	Vector3i mesh_block_position;
	VolumeID volume_id;
	PriorityDependency priority_dependency;

	const char *get_debug_name() const override {
		return "RenderDetailTexture";
	}

	void run(ThreadedTaskContext &ctx) override;
	void apply_result() override;
	TaskPriority get_priority() override;
	bool is_cancelled() override;

	// This is exposed for testing
	RenderDetailTextureGPUTask *make_gpu_task();

private:
	void run_on_cpu();
#ifdef VOXEL_ENABLE_GPU
	void run_on_gpu();
#endif
};

#ifdef VOXEL_ENABLE_GPU

// Performs final operations on the CPU after the GPU work is done
class RenderDetailTexturePass2Task : public IThreadedTask {
public:
	PackedByteArray atlas_data;
	DetailTextureData edited_tiles_texture_data;
	StdVector<DetailTextureData::Tile> tile_data;
	std::shared_ptr<DetailTextureOutput> output_textures;
	VolumeID volume_id;
	Vector3i mesh_block_position;
	Vector3i mesh_block_size;
	uint16_t atlas_width;
	uint16_t atlas_height;
	uint8_t lod_index;
	uint8_t tile_size_pixels;

	const char *get_debug_name() const override {
		return "RenderDetailTexturePass2";
	}

	void run(ThreadedTaskContext &ctx) override;
	void apply_result() override;
};

#endif

} // namespace zylann::voxel
