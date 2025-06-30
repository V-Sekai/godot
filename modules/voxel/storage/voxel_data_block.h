/**************************************************************************/
/*  voxel_data_block.h                                                    */
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

#include "../util/ref_count.h"
#include <cstdint>
#include <memory>

namespace zylann::voxel {

class VoxelBuffer;

// Stores voxel data for a chunk of the volume. Mesh and colliders are stored separately.
// Voxel data can be present, or not. If not present, it means we know the block contains no edits, and voxels can be
// obtained by querying generators.
// Voxel data can also be present as a cache of generators, for cheaper repeated queries.
class VoxelDataBlock {
public:
	RefCount viewers;

	VoxelDataBlock() {}

	VoxelDataBlock(unsigned int p_lod_index) : _lod_index(p_lod_index) {}

	VoxelDataBlock(std::shared_ptr<VoxelBuffer> &buffer, unsigned int p_lod_index) :
			_voxels(buffer), _lod_index(p_lod_index) {}

	VoxelDataBlock(VoxelDataBlock &&src) :
			viewers(src.viewers),
			_voxels(std::move(src._voxels)),
			_lod_index(src._lod_index),
			_needs_lodding(src._needs_lodding),
			_modified(src._modified),
			_edited(src._edited) {}

	VoxelDataBlock(const VoxelDataBlock &src) :
			viewers(src.viewers),
			_voxels(src._voxels),
			_lod_index(src._lod_index),
			_needs_lodding(src._needs_lodding),
			_modified(src._modified),
			_edited(src._edited) {}

	VoxelDataBlock &operator=(VoxelDataBlock &&src) {
		viewers = src.viewers;
		_lod_index = src._lod_index;
		_voxels = std::move(src._voxels);
		_needs_lodding = src._needs_lodding;
		_modified = src._modified;
		_edited = src._edited;
		return *this;
	}

	VoxelDataBlock operator=(const VoxelDataBlock &src) {
		viewers = src.viewers;
		_lod_index = src._lod_index;
		_voxels = src._voxels;
		_needs_lodding = src._needs_lodding;
		_modified = src._modified;
		_edited = src._edited;
		return *this;
	}

	inline unsigned int get_lod_index() const {
		return _lod_index;
	}

	// Tests if voxel data is present.
	// If false, it means the block has no edits and does not contain cached generated data,
	// so we may fallback on procedural generators on the fly or request a cache.
	inline bool has_voxels() const {
		return _voxels != nullptr;
	}

	// Get voxels, expecting them to be present
	VoxelBuffer &get_voxels() {
#ifdef DEBUG_ENABLED
		ZN_ASSERT(_voxels != nullptr);
#endif
		return *_voxels;
	}

	// Get voxels, expecting them to be present
	const VoxelBuffer &get_voxels_const() const {
#ifdef DEBUG_ENABLED
		ZN_ASSERT(_voxels != nullptr);
#endif
		return *_voxels;
	}

	// Get voxels, expecting them to be present
	std::shared_ptr<VoxelBuffer> get_voxels_shared() const {
#ifdef DEBUG_ENABLED
		ZN_ASSERT(_voxels != nullptr);
#endif
		return _voxels;
	}

	void set_voxels(const std::shared_ptr<VoxelBuffer> &buffer) {
		ZN_ASSERT_RETURN(buffer != nullptr);
		_voxels = buffer;
	}

	void clear_voxels() {
		_voxels = nullptr;
		_edited = false;
	}

	void set_modified(bool modified);

	inline bool is_modified() const {
		return _modified;
	}

	void set_needs_lodding(bool need_lodding) {
		_needs_lodding = need_lodding;
	}

	inline bool get_needs_lodding() const {
		return _needs_lodding;
	}

	inline void set_edited(bool edited) {
		_edited = edited;
	}

	inline bool is_edited() const {
		return _edited;
	}

private:
	// Voxel data. If null, it means the data may be obtained with procedural generation.
	std::shared_ptr<VoxelBuffer> _voxels;

	// TODO Storing lod index here might not be necessary, it is known since we have to get the map first.
	// For now it can remain here since in practice it doesn't cost space, due to other stored flags and alignment.
	uint8_t _lod_index = 0;

	// Indicates mipmaps need to be computed since this block was modified.
	bool _needs_lodding = false;

	// Indicates if this block is different from the time it was loaded (should be saved)
	bool _modified = false;

	// Tells if the block has ever been edited.
	// If `false`, then the data is a cache of generators and modifiers. It can be re-generated.
	// Once it becomes `true`, it usually never comes back to `false` unless reverted.
	bool _edited = false;

	// TODO Optimization: design a proper way to implement client-side caching for multiplayer
	//
	// Represents how many times the block was edited.
	// This allows to implement client-side caching in multiplayer.
	//
	// Note: when doing client-side caching, if the server decides to revert a block to generator output,
	// resetting version to 0 might not be a good idea, because if a client had version 1, it could mismatch with
	// the "new version 1" after the next edit. All clients having ever joined the server would have to be aware
	// of the revert before they start getting blocks with the server,
	// or need to be told which version is the "generated" one.
	// uint32_t _version;

	// Tells if it's worth requesting a more precise version of the data.
	// Will be `true` if it's not worth it.
	// bool _max_lod_hint = false;
};

} // namespace zylann::voxel
