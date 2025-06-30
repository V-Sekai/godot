/**************************************************************************/
/*  voxel_generator_image.h                                               */
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

#include "../../util/godot/macros.h"
#include "../../util/thread/rw_lock.h"
#include "voxel_generator_heightmap.h"

ZN_GODOT_FORWARD_DECLARE(class Image)

namespace zylann::voxel {

// Provides infinite tiling heightmap based on an image
class VoxelGeneratorImage : public VoxelGeneratorHeightmap {
	GDCLASS(VoxelGeneratorImage, VoxelGeneratorHeightmap)

public:
	VoxelGeneratorImage();
	~VoxelGeneratorImage();

	void set_image(Ref<Image> im);
	Ref<Image> get_image() const;

	void set_blur_enabled(bool enable);
	bool is_blur_enabled() const;

	Result generate_block(VoxelGenerator::VoxelQueryData input) override;

private:
	static void _bind_methods();

private:
	// Proper reference used for external access.
	Ref<Image> _image;

	struct Parameters {
		// This is a read-only copy of the image.
		// It wastes memory for sure, but Godot does not offer any way to secure this better.
		// If this is a problem one day, we could add an option to dereference the external image in game.
		Ref<Image> image;
		// Mostly here as demo/tweak. It's better recommended to use an EXR/float image.
		bool blur_enabled = false;
	};

	Parameters _parameters;
	RWLock _parameters_lock;
};

} // namespace zylann::voxel
