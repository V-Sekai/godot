/**************************************************************************/
/*  voxel_generator_multipass_cache_viewer.h                              */
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

#include "../../generators/multipass/voxel_generator_multipass_cb.h"
#include "../../util/containers/std_vector.h"
#include "../../util/godot/classes/control.h"
#include "../../util/godot/classes/image.h"
#include "../../util/godot/classes/image_texture.h"

namespace zylann::voxel {

class VoxelGeneratorMultipassCacheViewer : public Control {
	GDCLASS(VoxelGeneratorMultipassCacheViewer, Control)
public:
	VoxelGeneratorMultipassCacheViewer();

	void set_generator(Ref<VoxelGeneratorMultipassCB> generator);

private:
	void _notification(int p_what);
	void process();
	void draw();
	void update_image();

	// When compiling with GodotCpp, `_bind_methods` is not optional.
	static void _bind_methods() {}

	Ref<VoxelGeneratorMultipassCB> _generator;
	StdVector<VoxelGeneratorMultipassCB::DebugColumnState> _debug_column_states;
	Ref<Image> _image;
	Ref<ImageTexture> _texture;
	uint64_t _next_update_time = 0;
};

} // namespace zylann::voxel
