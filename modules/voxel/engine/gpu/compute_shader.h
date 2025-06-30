/**************************************************************************/
/*  compute_shader.h                                                      */
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

#include "../../util/godot/core/rid.h"
#include "../../util/godot/core/string.h"
#include "../../util/memory/memory.h"

ZN_GODOT_FORWARD_DECLARE(class RenderingDevice)

namespace zylann::voxel {

struct ComputeShaderInternal {
	RID rid;
#if DEBUG_ENABLED
	String debug_name;
#endif

	void clear(RenderingDevice &rd);
	void load_from_glsl(RenderingDevice &rd, String source_text, String name);

	// An invalid instance means the shader failed to compile
	inline bool is_valid() const {
		return rid.is_valid();
	}
};

class ComputeShader;

// See ComputeShaderResourceFactory
struct ComputeShaderFactory {
	ComputeShaderFactory() = delete;

	[[nodiscard]]
	static std::shared_ptr<ComputeShader> create_from_glsl(String source_text, String name);

	[[nodiscard]]
	static std::shared_ptr<ComputeShader> create_invalid();
};

// Thin RAII wrapper around compute shaders created with the `RenderingDevice` held inside `VoxelEngine`.
// If the source can change at runtime, it may be passed around using shared pointers and a new instance may be created,
// rather than clearing the old shader anytime, for thread-safety. A reference should be kept as long as a dispatch of
// this shader is running on the graphics card.
class ComputeShader {
public:
	friend struct ComputeShaderFactory;

	~ComputeShader();

	// Only use on GPU task thread
	RID get_rid() const;

private:
	ComputeShaderInternal _internal;
};

} // namespace zylann::voxel
