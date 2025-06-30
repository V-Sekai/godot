/**************************************************************************/
/*  rendering_device.h                                                    */
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

#if defined(ZN_GODOT)
#include <servers/rendering/rendering_device.h>
#elif defined(ZN_GODOT_EXTENSION)
#include <godot_cpp/classes/rendering_device.hpp>
using namespace godot;
#endif

#include "../macros.h"
#include "rd_shader_spirv.h"

ZN_GODOT_FORWARD_DECLARE(class RDShaderSource)

namespace zylann::godot {

void free_rendering_device_rid(RenderingDevice &rd, RID rid);

// TODO GDX: For some reason, the API exposed to scripts and extensions is not exposed to modules...
// This forces me to copy implementations to keep my code the same in both module and extension targets

Ref<RDShaderSPIRV> shader_compile_spirv_from_source(RenderingDevice &rd, RDShaderSource &p_source, bool p_allow_cache);
RID shader_create_from_spirv(RenderingDevice &rd, RDShaderSPIRV &p_spirv, String name = "");
RID texture_create(
		RenderingDevice &rd,
		RDTextureFormat &p_format,
		RDTextureView &p_view,
		const TypedArray<PackedByteArray> &p_data
);
RID uniform_set_create(RenderingDevice &rd, Array uniforms, RID shader, int shader_set);
RID sampler_create(RenderingDevice &rd, const RDSamplerState &sampler_state);
Error update_storage_buffer(
		RenderingDevice &rd,
		RID rid,
		unsigned int offset,
		unsigned int size,
		const PackedByteArray &pba
);

} // namespace zylann::godot
