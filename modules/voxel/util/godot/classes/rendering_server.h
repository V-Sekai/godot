/**************************************************************************/
/*  rendering_server.h                                                    */
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
#include <servers/rendering_server.h>
#elif defined(ZN_GODOT_EXTENSION)
#include <godot_cpp/classes/rendering_server.hpp>
using namespace godot;
#endif

#include "../../containers/std_vector.h"
#include "../macros.h"

ZN_GODOT_FORWARD_DECLARE(class ProjectSettings);

namespace zylann::godot {

inline void free_rendering_server_rid(RenderingServer &rs, const RID &rid) {
#if defined(ZN_GODOT)
	rs.free(rid);
#elif defined(ZN_GODOT_EXTENSION)
	rs.free_rid(rid);
#endif
}

struct ShaderParameterInfo {
	String name;
	Variant::Type type;
};

void get_shader_parameter_list(const RID &shader_rid, StdVector<ShaderParameterInfo> &out_parameters);

String get_current_rendering_method_name();

// Enum equivalent to strings used in ProjectSettings and RenderingServer.
enum RenderMethod {
	RENDER_METHOD_FORWARD_PLUS,
	RENDER_METHOD_MOBILE,
	RENDER_METHOD_GL_COMPATIBILITY,
	RENDER_METHOD_UNKNOWN,
};

RenderMethod get_current_rendering_method();

String get_current_rendering_driver_name();

enum RenderDriverName {
	RENDER_DRIVER_VULKAN,
	RENDER_DRIVER_D3D12,
	RENDER_DRIVER_METAL,
	RENDER_DRIVER_OPENGL3,
	RENDER_DRIVER_OPENGL3_ES,
	RENDER_DRIVER_OPENGL3_ANGLE,
	RENDER_DRIVER_UNKNOWN,
};

RenderDriverName get_current_rendering_driver();

// Enum equivalent of `ProjectSettings.rendering/driver/threads/thread_model`.
// This is unfortunately not exposed.
enum RenderThreadModel {
	RENDER_THREAD_UNSAFE,
	RENDER_THREAD_SAFE,
	RENDER_SEPARATE_THREAD,
};

RenderThreadModel get_render_thread_model(const ProjectSettings &settings);

// Tells if it is safe to call functions of the RenderingServer from a thread other than the main one.
bool is_render_thread_model_safe(const RenderThreadModel mode);

} // namespace zylann::godot
