/**************************************************************************/
/*  shader.cpp                                                            */
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

#include "shader.h"
#include "../../containers/std_vector.h"
#include "rendering_server.h"

namespace zylann::godot {

#ifdef TOOLS_ENABLED

// TODO Cannot use `Shader.has_uniform()` because it is unreliable.
// See https://github.com/godotengine/godot/issues/64467
bool shader_has_uniform(const Shader &shader, StringName uniform_name) {
	StdVector<ShaderParameterInfo> params;
	get_shader_parameter_list(shader.get_rid(), params);
	for (const ShaderParameterInfo &pi : params) {
		if (pi.name == uniform_name) {
			return true;
		}
	}
	// List<PropertyInfo> params;
	// RenderingServer::get_singleton()->get_shader_parameter_list(shader.get_rid(), &params);
	// for (const PropertyInfo &pi : params) {
	// 	if (pi.name == uniform_name) {
	// 		return true;
	// 	}
	// }
	return false;
}

String get_missing_uniform_names(Span<const StringName> expected_uniforms, const Shader &shader) {
	String missing_uniforms;

	// TODO Cannot use `Shader.has_uniform()` because it is unreliable.
	// See https://github.com/godotengine/godot/issues/64467
	// for (unsigned int i = 0; i < expected_uniforms.size(); ++i) {
	// 	StringName uniform_name = expected_uniforms[i];
	// 	ZN_ASSERT_CONTINUE(uniform_name != StringName());
	// 	if (!shader.has_uniform(uniform_name)) {
	// 		if (missing_uniforms.size() > 0) {
	// 			missing_uniforms += ", ";
	// 		}
	// 		missing_uniforms += uniform_name;
	// 	}
	// }

	StdVector<ShaderParameterInfo> params;
	get_shader_parameter_list(shader.get_rid(), params);

	for (const StringName &name : expected_uniforms) {
		bool found = false;
		for (const ShaderParameterInfo &pi : params) {
			if (pi.name == name) {
				found = true;
				break;
			}
		}

		if (!found) {
			if (missing_uniforms.length() > 0) {
				missing_uniforms += ", ";
			}
			missing_uniforms += name;
		}
	}

	return missing_uniforms;
}

#endif

} // namespace zylann::godot
