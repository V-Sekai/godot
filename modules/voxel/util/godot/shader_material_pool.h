/**************************************************************************/
/*  shader_material_pool.h                                                */
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

#include "../containers/span.h"
#include "../containers/std_vector.h"
#include "classes/shader_material.h"

namespace zylann::godot {

// Reasons to pool numerous copies of the same ShaderMaterial:
// - In the editor, the Shader `changed` signal is connected even if they aren't editable, which makes the shader manage
//   a huge list of connections to "listening" materials, making insertion/removal super slow.
// - The generic `Resource.duplicate()` behavior is super slow. 95% of the time is spent NOT setting shader params
//   (getting property list in a LINKED LIST, many of which have to reach the fallback for "generated" ones, allocation,
//   resolution of assignments using variant `set` function...).
// - Allocating the object alone takes a bit of time
// TODO Next step could be to make a thin wrapper and use RenderingServer directly?
class ShaderMaterialPool {
public:
	void set_template(Ref<ShaderMaterial> tpl);
	Ref<ShaderMaterial> get_template() const;

	Ref<ShaderMaterial> allocate();
	void recycle(Ref<ShaderMaterial> material);

	// Materials have a cache too, but this one is even more direct
	Span<const StringName> get_cached_shader_uniforms() const;

private:
	Ref<ShaderMaterial> _template_material;
	StdVector<StringName> _shader_params_cache;
	StdVector<Ref<ShaderMaterial>> _materials;
};

void copy_shader_params(const ShaderMaterial &src, ShaderMaterial &dst, Span<const StringName> params);

} // namespace zylann::godot
