/**************************************************************************/
/*  shader_material_pool.cpp                                              */
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

#include "shader_material_pool.h"
#include "../errors.h"
#include "../profiling.h"
#include "classes/rendering_server.h"

namespace zylann::godot {

void ShaderMaterialPool::set_template(Ref<ShaderMaterial> tpl) {
	_template_material = tpl;
	_materials.clear();
	_shader_params_cache.clear();

	if (_template_material.is_valid()) {
		Ref<Shader> shader = _template_material->get_shader();

		if (shader.is_valid()) {
			StdVector<godot::ShaderParameterInfo> params;
			get_shader_parameter_list(shader->get_rid(), params);

			for (const godot::ShaderParameterInfo &pi : params) {
				_shader_params_cache.push_back(pi.name);
			}
		}
	}
}

Ref<ShaderMaterial> ShaderMaterialPool::get_template() const {
	return _template_material;
}

Ref<ShaderMaterial> ShaderMaterialPool::allocate() {
	if (_template_material.is_null() || _template_material->get_shader().is_null()) {
		return Ref<ShaderMaterial>();
	}
	if (!_materials.empty()) {
		Ref<ShaderMaterial> material = _materials.back();
		_materials.pop_back();
		return material;
	}
	ZN_PROFILE_SCOPE();
	Ref<ShaderMaterial> material;
	material.instantiate();
	material->set_shader(_template_material->get_shader());
	for (const StringName &name : _shader_params_cache) {
		// Note, I don't need to make copies of textures. They are shared (at least those coming from the template
		// material).
		material->set_shader_parameter(name, _template_material->get_shader_parameter(name));
	}
	return material;
}

void ShaderMaterialPool::recycle(Ref<ShaderMaterial> material) {
	ZN_ASSERT_RETURN(material.is_valid());
	ZN_ASSERT_RETURN(_template_material.is_valid());
	ZN_ASSERT_RETURN(material->get_shader() == _template_material->get_shader());
	_materials.push_back(material);
}

Span<const StringName> ShaderMaterialPool::get_cached_shader_uniforms() const {
	return to_span(_shader_params_cache);
}

void copy_shader_params(const ShaderMaterial &src, ShaderMaterial &dst, Span<const StringName> params) {
	// Ref<Shader> shader = src.get_shader();
	// ZN_ASSERT_RETURN(shader.is_valid());
	// Not using `Shader::get_param_list()` because it is not exposed to the script/extension API, and it prepends
	// `shader_params/` to every parameter name, which is slow and not usable for our case.
	// TBH List is slow too, I don't know why Godot uses that for lists of shader params.
	// List<PropertyInfo> properties;
	// RenderingServer::get_singleton()->shader_get_shader_uniform_list(shader->get_rid(), &properties);
	// for (const PropertyInfo &property : properties) {
	// 	dst.set_shader_uniform(property.name, src.get_shader_uniform(property.name));
	// }
	for (unsigned int i = 0; i < params.size(); ++i) {
		const StringName &name = params[i];
		dst.set_shader_parameter(name, src.get_shader_parameter(name));
	}
}

} // namespace zylann::godot
