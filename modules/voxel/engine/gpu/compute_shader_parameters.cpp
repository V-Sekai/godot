/**************************************************************************/
/*  compute_shader_parameters.cpp                                         */
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

#include "compute_shader_parameters.h"
#include "../../util/godot/classes/rd_uniform.h"
#include "../voxel_engine.h"

namespace zylann::voxel {

void add_uniform_params(const StdVector<ComputeShaderParameter> &params, Array &uniforms, const RID filtering_sampler) {
	for (const ComputeShaderParameter &p : params) {
		ZN_ASSERT(p.resource != nullptr);
		ZN_ASSERT(p.resource->get_rid().is_valid());

		Ref<RDUniform> uniform;
		uniform.instantiate();

		uniform->set_binding(p.binding);

		switch (p.resource->get_type()) {
			case ComputeShaderResourceInternal::TYPE_TEXTURE_2D:
			case ComputeShaderResourceInternal::TYPE_TEXTURE_3D:
				uniform->set_uniform_type(RenderingDevice::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE);
				uniform->add_id(filtering_sampler);
				break;

			case ComputeShaderResourceInternal::TYPE_STORAGE_BUFFER:
				uniform->set_uniform_type(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER);
				break;

			default:
				// May add more types if necessary
				ZN_CRASH_MSG("Unhandled type");
				break;
		}

		uniform->add_id(p.resource->get_rid());

		uniforms.append(uniform);
	}
}

} // namespace zylann::voxel
