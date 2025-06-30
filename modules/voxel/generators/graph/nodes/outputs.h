/**************************************************************************/
/*  outputs.h                                                             */
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

#include "../node_type_db.h"

namespace zylann::voxel::pg {

void register_output_nodes(Span<NodeType> types) {
	using namespace math;

	{
		NodeType &t = types[VoxelGraphFunction::NODE_OUTPUT_SDF];
		t.name = "OutputSDF";
		t.category = CATEGORY_OUTPUT;
		t.inputs.push_back(NodeType::Port("sdf", 0.f, VoxelGraphFunction::AUTO_CONNECT_Y));
		t.outputs.push_back(NodeType::Port("_out"));
		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			const Runtime::Buffer &input = ctx.get_input(0);
			Runtime::Buffer &out = ctx.get_output(0);
			ZN_ASSERT(out.data != nullptr);
			memcpy(out.data, input.data, input.size * sizeof(float));
		};
		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			const Interval a = ctx.get_input(0);
			ctx.set_output(0, a);
		};
	}
	{
		NodeType &t = types[VoxelGraphFunction::NODE_OUTPUT_WEIGHT];
		t.name = "OutputWeight";
		t.category = CATEGORY_OUTPUT;
		t.inputs.push_back(NodeType::Port("weight"));
		t.outputs.push_back(NodeType::Port("_out"));
		NodeType::Param layer_param("layer", Variant::INT, 0);
		layer_param.has_range = true;
		layer_param.min_value = 0;
		layer_param.max_value = 15;
		t.params.push_back(layer_param);
		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			const Runtime::Buffer &input = ctx.get_input(0);
			Runtime::Buffer &out = ctx.get_output(0);
			for (unsigned int i = 0; i < out.size; ++i) {
				out.data[i] = clamp(input.data[i], 0.f, 1.f);
			}
		};
		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			const Interval a = ctx.get_input(0);
			ctx.set_output(0, clamp(a, Interval::from_single_value(0.f), Interval::from_single_value(1.f)));
		};
	}
	{
		NodeType &t = types[VoxelGraphFunction::NODE_OUTPUT_TYPE];
		t.name = "OutputType";
		t.category = CATEGORY_OUTPUT;
		t.inputs.push_back(NodeType::Port("type"));
		t.outputs.push_back(NodeType::Port("_out"));
		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			const Runtime::Buffer &input = ctx.get_input(0);
			Runtime::Buffer &out = ctx.get_output(0);
			memcpy(out.data, input.data, input.size * sizeof(float));
		};
		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			const Interval a = ctx.get_input(0);
			ctx.set_output(0, a);
		};
	}
	{
		NodeType &t = types[VoxelGraphFunction::NODE_OUTPUT_SINGLE_TEXTURE];
		t.name = "OutputSingleTexture";
		t.category = CATEGORY_OUTPUT;
		t.inputs.push_back(NodeType::Port("index"));
		t.outputs.push_back(NodeType::Port("_out"));
		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			const Runtime::Buffer &input = ctx.get_input(0);
			Runtime::Buffer &out = ctx.get_output(0);
			memcpy(out.data, input.data, input.size * sizeof(float));
		};
		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			const Interval a = ctx.get_input(0);
			ctx.set_output(0, a);
		};
	}
	{
		NodeType &t = types[VoxelGraphFunction::NODE_CUSTOM_OUTPUT];
		t.name = "CustomOutput";
		t.category = CATEGORY_OUTPUT;
		t.inputs.push_back(NodeType::Port("value"));
		t.outputs.push_back(NodeType::Port("_out"));
		// t.params.push_back(NodeType::Param("binding", Variant::INT, 0));
		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			const Runtime::Buffer &input = ctx.get_input(0);
			Runtime::Buffer &out = ctx.get_output(0);
			memcpy(out.data, input.data, input.size * sizeof(float));
		};
		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			const Interval a = ctx.get_input(0);
			ctx.set_output(0, a);
		};
	}
}

} // namespace zylann::voxel::pg
