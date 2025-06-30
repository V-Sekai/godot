/**************************************************************************/
/*  misc.h                                                                */
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

inline float select(float a, float b, float threshold, float t) {
	return t < threshold ? a : b;
}

void register_misc_nodes(Span<NodeType> types) {
	using namespace math;

	{
		NodeType &t = types[VoxelGraphFunction::NODE_CONSTANT];
		t.name = "Constant";
		t.category = CATEGORY_CONSTANT;
		t.outputs.push_back(NodeType::Port("value"));
		t.params.push_back(NodeType::Param("value", Variant::FLOAT));
	}
	{
		NodeType &t = types[VoxelGraphFunction::NODE_RELAY];
		t.name = "Relay";
		t.category = CATEGORY_RELAY;
		t.inputs.push_back(NodeType::Port("in"));
		t.outputs.push_back(NodeType::Port("out"));
		t.is_pseudo_node = true;
	}
	{
		NodeType &t = types[VoxelGraphFunction::NODE_COMMENT];
		t.name = "Comment";
		t.category = CATEGORY_DEBUG;
		NodeType::Param text_param("text", Variant::STRING, Variant(""));
		text_param.multiline = true;
		t.params.push_back(text_param);
		t.debug_only = true;
		t.is_pseudo_node = true;
	}
	{
		NodeType &t = types[VoxelGraphFunction::NODE_FUNCTION];
		t.name = "Function";
		t.category = CATEGORY_FUNCTIONS;

		NodeType::Param func_param("_function", VoxelGraphFunction::get_class_static(), nullptr);
		func_param.hidden = true;
		t.params.push_back(func_param);

		t.debug_only = false;
		t.is_pseudo_node = true;
	}
	{
		struct Params {
			float threshold;
		};
		NodeType &t = types[VoxelGraphFunction::NODE_SELECT];
		// t < threshold ? a : b
		t.name = "Select";
		t.category = CATEGORY_CONVERT;
		t.inputs.push_back(NodeType::Port("a"));
		t.inputs.push_back(NodeType::Port("b"));
		t.inputs.push_back(NodeType::Port("t"));
		t.outputs.push_back(NodeType::Port("out"));
		t.params.push_back(NodeType::Param("threshold", Variant::FLOAT, 0.f));

		t.compile_func = [](CompileContext &ctx) {
			Params p;
			p.threshold = ctx.get_param(0).operator float();
			ctx.set_params(p);
		};

		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			bool a_ignored;
			bool b_ignored;
			const Runtime::Buffer &a = ctx.try_get_input(0, a_ignored);
			const Runtime::Buffer &b = ctx.try_get_input(1, b_ignored);
			const Runtime::Buffer &tested_value = ctx.get_input(2);
			const float threshold = ctx.get_params<Params>().threshold;

			Runtime::Buffer &out = ctx.get_output(0);

			const uint32_t buffer_size = out.size;

			if (a_ignored) {
				memcpy(out.data, b.data, buffer_size * sizeof(float));

			} else if (b_ignored) {
				memcpy(out.data, a.data, buffer_size * sizeof(float));

			} else if (tested_value.is_constant) {
				const float *src = tested_value.constant_value < threshold ? a.data : b.data;
				memcpy(out.data, src, buffer_size * sizeof(float));

			} else if (a.is_constant && b.is_constant && a.constant_value == b.constant_value) {
				memcpy(out.data, a.data, buffer_size * sizeof(float));

			} else {
				for (uint32_t i = 0; i < buffer_size; ++i) {
					out.data[i] = select(a.data[i], b.data[i], threshold, tested_value.data[i]);
				}
			}
		};

		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			const Interval a = ctx.get_input(0);
			const Interval b = ctx.get_input(1);
			const Interval tested_value = ctx.get_input(2);
			const float threshold = ctx.get_params<Params>().threshold;

			if (tested_value.min >= threshold) {
				ctx.set_output(0, b);
				// `a` won't be used
				ctx.ignore_input(0);

			} else if (tested_value.max < threshold) {
				ctx.set_output(0, a);
				// `b` won't be used
				ctx.ignore_input(1);

			} else {
				ctx.set_output(0, Interval::from_union(a, b));
			}
		};

		t.shader_gen_func = [](ShaderGenContext &ctx) {
			const float threshold = ctx.get_param(0);
			ctx.add_format(
					"{} = {} < {} ? {} : {};\n",
					ctx.get_output_name(0),
					ctx.get_input_name(2),
					threshold,
					ctx.get_input_name(0),
					ctx.get_input_name(1)
			);
		};
	}
	{
		NodeType &t = types[VoxelGraphFunction::NODE_EXPRESSION];
		t.name = "Expression";
		t.category = CATEGORY_MATH;
		NodeType::Param expression_param("expression", Variant::STRING, "0");
		expression_param.multiline = false;
		t.params.push_back(expression_param);
		t.outputs.push_back(NodeType::Port("out"));
		t.compile_func = [](CompileContext &ctx) {
			ctx.make_error(ZN_TTR("Internal error, expression wasn't expanded"));
		};
		t.is_pseudo_node = true;
	}
}

} // namespace zylann::voxel::pg
