/**************************************************************************/
/*  graph_editor_adapter.h                                                */
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

#include "../../generators/graph/voxel_generator_graph.h"
#include "../../generators/graph/voxel_graph_function.h"

namespace zylann::voxel {

// To workaround some legacy differences between graph types, so that the graph editor can offer the same features when
// editing either. Eventually we should refactor things so these differences go away.
struct GraphEditorAdapter {
	Ref<VoxelGeneratorGraph> generator;
	Ref<pg::VoxelGraphFunction> graph;
	StdVector<float> dummy_input_buffer;

	GraphEditorAdapter(Ref<VoxelGeneratorGraph> p_generator, Ref<pg::VoxelGraphFunction> p_graph) :
			generator(p_generator), graph(p_graph) {}

	inline bool is_good() const {
		if (generator.is_valid()) {
			return generator->is_good();
		}
		if (graph.is_valid()) {
			return graph->is_compiled();
		}
		return false;
	}

	bool try_get_output_port_address(ProgramGraph::PortLocation src, uint32_t &out_addr) const {
		if (generator.is_valid()) {
			return generator->try_get_output_port_address(src, out_addr);
		}
		std::shared_ptr<pg::VoxelGraphFunction::CompiledGraph> cg = graph->get_compiled_graph();
		uint16_t addr;
		const bool found = cg->runtime.try_get_output_port_address(src, addr);
		out_addr = addr;
		return found;
	}

	void generate_set(Span<const float> x, Span<const float> y, Span<const float> z) {
		if (generator.is_valid()) {
			generator->generate_set(x, y, z);
			return;
		}

		Span<const pg::VoxelGraphFunction::Port> inputs = graph->get_input_definitions();
		StdVector<Span<const float>> input_buffers;
		for (unsigned int i = 0; i < inputs.size(); ++i) {
			const pg::VoxelGraphFunction::Port &port = inputs[i];
			switch (port.type) {
				case pg::VoxelGraphFunction::NODE_INPUT_X:
					input_buffers.push_back(x);
					break;
				case pg::VoxelGraphFunction::NODE_INPUT_Y:
					input_buffers.push_back(y);
					break;
				case pg::VoxelGraphFunction::NODE_INPUT_Z:
					input_buffers.push_back(z);
					break;
				default:
					dummy_input_buffer.resize(x.size(), 0.f);
					input_buffers.push_back(to_span(dummy_input_buffer));
					break;
			}
		}

		// Output is thrown away, we just want to run to get intermediates.
		// We also don't constrain processing buffer size so generation happens in one pass and we can read full-size
		// intermediates... but that's not great with big graphs. We should probably allow a way to consume the data
		// after each pass? Or directly support Preview nodes as outputs?
		graph->execute(to_span_const(input_buffers), Span<Span<float>>(), x.size(), true);
	}

	const pg::Runtime::State &get_last_state_from_current_thread() {
		if (generator.is_valid()) {
			return VoxelGeneratorGraph::get_last_state_from_current_thread();
		} else {
			return pg::VoxelGraphFunction::get_runtime_cache_tls().state;
		}
	}

	void debug_analyze_range(const Vector3i min_pos, const Vector3i max_pos) {
		const bool with_optimized_execution_map = true;

		if (generator.is_valid()) {
			generator->debug_analyze_range(min_pos, max_pos, with_optimized_execution_map);
			return;
		}

		Span<const pg::VoxelGraphFunction::Port> inputs = graph->get_input_definitions();
		StdVector<math::Interval> input_ranges;
		for (unsigned int i = 0; i < inputs.size(); ++i) {
			const pg::VoxelGraphFunction::Port &port = inputs[i];
			switch (port.type) {
				case pg::VoxelGraphFunction::NODE_INPUT_X:
					input_ranges.push_back(math::Interval(min_pos.x, max_pos.x));
					break;
				case pg::VoxelGraphFunction::NODE_INPUT_Y:
					input_ranges.push_back(math::Interval(min_pos.y, max_pos.y));
					break;
				case pg::VoxelGraphFunction::NODE_INPUT_Z:
					input_ranges.push_back(math::Interval(min_pos.z, max_pos.z));
					break;
				default:
					input_ranges.push_back(math::Interval());
					break;
			}
		}

		graph->debug_analyze_range(to_span(input_ranges), with_optimized_execution_map);
	}
};

} // namespace zylann::voxel
