/**************************************************************************/
/*  voxel_graph_editor_node.h                                             */
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

#include "../../generators/graph/voxel_graph_runtime.h"
#include "../../util/containers/std_vector.h"
#include "../../util/godot/classes/graph_node.h"

ZN_GODOT_FORWARD_DECLARE(class ColorRect)
ZN_GODOT_FORWARD_DECLARE(class Label)

namespace zylann::voxel {

class VoxelGraphEditorNodePreview;
struct GraphEditorAdapter;

namespace pg {
class VoxelGraphFunction;
}

// GUI graph node with a few custom data attached.
class VoxelGraphEditorNode : public GraphNode {
	GDCLASS(VoxelGraphEditorNode, GraphNode)
public:
	static VoxelGraphEditorNode *create(const pg::VoxelGraphFunction &graph, uint32_t node_id);

	void update_title(const pg::VoxelGraphFunction &graph);
	void poll(const pg::VoxelGraphFunction &graph);

	void update_range_analysis_tooltips(const GraphEditorAdapter &adapter, const pg::Runtime::State &state);
	void clear_range_analysis_tooltips();

	void update_layout(const pg::VoxelGraphFunction &graph);
	void update_comment_text(const pg::VoxelGraphFunction &graph);

	bool has_outputs() const {
		return _output_labels.size() > 0;
	}

	inline uint32_t get_generator_node_id() const {
		return _node_id;
	}

	inline VoxelGraphEditorNodePreview *get_preview() const {
		return _preview;
	}

	void set_profiling_ratio_visible(bool p_visible);
	void set_profiling_ratio(float ratio);

private:
	void update_title(const pg::VoxelGraphFunction &graph, uint32_t node_id);
	void poll_default_inputs(const pg::VoxelGraphFunction &graph);
	void poll_params(const pg::VoxelGraphFunction &graph);

	void _notification(int p_what);

	static void _bind_methods();

	uint32_t _node_id = 0;
	VoxelGraphEditorNodePreview *_preview = nullptr;
	StdVector<Control *> _output_labels;
	Label *_comment_label = nullptr;

	struct InputHint {
		Label *label;
		Variant last_value;
	};

	StdVector<InputHint> _input_hints;
	StdVector<Node *> _rows;

	float _profiling_ratio = 0.f;
	bool _profiling_ratio_enabled = false;
	bool _is_relay = false;
	bool _is_comment = false;
};

} // namespace zylann::voxel
