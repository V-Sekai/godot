/**************************************************************************/
/*  voxel_graph_node_inspector_wrapper.h                                  */
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
#include "../../util/godot/classes/ref_counted.h"

ZN_GODOT_FORWARD_DECLARE(class EditorUndoRedoManager)

namespace zylann::voxel {

class VoxelGraphEditor;

// Nodes aren't resources so this translates them into a form the inspector can understand.
// This makes it easier to support undo/redo and sub-resources.
// WARNING: `AnimationPlayer` will allow to keyframe properties, but there really is no support for that.
class VoxelGraphNodeInspectorWrapper : public RefCounted {
	GDCLASS(VoxelGraphNodeInspectorWrapper, RefCounted)
public:
	void setup(uint32_t p_node_id, VoxelGraphEditor *ed);

	// May be called when the graph editor is destroyed. This prevents from accessing dangling pointers in the eventual
	// case where UndoRedo invokes functions from this editor after the plugin is removed.
	void detach_from_graph_editor();

	inline Ref<pg::VoxelGraphFunction> get_graph() const {
		return _graph;
	}
	inline Ref<VoxelGeneratorGraph> get_generator() const {
		return _generator;
	}

protected:
	void _get_property_list(List<PropertyInfo> *p_list) const;
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	bool _dont_undo_redo() const;

private:
	static void _bind_methods();

	Ref<pg::VoxelGraphFunction> _graph;
	Ref<VoxelGeneratorGraph> _generator;
	uint32_t _node_id = ProgramGraph::NULL_ID;
	VoxelGraphEditor *_graph_editor = nullptr;
};

} // namespace zylann::voxel
