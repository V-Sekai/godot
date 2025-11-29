/**************************************************************************/
/*  backtracking.cpp                                                      */
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

#include "backtracking.h"
#include "graph_operations.h"

PlannerBacktracking::BacktrackResult PlannerBacktracking::backtrack(PlannerSolutionGraph p_graph, int p_parent_node_id, int p_current_node_id, Dictionary p_state, TypedArray<Variant> p_blacklisted_commands) {
	// Mark current node as failed
	p_graph.set_node_status(p_current_node_id, PlannerNodeStatus::STATUS_FAILED);
	
	// Reset current node's selected_method and state (IPyHOP-style)
	Dictionary current_node = p_graph.get_node(p_current_node_id);
	// Validate required dictionary keys exist
	if (!current_node.has("type")) {
		// Invalid node structure, return failure
		BacktrackResult result;
		result.parent_node_id = -1;
		result.current_node_id = -1;
		result.graph = p_graph;
		result.state = p_state;
		result.blacklisted_commands = p_blacklisted_commands;
		return result;
	}
	int current_node_type = current_node["type"];
	if (current_node_type == static_cast<int>(PlannerNodeType::TYPE_TASK) ||
			current_node_type == static_cast<int>(PlannerNodeType::TYPE_UNIGOAL) ||
			current_node_type == static_cast<int>(PlannerNodeType::TYPE_MULTIGOAL)) {
		current_node["selected_method"] = Variant();
		current_node["state"] = Dictionary();
		p_graph.update_node(p_current_node_id, current_node);
	}

	// Remove descendants of the failed node
	PlannerGraphOperations::remove_descendants(p_graph, p_current_node_id);
	
	// Remove the failed node from its parent's successors list
	// This prevents failed nodes from being considered as part of the solution path
	PlannerGraphOperations::remove_node_from_parent(p_graph, p_current_node_id);

	// Find the nearest ancestor that can be retried
	int new_parent_node_id = p_parent_node_id;

	// Traverse up the tree to find a node that can be retried
	while (new_parent_node_id >= 0) {
		Dictionary node = p_graph.get_node(new_parent_node_id);
		// Validate required dictionary keys exist
		if (!node.has("type")) {
			// Invalid node structure, return failure
			BacktrackResult result;
			result.parent_node_id = -1;
			result.current_node_id = -1;
			result.graph = p_graph;
			result.state = p_state;
			result.blacklisted_commands = p_blacklisted_commands;
			return result;
		}
		int node_type = node["type"];

		// Validate available_methods exists before accessing
		// Some node types may not have available_methods (e.g., ACTION nodes)
		TypedArray<Callable> available_methods;
		if (node.has("available_methods")) {
			Variant methods_var = node["available_methods"];
			// Try to convert to TypedArray<Callable> - handle both Array and TypedArray
			if (methods_var.get_type() == Variant::ARRAY) {
				Array methods_array = methods_var;
				// Convert to TypedArray and check size after conversion
				available_methods = TypedArray<Callable>(methods_array);
			}
		}

		// Check if this node has alternative methods
		bool can_retry = false;

		if (node_type == static_cast<int>(PlannerNodeType::TYPE_TASK) ||
				node_type == static_cast<int>(PlannerNodeType::TYPE_UNIGOAL) ||
				node_type == static_cast<int>(PlannerNodeType::TYPE_MULTIGOAL)) {
			// Check if there are available methods
			if (available_methods.size() > 0) {
				can_retry = true;
			}
		}

		if (can_retry) {
			// Found a node with available methods, retry it
			p_graph.set_node_status(new_parent_node_id, PlannerNodeStatus::STATUS_OPEN);
			
			// Reset selected_method and state (IPyHOP-style)
			// This allows the node to try all methods again from the beginning
			// Re-fetch node to get latest state after status change
			Dictionary updated_node = p_graph.get_node(new_parent_node_id);
			updated_node["selected_method"] = Variant();
			updated_node["state"] = Dictionary();
			p_graph.update_node(new_parent_node_id, updated_node);

			BacktrackResult result;
			// Return the retriable node as parent_node_id so planning can continue from it
			result.parent_node_id = new_parent_node_id;
			result.current_node_id = new_parent_node_id;
			result.graph = p_graph;
			result.state = p_state;
			result.blacklisted_commands = p_blacklisted_commands;
			return result;
		} else {
			// No more methods, this node also fails, continue backtracking
			p_graph.set_node_status(new_parent_node_id, PlannerNodeStatus::STATUS_FAILED);
			new_parent_node_id = PlannerGraphOperations::find_predecessor(p_graph, new_parent_node_id);
		}
	}

	// Reached root, return failure
	BacktrackResult result;
	result.parent_node_id = -1;
	result.current_node_id = -1;
	result.graph = p_graph;
	result.state = p_state;
	result.blacklisted_commands = p_blacklisted_commands;
	return result;
}
