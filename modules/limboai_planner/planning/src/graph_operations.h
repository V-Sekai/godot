/**************************************************************************/
/*  graph_operations.h                                                    */
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

// SPDX-FileCopyrightText: 2025-present K. S. Ernest (iFire) Lee
// SPDX-License-Identifier: MIT

#include "core/variant/variant.h"
#include "domain.h"
#include "solution_graph.h"

class PlannerGraphOperations {
public:
	// Determine node type from node_info
	static PlannerNodeType get_node_type(Variant p_node_info, Dictionary p_command_dict, Dictionary p_task_dict, Dictionary p_unigoal_dict, int p_verbose = 0);

	// Add nodes and edges to solution graph
	static int add_nodes_and_edges(PlannerSolutionGraph &p_graph, int p_parent_node_id, Array p_children_node_info_list, Dictionary p_command_dict, Dictionary p_task_dict, Dictionary p_unigoal_dict, TypedArray<Callable> p_multigoal_methods, int p_verbose = 0);

	// Find first open node in successors of parent
	static Variant find_open_node(PlannerSolutionGraph &p_graph, int p_parent_node_id);

	// Find predecessor of a node
	static int find_predecessor(PlannerSolutionGraph &p_graph, int p_node_id);

	// Remove descendants of a node
	// If p_also_remove_from_parent is true, also remove the node itself from its parent's successors list
	static void remove_descendants(PlannerSolutionGraph &p_graph, int p_node_id, bool p_also_remove_from_parent = false);

	// Extract solution plan (sequence of commands) from graph
	// Commands with temporal constraints are sorted by start_time (STN-Based Plan Extraction)
	// Uses start_time, or calculates from (end_time - duration) if start_time not available
	// Commands without temporal constraints maintain DFS order
	static Array extract_solution_plan(PlannerSolutionGraph &p_graph, int p_verbose = 0);
	// Extract only "new" commands from graph (for replanning)
	static Array extract_new_commands(PlannerSolutionGraph &p_graph);
	// Execute commands directly from solution graph, returning final state
	// Takes initial state and domain, executes commands as it traverses the graph
	// Uses domain->command_dictionary to look up command callables
	static Dictionary execute_solution_graph(PlannerSolutionGraph &p_graph, Dictionary p_initial_state, Ref<PlannerDomain> p_domain);

private:
	// Helper for remove_descendants - collects all descendant node IDs
	static void do_get_descendants(PlannerSolutionGraph &p_graph, TypedArray<int> p_current_nodes, TypedArray<int> &p_visited, TypedArray<int> &p_result);
};
