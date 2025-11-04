/**************************************************************************/
/*  graph_operations.h                                                    */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#pragma once

// SPDX-FileCopyrightText: 2025-present K. S. Ernest (iFire) Lee
// SPDX-License-Identifier: MIT

#include "solution_graph.h"
#include "domain.h"
#include "multigoal.h"
#include "core/variant/variant.h"

class PlannerGraphOperations {
public:
	// Determine node type from node_info
	static PlannerNodeType get_node_type(Variant p_node_info, Dictionary p_action_dict, Dictionary p_task_dict, Dictionary p_unigoal_dict);
	
	// Add nodes and edges to solution graph
	static int add_nodes_and_edges(PlannerSolutionGraph &p_graph, int p_parent_node_id, Array p_children_node_info_list, Dictionary p_action_dict, Dictionary p_task_dict, Dictionary p_unigoal_dict, TypedArray<Callable> p_multigoal_methods);
	
	// Find first open node in successors of parent
	static Variant find_open_node(PlannerSolutionGraph &p_graph, int p_parent_node_id);
	
	// Find predecessor of a node
	static int find_predecessor(PlannerSolutionGraph &p_graph, int p_node_id);
	
	// Remove descendants of a node
	static void remove_descendants(PlannerSolutionGraph &p_graph, int p_node_id);
	
	// Extract solution plan (sequence of actions) from graph
	static Array extract_solution_plan(PlannerSolutionGraph &p_graph);
	
private:
	static void do_get_descendants(PlannerSolutionGraph &p_graph, TypedArray<int> p_current_nodes, TypedArray<int> &p_visited, TypedArray<int> &p_result);
};

