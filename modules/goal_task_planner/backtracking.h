/**************************************************************************/
/*  backtracking.h                                                        */
/**************************************************************************/

#pragma once

// SPDX-FileCopyrightText: 2025-present K. S. Ernest (iFire) Lee
// SPDX-License-Identifier: MIT

#include "solution_graph.h"
#include "graph_operations.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"

class PlannerBacktracking {
public:
	struct BacktrackResult {
		int parent_node_id;
		int current_node_id;
		PlannerSolutionGraph graph;
		Dictionary state;
		TypedArray<Variant> blacklisted_commands;
	};
	
	// Backtrack from a failed node
	static BacktrackResult backtrack(PlannerSolutionGraph p_graph, int p_parent_node_id, int p_current_node_id, Dictionary p_state, TypedArray<Variant> p_blacklisted_commands);
};

