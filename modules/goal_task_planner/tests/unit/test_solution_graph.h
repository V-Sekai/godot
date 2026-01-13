/**************************************************************************/
/*  test_solution_graph.h                                                 */
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

#include "modules/goal_task_planner/src/graph_operations.h"
#include "modules/goal_task_planner/src/solution_graph.h"
#include "tests/test_macros.h"

namespace TestSolutionGraph {

TEST_CASE("[Modules][Planner][SolutionGraph] Basic Operations") {
	PlannerSolutionGraph graph;

	SUBCASE("Initialization") {
		// Root node (0) should exist
		Dictionary root = graph.get_node(0);
		CHECK_FALSE(root.is_empty());
		CHECK(int(root["type"]) == (int)PlannerNodeType::TYPE_ROOT);
		CHECK(String(root["tag"]) == "old");

		Dictionary graph_dict = graph.get_graph();
		CHECK(graph_dict.has(0));
	}

	SUBCASE("Create Node") {
		Variant info = "command1";
		TypedArray<Callable> methods;
		int node_id = graph.create_node(PlannerNodeType::TYPE_COMMAND, info, methods);

		CHECK(node_id == 1);

		Dictionary node = graph.get_node(node_id);
		CHECK(int(node["type"]) == (int)PlannerNodeType::TYPE_COMMAND);
		CHECK(String(node["info"]) == "command1");
		CHECK(String(node["tag"]) == "new");
		CHECK(int(node["status"]) == (int)PlannerNodeStatus::STATUS_OPEN);
	}

	SUBCASE("Update Node") {
		int node_id = graph.create_node(PlannerNodeType::TYPE_TASK, "task1");
		Dictionary node = graph.get_node(node_id);

		node["status"] = (int)PlannerNodeStatus::STATUS_CLOSED;
		graph.update_node(node_id, node);

		Dictionary updated = graph.get_node(node_id);
		CHECK(int(updated["status"]) == (int)PlannerNodeStatus::STATUS_CLOSED);

		// Verify internal struct also updated
		const PlannerNodeStruct *internal = graph.get_node_internal(node_id);
		CHECK(internal->status == PlannerNodeStatus::STATUS_CLOSED);
	}
}

TEST_CASE("[Modules][Planner][GraphOps] Operations") {
	PlannerSolutionGraph graph;

	SUBCASE("Add Nodes and Edges") {
		// Setup parent node
		int parent_id = graph.create_node(PlannerNodeType::TYPE_TASK, "task1");

		// Mock dictionaries
		Dictionary action_dict;
		action_dict["action1"] = true;
		Dictionary task_dict;
		Dictionary unigoal_dict;
		TypedArray<Callable> multigoal_methods;

		Array children_info;
		children_info.push_back("action1");
		children_info.push_back("subtask1");

		// Returns last added node id? or count?
		// Header says: static int add_nodes_and_edges(...)
		PlannerGraphOperations::add_nodes_and_edges(graph, parent_id, children_info, action_dict, task_dict, unigoal_dict, multigoal_methods);

		// Should have created 2 nodes.
		// IDs: 0 (root), 1 (parent), 2 (action1), 3 (subtask1)
		// Check if nodes exist
		CHECK(graph.get_node(2).is_empty() == false);
		CHECK(graph.get_node(3).is_empty() == false);

		// Check edges (parent's children list)
		Dictionary parent = graph.get_node(parent_id);
		Array children = parent["successors"];
		CHECK(children.size() == 2);
		if (children.size() >= 2) {
			CHECK(int(children[0]) == 2);
			CHECK(int(children[1]) == 3);
		}

		// Check predecessors
		CHECK(PlannerGraphOperations::find_predecessor(graph, 2) == parent_id);
		CHECK(PlannerGraphOperations::find_predecessor(graph, 3) == parent_id);
	}

	SUBCASE("Find Open Node") {
		int n1 = graph.create_node(PlannerNodeType::TYPE_COMMAND, "a1"); // OPEN by default
		int n2 = graph.create_node(PlannerNodeType::TYPE_COMMAND, "a2");

		// Set n1 closed
		Dictionary d1 = graph.get_node(n1);
		d1["status"] = (int)PlannerNodeStatus::STATUS_CLOSED;
		Array d1_successors;
		d1_successors.push_back(n2);
		d1["successors"] = d1_successors; // Link n1 -> n2
		graph.update_node(n1, d1);

		// Link root -> n1
		Dictionary root = graph.get_node(0);
		Array root_successors;
		root_successors.push_back(n1);
		root["successors"] = root_successors;
		graph.update_node(0, root);

		// Let's test find_open_node on n1 (which has open child n2)
		Variant res = PlannerGraphOperations::find_open_node(graph, n1);
		CHECK(res.get_type() == Variant::INT);
		CHECK(int(res) == n2);

		// If child is closed?
		Dictionary d2 = graph.get_node(n2);
		d2["status"] = (int)PlannerNodeStatus::STATUS_CLOSED;
		graph.update_node(n2, d2);

		res = PlannerGraphOperations::find_open_node(graph, n1);
		CHECK(res.get_type() == Variant::NIL); // Should return null/nil if no open node found
	}

	SUBCASE("Remove Descendants") {
		int n1 = graph.create_node(PlannerNodeType::TYPE_TASK, "t1");
		int n2 = graph.create_node(PlannerNodeType::TYPE_COMMAND, "a1");

		Dictionary d1 = graph.get_node(n1);
		Array ch;
		ch.push_back(n2);
		d1["successors"] = ch;
		graph.update_node(n1, d1);

		// Remove descendants of n1 (should remove n2)
		PlannerGraphOperations::remove_descendants(graph, n1);

		// n2 should be gone?
		// get_node returns empty dict if not found
		CHECK(graph.get_node(n2).is_empty());

		// n1 should still exist
		CHECK_FALSE(graph.get_node(n1).is_empty());

		// n1's children list should be empty
		Dictionary d1_updated = graph.get_node(n1);
		Array kids = d1_updated["successors"];
		CHECK(kids.is_empty());
	}
}

} // namespace TestSolutionGraph
