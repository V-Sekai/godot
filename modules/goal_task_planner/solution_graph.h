/**************************************************************************/
/*  solution_graph.h                                                      */
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

#include "core/variant/callable.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"
#include "core/variant/variant.h"

// Node types matching Elixir planner
enum class PlannerNodeType {
	TYPE_ROOT = 0, // :D - Root node
	TYPE_ACTION = 1, // :A - Action node
	TYPE_TASK = 2, // :T - Task node
	TYPE_GOAL = 3, // :G - Goal node
	TYPE_MULTIGOAL = 4, // :M - MultiGoal node
	TYPE_VERIFY_GOAL = 5, // :VG - Verify Goal node
	TYPE_VERIFY_MULTIGOAL = 6 // :VM - Verify MultiGoal node
};

// Node status matching Elixir planner
enum class PlannerNodeStatus {
	STATUS_OPEN = 0, // :O - Open (not yet processed)
	STATUS_CLOSED = 1, // :C - Closed/Completed (successfully processed)
	STATUS_FAILED = 2, // :F - Failed
	STATUS_NOT_APPLICABLE = 3 // :NA - Not applicable
};

class PlannerSolutionGraph {
public:
	// Solution graph: Dictionary<int, Dictionary> where key is node_id
	Dictionary graph;
	int next_node_id;

	PlannerSolutionGraph() {
		graph = Dictionary();
		next_node_id = 0;
		// Initialize root node (node 0)
		Dictionary root_node;
		root_node["type"] = static_cast<int>(PlannerNodeType::TYPE_ROOT);
		root_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_NOT_APPLICABLE);
		root_node["info"] = Variant("root");
		root_node["successors"] = TypedArray<int>();
		root_node["state"] = Dictionary();
		root_node["selected_method"] = Variant();
		root_node["available_methods"] = TypedArray<Callable>();
		root_node["action"] = Variant();
		root_node["start_time"] = Variant(static_cast<int64_t>(0));
		root_node["end_time"] = Variant(static_cast<int64_t>(0));
		root_node["duration"] = Variant(static_cast<int64_t>(0));
		graph[0] = root_node;
		next_node_id = 1;
	}

	// Create a new node and return its ID
	int create_node(PlannerNodeType p_type, Variant p_info, TypedArray<Callable> p_available_methods = TypedArray<Callable>(), Callable p_action = Callable()) {
		int node_id = next_node_id++;
		Dictionary node;
		node["type"] = static_cast<int>(p_type);
		node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_OPEN);
		node["info"] = p_info;
		node["successors"] = TypedArray<int>();
		node["state"] = Dictionary();
		node["selected_method"] = Variant();
		node["available_methods"] = p_available_methods;
		node["action"] = p_action;
		node["start_time"] = Variant(static_cast<int64_t>(0));
		node["end_time"] = Variant(static_cast<int64_t>(0));
		node["duration"] = Variant(static_cast<int64_t>(0));
		graph[node_id] = node;
		return node_id;
	}

	// Get node by ID
	Dictionary get_node(int p_node_id) const {
		return graph[p_node_id];
	}

	// Update node
	void update_node(int p_node_id, Dictionary p_node) {
		graph[p_node_id] = p_node;
	}

	// Get graph Dictionary (for graph operations that need direct access)
	Dictionary &get_graph() {
		return graph;
	}

	// Add successor to a node
	void add_successor(int p_parent_id, int p_child_id) {
		Dictionary parent = graph[p_parent_id];
		TypedArray<int> successors = parent["successors"];
		successors.push_back(p_child_id);
		parent["successors"] = successors;
		graph[p_parent_id] = parent;
	}

	// Set node status
	void set_node_status(int p_node_id, PlannerNodeStatus p_status) {
		Dictionary node = graph[p_node_id];
		node["status"] = static_cast<int>(p_status);
		graph[p_node_id] = node;
	}

	// Save state snapshot at node
	void save_state_snapshot(int p_node_id, Dictionary p_state) {
		Dictionary node = graph[p_node_id];
		node["state"] = p_state.duplicate();
		graph[p_node_id] = node;
	}

	// Get state snapshot from node
	Dictionary get_state_snapshot(int p_node_id) const {
		Dictionary node = graph[p_node_id];
		Dictionary state = node["state"];
		if (state.is_empty()) {
			return Dictionary();
		}
		return state.duplicate();
	}
};
