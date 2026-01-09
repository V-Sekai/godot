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

#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/variant/callable.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"
#include "core/variant/variant.h"

// Node types matching Elixir planner
enum class PlannerNodeType {
	TYPE_ROOT = 0, // :D - Root node
	TYPE_COMMAND = 1, // :A - Command node
	TYPE_TASK = 2, // :T - Task node
	TYPE_UNIGOAL = 3, // :G - Unigoal node (internal use only, not for todo list input)
	TYPE_MULTIGOAL = 4, // :M - MultiGoal node
	TYPE_VERIFY_GOAL = 5, // :VG - Verify Unigoal node (internal use only)
	TYPE_VERIFY_MULTIGOAL = 6 // :VM - Verify MultiGoal node
};

// Node status matching Elixir planner
enum class PlannerNodeStatus {
	STATUS_OPEN = 0, // :O - Open (not yet processed)
	STATUS_CLOSED = 1, // :C - Closed/Completed (successfully processed)
	STATUS_FAILED = 2, // :F - Failed
	STATUS_NOT_APPLICABLE = 3 // :NA - Not applicable
};

class PlannerPlan; // Forward declaration

// Internal node structure for fast access (optimization)
struct PlannerNodeStruct {
	PlannerNodeType type;
	PlannerNodeStatus status;
	Variant info;
	LocalVector<int> successors; // Fast vector instead of TypedArray
	Dictionary state;
	Variant selected_method;
	TypedArray<Callable> available_methods;
	Callable command;
	int64_t start_time;
	int64_t end_time;
	int64_t duration;
	String tag;
	// Plan explanation and debugging
	Dictionary decision_info; // Stores why method was chosen, alternatives considered, scores, etc.

	PlannerNodeStruct() :
			type(PlannerNodeType::TYPE_ROOT),
			status(PlannerNodeStatus::STATUS_OPEN),
			start_time(0),
			end_time(0),
			duration(0),
			tag("new") {
		decision_info = Dictionary();
	}
};

class PlannerSolutionGraph {
private:
	// Fast internal structure using HashMap and LocalVector
	HashMap<int, PlannerNodeStruct> graph_internal;
	int next_node_id;

public:
	// Solution graph: Dictionary<int, Dictionary> where key is node_id
	// Kept for GDScript API compatibility (lazy conversion from graph_internal)
	Dictionary graph;

	PlannerSolutionGraph() {
		graph = Dictionary();
		next_node_id = 0;
		// Initialize root node (node 0) in internal structure
		PlannerNodeStruct root_node;
		root_node.type = PlannerNodeType::TYPE_ROOT;
		root_node.status = PlannerNodeStatus::STATUS_NOT_APPLICABLE;
		root_node.info = Variant("root");
		root_node.state = Dictionary();
		root_node.tag = "old"; // Root is always "old"
		graph_internal[0] = root_node;
		// Also set in Dictionary for API compatibility
		graph[0] = _node_struct_to_dictionary(root_node);
		next_node_id = 1;
	}

	// Create a new node and return its ID
	int create_node(PlannerNodeType p_type, Variant p_info, TypedArray<Callable> p_available_methods = TypedArray<Callable>(), Callable p_command = Callable()) {
		int node_id = next_node_id++;
		PlannerNodeStruct node;
		node.type = p_type;
		node.status = PlannerNodeStatus::STATUS_OPEN;
		// CRITICAL: Use deep copy to preserve nested arrays in Variant storage
		if (p_info.get_type() == Variant::ARRAY) {
			Array arr = p_info;
			node.info = arr.duplicate(true); // Deep copy to preserve nested arrays
		} else {
			node.info = p_info;
		}
		node.state = Dictionary();
		node.available_methods = p_available_methods;
		node.command = p_command;
		node.tag = "new"; // New nodes default to "new"
		graph_internal[node_id] = node;
		// Also update Dictionary for API compatibility
		graph[node_id] = _node_struct_to_dictionary(node);
		return node_id;
	}

	// Get node by ID (returns Dictionary for GDScript API)
	Dictionary get_node(int p_node_id) const {
		const PlannerNodeStruct *node = graph_internal.getptr(p_node_id);
		if (node == nullptr) {
			return Dictionary(); // Return empty dictionary if node doesn't exist
		}
		return _node_struct_to_dictionary(*node);
	}

	// Get node struct by ID (internal fast access)
	const PlannerNodeStruct *get_node_internal(int p_node_id) const {
		return graph_internal.getptr(p_node_id);
	}

	// Get mutable node struct by ID (internal fast access)
	PlannerNodeStruct *get_node_internal_mut(int p_node_id) {
		return graph_internal.getptr(p_node_id);
	}

	// Update node (from Dictionary - for backward compatibility)
	void update_node(int p_node_id, Dictionary p_node) {
		graph_internal[p_node_id] = _dictionary_to_node_struct(p_node);
		graph[p_node_id] = p_node; // Keep Dictionary in sync
	}

	// Update node struct (internal fast access)
	void update_node_internal(int p_node_id, const PlannerNodeStruct &p_node) {
		graph_internal[p_node_id] = p_node;
		graph[p_node_id] = _node_struct_to_dictionary(p_node); // Keep Dictionary in sync
	}

	// Get graph Dictionary (for graph operations that need direct access)
	// Returns cached Dictionary (updated on every modification)
	// This avoids expensive conversions on every call
	const Dictionary &get_graph() const {
		return graph;
	}

	// Get internal graph HashMap (for fast internal operations)
	const HashMap<int, PlannerNodeStruct> &get_graph_internal() const {
		return graph_internal;
	}

	// Get mutable internal graph HashMap (for operations that need to modify)
	HashMap<int, PlannerNodeStruct> &get_graph_internal_mut() {
		return graph_internal;
	}

	// Helper for loading graphs - convert Dictionary to internal structure
	void load_from_dictionary(const Dictionary &p_graph) {
		graph_internal.clear();
		graph = Dictionary();
		Array graph_keys = p_graph.keys();
		for (int i = 0; i < graph_keys.size(); i++) {
			Variant key = graph_keys[i];
			if (key.get_type() != Variant::INT) {
				continue;
			}
			int node_id = key;
			Dictionary node_dict = p_graph[node_id];
			PlannerNodeStruct node = _dictionary_to_node_struct(node_dict);
			graph_internal[node_id] = node;
			graph[node_id] = node_dict; // Keep Dictionary in sync
		}
		// Find max node ID to set next_node_id
		int max_id = -1;
		for (int i = 0; i < graph_keys.size(); i++) {
			Variant key = graph_keys[i];
			if (key.get_type() == Variant::INT) {
				int node_id = key;
				if (node_id > max_id) {
					max_id = node_id;
				}
			}
		}
		next_node_id = (max_id >= 0) ? (max_id + 1) : 1;
	}

	// Add successor to a node (optimized: uses internal structure)
	void add_successor(int p_parent_id, int p_child_id) {
		PlannerNodeStruct *parent = graph_internal.getptr(p_parent_id);
		if (parent != nullptr) {
			parent->successors.push_back(p_child_id);
			// Update Dictionary for API compatibility
			graph[p_parent_id] = _node_struct_to_dictionary(*parent);
		}
	}

	// Set node status (optimized: uses internal structure)
	void set_node_status(int p_node_id, PlannerNodeStatus p_status) {
		PlannerNodeStruct *node = graph_internal.getptr(p_node_id);
		if (node != nullptr) {
			node->status = p_status;
			// Update Dictionary for API compatibility
			graph[p_node_id] = _node_struct_to_dictionary(*node);
		}
	}

	// Save state snapshot at node (optimized: uses internal structure)
	void save_state_snapshot(int p_node_id, Dictionary p_state) {
		PlannerNodeStruct *node = graph_internal.getptr(p_node_id);
		if (node != nullptr) {
			// CRITICAL: Use deep duplicate to ensure nested dictionaries are copied
			node->state = p_state.duplicate(true);
			// Update Dictionary for API compatibility
			graph[p_node_id] = _node_struct_to_dictionary(*node);
		}
	}

	// Get state snapshot from node (optimized: uses internal structure)
	Dictionary get_state_snapshot(int p_node_id) const {
		const PlannerNodeStruct *node = graph_internal.getptr(p_node_id);
		if (node == nullptr || node->state.is_empty()) {
			return Dictionary();
		}
		// CRITICAL: Use deep duplicate to ensure nested dictionaries are copied
		return node->state.duplicate(true);
	}

	// Set node tag ("new" or "old") (optimized: uses internal structure)
	void set_node_tag(int p_node_id, const String &p_tag) {
		PlannerNodeStruct *node = graph_internal.getptr(p_node_id);
		if (node != nullptr) {
			node->tag = p_tag;
			// Update Dictionary for API compatibility
			graph[p_node_id] = _node_struct_to_dictionary(*node);
		}
	}

	// Get node tag (optimized: uses internal structure)
	String get_node_tag(int p_node_id) const {
		const PlannerNodeStruct *node = graph_internal.getptr(p_node_id);
		if (node == nullptr) {
			return String("new"); // Default to "new" if node doesn't exist
		}
		return node->tag;
	}

	// Get next node ID (for graph operations)
	int get_next_node_id() const {
		return next_node_id;
	}

	// Set next node ID (for loading graphs)
	void set_next_node_id(int p_next_id) {
		next_node_id = p_next_id;
	}

private:
	// Helper: Convert NodeStruct to Dictionary (for GDScript API)
	Dictionary _node_struct_to_dictionary(const PlannerNodeStruct &p_node) const {
		Dictionary dict;
		dict["type"] = static_cast<int>(p_node.type);
		dict["status"] = static_cast<int>(p_node.status);
		dict["info"] = p_node.info;
		// Convert LocalVector<int> to TypedArray<int>
		TypedArray<int> successors_array;
		for (size_t i = 0; i < p_node.successors.size(); i++) {
			successors_array.push_back(p_node.successors[i]);
		}
		dict["successors"] = successors_array;
		dict["state"] = p_node.state;
		dict["selected_method"] = p_node.selected_method;
		dict["available_methods"] = p_node.available_methods;
		dict["command"] = p_node.command;
		dict["start_time"] = Variant(static_cast<int64_t>(p_node.start_time));
		dict["end_time"] = Variant(static_cast<int64_t>(p_node.end_time));
		dict["duration"] = Variant(static_cast<int64_t>(p_node.duration));
		dict["tag"] = Variant(p_node.tag);
		dict["decision_info"] = p_node.decision_info;
		return dict;
	}

	// Helper: Convert Dictionary to NodeStruct (for backward compatibility)
	PlannerNodeStruct _dictionary_to_node_struct(const Dictionary &p_dict) const {
		PlannerNodeStruct node;
		if (p_dict.has("type")) {
			node.type = static_cast<PlannerNodeType>(static_cast<int>(p_dict["type"]));
		}
		if (p_dict.has("status")) {
			node.status = static_cast<PlannerNodeStatus>(static_cast<int>(p_dict["status"]));
		}
		if (p_dict.has("info")) {
			node.info = p_dict["info"];
		}
		if (p_dict.has("successors")) {
			// Convert TypedArray<int> to LocalVector<int>
			TypedArray<int> successors_array = p_dict["successors"];
			node.successors.clear();
			for (int i = 0; i < successors_array.size(); i++) {
				node.successors.push_back(successors_array[i]);
			}
		}
		if (p_dict.has("state")) {
			node.state = p_dict["state"];
		}
		if (p_dict.has("selected_method")) {
			node.selected_method = p_dict["selected_method"];
		}
		if (p_dict.has("available_methods")) {
			Variant methods_var = p_dict["available_methods"];
			if (methods_var.get_type() == Variant::ARRAY) {
				node.available_methods = TypedArray<Callable>(methods_var);
			}
		}
		if (p_dict.has("command")) {
			node.command = p_dict["command"];
		}
		if (p_dict.has("start_time")) {
			node.start_time = static_cast<int64_t>(p_dict["start_time"]);
		}
		if (p_dict.has("end_time")) {
			node.end_time = static_cast<int64_t>(p_dict["end_time"]);
		}
		if (p_dict.has("duration")) {
			node.duration = static_cast<int64_t>(p_dict["duration"]);
		}
		if (p_dict.has("tag")) {
			node.tag = p_dict["tag"];
		}
		if (p_dict.has("decision_info")) {
			node.decision_info = p_dict["decision_info"];
		}
		return node;
	}
};
