/**************************************************************************/
/*  planner_result.h                                                      */
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

#include "core/io/resource.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"
#include "solution_graph.h"

class PlannerResult : public Resource {
	GDCLASS(PlannerResult, Resource);

private:
	Dictionary final_state;
	Dictionary solution_graph; // The graph Dictionary from PlannerSolutionGraph
	bool success;

public:
	PlannerResult();

	Dictionary get_final_state() const { return final_state; }
	void set_final_state(Dictionary p_state) { final_state = p_state; }

	Dictionary get_solution_graph() const { return solution_graph; }
	void set_solution_graph(Dictionary p_graph) { solution_graph = p_graph; }

	bool get_success() const { return success; }
	void set_success(bool p_success) { success = p_success; }

	// Extract array of actions from the solution graph
	Array extract_plan(int p_verbose = 0) const;

	// Helper methods for working with the solution graph
	Array find_failed_nodes() const;
	Dictionary get_node(int p_node_id) const;
	Array get_all_nodes() const;
	bool has_node(int p_node_id) const;

	// Plan explanation and debugging methods
	Dictionary explain_plan() const; // Get explanation of why this plan was chosen
	Array get_alternative_methods(int p_node_id) const; // Get alternative methods considered for a node
	Dictionary get_decision_path(int p_node_id) const; // Get decision path from root to node
	Dictionary to_graph_json() const; // Export plan graph as JSON for visualization
	String get_node_explanation(int p_node_id) const; // Get human-readable explanation for a node

protected:
	static void _bind_methods();
};
