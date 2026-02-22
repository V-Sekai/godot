/**************************************************************************/
/*  backtracking.h                                                        */
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

#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"
#include "graph_operations.h"
#include "solution_graph.h"

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
	// p_verbose: verbosity level (0=none, 1=basic, 2=detailed, 3=very detailed)
	static BacktrackResult backtrack(PlannerSolutionGraph p_graph, int p_parent_node_id, int p_current_node_id, Dictionary p_state, TypedArray<Variant> p_blacklisted_commands, int p_verbose = 0);
};
