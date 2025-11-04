/**************************************************************************/
/*  goal_solver.h                                                         */
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

#include "core/string/ustring.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/variant/dictionary.h"
#include "core/variant/array.h"
#include "core/variant/callable.h"
#include "planner_metadata.h"

// Goal solver for optimizing unigoal ordering and handling temporal metadata
// Uses internal data structures (HashMap, LocalVector) for efficiency
class PlannerGoalSolver {
private:
	// Constraining factor for a goal/task (fewer methods = more constraining)
	struct ConstrainingFactor {
		int method_count;
		bool has_temporal_constraints;
		
		ConstrainingFactor() : method_count(0), has_temporal_constraints(false) {}
		ConstrainingFactor(int p_count, bool p_temporal) : method_count(p_count), has_temporal_constraints(p_temporal) {}
		
		// Compare: more constraining = fewer methods, or has temporal constraints
		bool operator<(const ConstrainingFactor &p_other) const {
			if (has_temporal_constraints != p_other.has_temporal_constraints) {
				return has_temporal_constraints; // Temporal constraints make it more constraining
			}
			return method_count < p_other.method_count; // Fewer methods = more constraining
		}
	};
	
	// Internal storage for goal ordering
	struct GoalWithFactor {
		Variant goal;
		ConstrainingFactor factor;
		
		GoalWithFactor(const Variant &p_goal, const ConstrainingFactor &p_factor) : goal(p_goal), factor(p_factor) {}
	};
	
	// Calculate constraining factor for a unigoal
	ConstrainingFactor calculate_constraining_factor(const Variant &p_goal, const Dictionary &p_state, const Dictionary &p_unigoal_method_dict) const;
	
	// Extract temporal constraints from goal/item
	PlannerMetadata extract_temporal_constraints(const Variant &p_item) const;
	
public:
	PlannerGoalSolver() {}
	~PlannerGoalSolver() {}
	
	// Optimize unigoal order: most constraining first, least constraining last
	// Uses HashMap/LocalVector internally for efficiency
	Array optimize_unigoal_order(const Array &p_unigoals, const Dictionary &p_state, const Dictionary &p_unigoal_method_dict);
	
	// Attach temporal constraints to commands, multigoals, unigoals, actions, tasks
	Variant attach_temporal_constraints(const Variant &p_item, const Dictionary &p_temporal_constraints);
	
	// Extract temporal constraints from item
	Dictionary get_temporal_constraints(const Variant &p_item) const;
	
	// Check if item has temporal constraints
	bool has_temporal_constraints(const Variant &p_item) const;
};

