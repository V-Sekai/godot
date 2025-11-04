/**************************************************************************/
/*  goal_solver.cpp                                                       */
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

#include "goal_solver.h"
#include "core/variant/callable.h"
#include "core/variant/typed_array.h"
#include "core/templates/sort_array.h"

PlannerGoalSolver::ConstrainingFactor PlannerGoalSolver::calculate_constraining_factor(const Variant &p_goal, const Dictionary &p_state, const Dictionary &p_unigoal_method_dict) const {
	ConstrainingFactor factor;
	
	// Extract goal info (assuming format: [state_var_name, argument, desired_value])
	if (p_goal.get_type() != Variant::ARRAY) {
		return factor;
	}
	
	Array goal_arr = p_goal;
	if (goal_arr.size() < 3) {
		return factor;
	}
	
	String state_var_name = goal_arr[0];
	
	// Count available methods for this unigoal
	if (p_unigoal_method_dict.has(state_var_name)) {
		Variant methods_var = p_unigoal_method_dict[state_var_name];
		if (methods_var.get_type() == Variant::ARRAY) {
			TypedArray<Callable> methods = methods_var;
			factor.method_count = methods.size();
		}
	}
	
	// Check for temporal constraints
	PlannerMetadata metadata = extract_temporal_constraints(p_goal);
	if (!metadata.start_time.is_empty() || !metadata.end_time.is_empty()) {
		factor.has_temporal_constraints = true;
	}
	
	return factor;
}

PlannerMetadata PlannerGoalSolver::extract_temporal_constraints(const Variant &p_item) const {
	PlannerMetadata metadata;
	
	// Check if item has temporal_constraints field
	if (p_item.get_type() == Variant::DICTIONARY) {
		Dictionary item_dict = p_item;
		if (item_dict.has("temporal_constraints")) {
			Dictionary temporal_dict = item_dict["temporal_constraints"];
			metadata = PlannerMetadata::from_dictionary(temporal_dict);
		}
	} else if (p_item.get_type() == Variant::ARRAY) {
		Array item_arr = p_item;
		// Temporal constraints might be stored as last element or in a wrapper
		// Check if last element is a dictionary with temporal constraints
		if (item_arr.size() > 0) {
			Variant last = item_arr[item_arr.size() - 1];
			if (last.get_type() == Variant::DICTIONARY) {
				Dictionary last_dict = last;
				if (last_dict.has("temporal_constraints")) {
					Dictionary temporal_dict = last_dict["temporal_constraints"];
					metadata = PlannerMetadata::from_dictionary(temporal_dict);
				}
			}
		}
	}
	
	return metadata;
}

Array PlannerGoalSolver::optimize_unigoal_order(const Array &p_unigoals, const Dictionary &p_state, const Dictionary &p_unigoal_method_dict) {
	// Use LocalVector internally for efficiency
	LocalVector<GoalWithFactor> goals_with_factors;
	
	// Calculate constraining factors for each unigoal
	for (int i = 0; i < p_unigoals.size(); i++) {
		Variant goal = p_unigoals[i];
		ConstrainingFactor factor = calculate_constraining_factor(goal, p_state, p_unigoal_method_dict);
		goals_with_factors.push_back(GoalWithFactor(goal, factor));
	}
	
	// Sort by constraining factor (most constraining first)
	// Use insertion sort for small arrays, or std::sort-like approach
	if (goals_with_factors.size() > 1) {
		// Simple insertion sort: most constraining first
		for (uint32_t i = 1; i < goals_with_factors.size(); i++) {
			GoalWithFactor key = goals_with_factors[i];
			int j = i - 1;
			
			// Move elements with less constraining factors to the right
			while (j >= 0 && goals_with_factors[j].factor < key.factor) {
				goals_with_factors[j + 1] = goals_with_factors[j];
				j--;
			}
			goals_with_factors[j + 1] = key;
		}
	}
	
	// Convert back to Array for GDScript interface
	Array ordered_goals;
	ordered_goals.resize(goals_with_factors.size());
	for (uint32_t i = 0; i < goals_with_factors.size(); i++) {
		ordered_goals[i] = goals_with_factors[i].goal;
	}
	
	return ordered_goals;
}

Variant PlannerGoalSolver::attach_temporal_metadata(const Variant &p_item, const Dictionary &p_temporal_metadata) {
	PlannerMetadata metadata = PlannerMetadata::from_dictionary(p_temporal_metadata);
	
	// Create a wrapper dictionary with the item and temporal metadata
	Dictionary result;
	
	if (p_item.get_type() == Variant::DICTIONARY) {
		// If already a dictionary, add temporal_metadata field
		result = Dictionary(p_item);
		result["temporal_metadata"] = metadata.to_dictionary();
	} else if (p_item.get_type() == Variant::ARRAY) {
		// If array, wrap in dictionary with temporal metadata
		result["item"] = p_item;
		result["temporal_metadata"] = metadata.to_dictionary();
	} else {
		// For other types, wrap in dictionary
		result["item"] = p_item;
		result["temporal_metadata"] = metadata.to_dictionary();
	}
	
	return result;
}

Dictionary PlannerGoalSolver::get_temporal_metadata(const Variant &p_item) const {
	PlannerMetadata metadata = extract_temporal_metadata(p_item);
	return metadata.to_dictionary();
}

bool PlannerGoalSolver::has_temporal_metadata(const Variant &p_item) const {
	PlannerMetadata metadata = extract_temporal_metadata(p_item);
	return metadata.is_valid();
}

