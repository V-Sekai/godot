/**************************************************************************/
/*  multigoal.cpp                                                         */
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

#include "multigoal.h"

#include "core/object/class_db.h"

void PlannerMultigoal::_bind_methods() {
	ClassDB::bind_static_method("PlannerMultigoal", D_METHOD("is_multigoal_array", "variant"), &PlannerMultigoal::is_multigoal_array);
	ClassDB::bind_static_method("PlannerMultigoal", D_METHOD("get_goal_tag", "multigoal"), &PlannerMultigoal::get_goal_tag);
	ClassDB::bind_static_method("PlannerMultigoal", D_METHOD("set_goal_tag", "multigoal", "tag"), &PlannerMultigoal::set_goal_tag);
	ClassDB::bind_static_method("PlannerMultigoal", D_METHOD("method_goals_not_achieved", "state", "multigoal_array"), &PlannerMultigoal::method_goals_not_achieved);
	ClassDB::bind_static_method("PlannerMultigoal", D_METHOD("method_verify_multigoal", "state", "method", "multigoal_array", "depth", "verbose"), &PlannerMultigoal::method_verify_multigoal);
}

// Check if a Variant is an Array multigoal (Array of unigoal arrays)
// A multigoal is an Array where the first element is also an Array (unigoal)
// This distinguishes it from a single unigoal which is [predicate, subject, value]
bool PlannerMultigoal::is_multigoal_array(const Variant &p_variant) {
	if (p_variant.get_type() == Variant::DICTIONARY) {
		// Check if it's a wrapped multigoal
		Dictionary dict = p_variant;
		if (dict.has("item")) {
			Variant item = dict["item"];
			if (item.get_type() == Variant::ARRAY) {
				Array arr = item;
				if (!arr.is_empty() && arr[0].get_type() == Variant::ARRAY) {
					return true;
				}
			}
		}
		return false;
	}
	if (p_variant.get_type() != Variant::ARRAY) {
		return false;
	}
	Array arr = p_variant;
	if (arr.is_empty()) {
		return false; // Empty array is not a multigoal
	}
	// Check if first element is also an Array (unigoal)
	Variant first = arr[0];
	return first.get_type() == Variant::ARRAY;
}

String PlannerMultigoal::get_goal_tag(const Variant &p_multigoal) {
	if (p_multigoal.get_type() == Variant::DICTIONARY) {
		Dictionary dict = p_multigoal;
		if (dict.has("goal_tag")) {
			return dict["goal_tag"];
		}
	}
	return String(); // No tag found
}

Variant PlannerMultigoal::set_goal_tag(const Variant &p_multigoal, const String &p_tag) {
	Variant actual_multigoal = p_multigoal;

	// Unwrap if already wrapped
	if (p_multigoal.get_type() == Variant::DICTIONARY) {
		Dictionary dict = p_multigoal;
		if (dict.has("item")) {
			actual_multigoal = dict["item"];
		} else {
			// Already a dictionary, just add/update tag
			Dictionary result = dict.duplicate();
			result["goal_tag"] = p_tag;
			return result;
		}
	}

	// Wrap in dictionary with tag
	Dictionary result;
	result["item"] = actual_multigoal;
	result["goal_tag"] = p_tag;
	return result;
}

// Return Array of unigoals that are not yet achieved
// Each unigoal is [predicate, subject, value]
Array PlannerMultigoal::method_goals_not_achieved(const Dictionary &p_state, const Array &p_multigoal_array) {
	Array unmatched_goals;

	for (int i = 0; i < p_multigoal_array.size(); ++i) {
		Variant goal_variant = p_multigoal_array[i];
		if (goal_variant.get_type() != Variant::ARRAY) {
			continue; // Skip invalid unigoals
		}
		Array unigoal = goal_variant;
		if (unigoal.size() < 3) {
			continue; // Invalid unigoal format (must be 3-element: [predicate, subject, value])
		}

		String predicate = unigoal[0]; // e.g., "affection", "study_points", "relationship_points"
		String subject = unigoal[1]; // e.g., "protagonist_class_president", "yuki", "relationship_points_yuki_maya"
		Variant desired_value = unigoal[2]; // e.g., 50, 10, 3

		// Subject should never be empty
		if (subject.is_empty()) {
			continue; // Invalid unigoal format - subject is required
		}

		// Check if goal is achieved (exact equality)
		// Note: "At least" semantics (>=) should be handled at the domain level
		// by task methods and multigoal methods, not in the planner core
		bool achieved = false;

		// Handle flat predicates with 3-element format: ["relationship_points", "relationship_points_yuki_maya", 3]
		// In this case, predicate is "relationship_points" and subject is the full flat predicate
		// Check state[subject] == value (i.e., state["relationship_points_yuki_maya"] == 3)
		if (predicate == "relationship_points" && subject.begins_with("relationship_points_")) {
			if (p_state.has(subject)) {
				Variant flat_predicate_val = p_state[subject];
				if (flat_predicate_val == desired_value) {
					achieved = true;
				}
			}
		} else if (p_state.has(predicate)) {
			// Handle nested predicates: state[predicate][subject] == value
			// e.g., state["study_points"]["yuki"] == 10
			Variant predicate_val = p_state[predicate];
			if (predicate_val.get_type() == Variant::DICTIONARY) {
				Dictionary predicate_dict = predicate_val;
				if (predicate_dict.has(subject)) {
					Variant subject_val = predicate_dict[subject];

					// Handle nested predicates where value is a Dictionary {companion: companion_id, target: target_value}
					// Format: ["relationship_points", "yuki", {"companion": "maya", "target": 3}]
					// State structure: state["relationship_points"]["yuki"]["maya"] = 3
					if (desired_value.get_type() == Variant::DICTIONARY && subject_val.get_type() == Variant::DICTIONARY) {
						Dictionary value_dict = desired_value;
						if (value_dict.has("companion") && value_dict.has("target")) {
							String companion_id = value_dict["companion"];
							Variant target_value = value_dict["target"];
							Dictionary subject_dict = subject_val;
							if (subject_dict.has(companion_id) && subject_dict[companion_id] == target_value) {
								achieved = true;
							}
						}
					} else {
						// Standard case: state[predicate][subject] == value
						if (subject_val == desired_value) {
							achieved = true;
						}
					}
				}
			}
		}

		if (!achieved) {
			unmatched_goals.push_back(unigoal);
		}
	}

	return unmatched_goals;
}

Variant PlannerMultigoal::method_verify_multigoal(const Dictionary &p_state, const String &p_method, const Array &p_multigoal_array, int p_depth, int p_verbose) {
	Array unmatched_goals = method_goals_not_achieved(p_state, p_multigoal_array);
	if (unmatched_goals.size() > 0) {
		if (p_verbose >= 3) {
			print_line(vformat("Depth %d: method %s didn't achieve multigoal, %d goals remaining", p_depth, p_method, unmatched_goals.size()));
		}
		return false;
	}

	if (p_verbose >= 3) {
		print_line(vformat("Depth %d: method %s achieved multigoal", p_depth, p_method));
	}
	return Array();
}
