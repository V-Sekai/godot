/**************************************************************************/
/*  test_helpers.h                                                        */
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
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "../../planner_result.h"
#include "core/variant/array.h"
#include "core/variant/variant.h"

namespace TestComprehensivePlanner {

// Helper function to validate a plan result
// Returns true if result is valid and successful
// Optionally checks that plan is not empty when expect_non_empty is true
static bool is_valid_plan_result(Ref<PlannerResult> result, bool expect_non_empty = false) {
	if (!result.is_valid() || !result->get_success()) {
		// Planning failed - this is NOT a valid plan
		return false;
	}
	Array plan = result->extract_plan();
	if (expect_non_empty && plan.is_empty()) {
		// Expected non-empty plan but got empty - this is NOT a valid plan
		return false;
	}
	return true;
}

// Helper function to check plan contains expected action
// Plan is Array of action arrays like [["action_name", arg1, arg2], ...]
static bool plan_contains_action(Array plan, String action_name) {
	for (int i = 0; i < plan.size(); i++) {
		Variant item = plan[i];
		if (item.get_type() == Variant::ARRAY) {
			Array action = item;
			if (!action.is_empty()) {
				Variant first = action[0];
				// Handle both string and unwrapped dictionary cases
				if (first.get_type() == Variant::STRING && first == action_name) {
					return true;
				}
				// Handle dictionary-wrapped actions
				if (first.get_type() == Variant::DICTIONARY) {
					Dictionary dict = first;
					if (dict.has("item")) {
						Variant unwrapped = dict["item"];
						if (unwrapped.get_type() == Variant::ARRAY) {
							Array unwrapped_arr = unwrapped;
							if (!unwrapped_arr.is_empty() && unwrapped_arr[0] == action_name) {
								return true;
							}
						}
					}
				}
			}
		}
	}
	return false;
}

// Helper to validate plan structure matches expected fixture
// expected_min_actions: minimum number of actions expected in plan
// expected_actions: array of action names that should be present
static bool validate_plan_against_fixture(Array plan, int expected_min_actions, Array expected_actions = Array()) {
	if (plan.size() < expected_min_actions) {
		return false;
	}

	// Check that expected actions are present
	for (int i = 0; i < expected_actions.size(); i++) {
		String action_name = expected_actions[i];
		if (!plan_contains_action(plan, action_name)) {
			return false;
		}
	}

	return true;
}

} // namespace TestComprehensivePlanner

