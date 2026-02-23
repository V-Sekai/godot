/**************************************************************************/
/*  minimal_backtracking_domain.h                                         */
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

#include "../../src/domain.h"
#include "core/variant/callable.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"

// Minimal domain with backtracking: Task has two methods, first fails, second succeeds
namespace MinimalBacktrackingDomain {

// Minimal action: Increment a counter
Dictionary action_increment(Dictionary p_state, int p_amount) {
	Dictionary new_state = p_state.duplicate(true);

	// Get current value
	int current_value = 0;
	if (new_state.has("value")) {
		Dictionary value_dict = new_state["value"];
		if (value_dict.has("value")) {
			current_value = value_dict["value"];
		}
	} else {
		Dictionary value_dict;
		value_dict["value"] = 0;
		new_state["value"] = value_dict;
	}

	// Increment
	Dictionary value_dict = new_state["value"];
	value_dict["value"] = current_value + p_amount;
	new_state["value"] = value_dict;

	return new_state;
}

// Task method 1: Returns NIL (fails) - this will cause backtracking
Variant task_increment_method_fail(Dictionary p_state) {
	// Return NIL to indicate failure (Array() would be success)
	return Variant();
}

// Task method 2: Returns the increment action (succeeds)
Variant task_increment_method_succeed(Dictionary p_state) {
	Array result;
	Array action;
	action.push_back("action_increment");
	action.push_back(1); // increment by 1
	result.push_back(action);
	return result; // Array indicates success
}

// Callable wrapper for static methods
class MinimalBacktrackingDomainCallable {
public:
	static Dictionary action_increment(Dictionary p_state, int p_amount) {
		return MinimalBacktrackingDomain::action_increment(p_state, p_amount);
	}

	static Variant task_increment_method_fail(Dictionary p_state) {
		return MinimalBacktrackingDomain::task_increment_method_fail(p_state);
	}

	static Variant task_increment_method_succeed(Dictionary p_state) {
		return MinimalBacktrackingDomain::task_increment_method_succeed(p_state);
	}
};

// Helper: Create minimal backtracking domain
Ref<PlannerDomain> create_minimal_backtracking_domain() {
	Ref<PlannerDomain> domain = memnew(PlannerDomain);

	// Add action with explicit name to ensure it survives tools-disabled builds.
	domain->add_command("action_increment", callable_mp_static(&MinimalBacktrackingDomainCallable::action_increment));
	TypedArray<Callable> task_methods;
	task_methods.push_back(callable_mp_static(&MinimalBacktrackingDomainCallable::task_increment_method_fail));
	task_methods.push_back(callable_mp_static(&MinimalBacktrackingDomainCallable::task_increment_method_succeed));
	domain->add_task_methods("increment", task_methods);

	return domain;
}

} // namespace MinimalBacktrackingDomain
