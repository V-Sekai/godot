/**************************************************************************/
/*  planning_test_domains.h                                               */
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
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                             */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.       */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY    */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/variant/callable.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"

#include "modules/limboai/planning/src/domain.h"

namespace PlanningTestDomains {

// Shared command: increment state["value"]["value"] by p_amount
static Dictionary action_increment(Dictionary p_state, int p_amount) {
	Dictionary new_state = p_state.duplicate(true);
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
	Dictionary value_dict = new_state["value"];
	value_dict["value"] = current_value + p_amount;
	new_state["value"] = value_dict;
	return new_state;
}

// Unigoal method: (state, subject, value) -> Array of planner elements (one command)
static Array unigoal_value_method(Dictionary p_state, String p_subject, Variant p_value) {
	Array result;
	Array action;
	action.push_back("action_increment");
	action.push_back(1);
	result.push_back(action);
	return result;
}

// Task method (succeeds)
static Variant task_increment_succeed(Dictionary p_state) {
	Array result;
	Array action;
	action.push_back("action_increment");
	action.push_back(1);
	result.push_back(action);
	return result;
}

// Task method (fails) for backtracking
static Variant task_increment_fail(Dictionary p_state) {
	return Variant();
}

class PlanningTestDomainsCallable {
public:
	static Dictionary action_increment(Dictionary p_state, int p_amount) {
		return PlanningTestDomains::action_increment(p_state, p_amount);
	}
	static Array unigoal_value_method(Dictionary p_state, String p_subject, Variant p_value) {
		return PlanningTestDomains::unigoal_value_method(p_state, p_subject, p_value);
	}
	static Variant task_increment_succeed(Dictionary p_state) {
		return PlanningTestDomains::task_increment_succeed(p_state);
	}
	static Variant task_increment_fail(Dictionary p_state) {
		return PlanningTestDomains::task_increment_fail(p_state);
	}
};

// Minimal goal domain: one predicate "value", one command, one unigoal method
static Ref<PlannerDomain> create_minimal_goal_domain() {
	Ref<PlannerDomain> domain = memnew(PlannerDomain);
	domain->add_command("action_increment", callable_mp_static(&PlanningTestDomainsCallable::action_increment));
	TypedArray<Callable> unigoal_methods;
	unigoal_methods.push_back(callable_mp_static(&PlanningTestDomainsCallable::unigoal_value_method));
	domain->add_unigoal_methods("value", unigoal_methods);
	return domain;
}

// Minimal HTN domain: one compound task "increment", one method returning one command
static Ref<PlannerDomain> create_minimal_htn_domain() {
	Ref<PlannerDomain> domain = memnew(PlannerDomain);
	domain->add_command("action_increment", callable_mp_static(&PlanningTestDomainsCallable::action_increment));
	TypedArray<Callable> task_methods;
	task_methods.push_back(callable_mp_static(&PlanningTestDomainsCallable::task_increment_succeed));
	domain->add_task_methods("increment", task_methods);
	return domain;
}

// Minimal backtracking domain: one task "increment", two methods (first fails, second succeeds)
static Ref<PlannerDomain> create_minimal_backtracking_domain() {
	Ref<PlannerDomain> domain = memnew(PlannerDomain);
	domain->add_command("action_increment", callable_mp_static(&PlanningTestDomainsCallable::action_increment));
	TypedArray<Callable> task_methods;
	task_methods.push_back(callable_mp_static(&PlanningTestDomainsCallable::task_increment_fail));
	task_methods.push_back(callable_mp_static(&PlanningTestDomainsCallable::task_increment_succeed));
	domain->add_task_methods("increment", task_methods);
	return domain;
}

} // namespace PlanningTestDomains
