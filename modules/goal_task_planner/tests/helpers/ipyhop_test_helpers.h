/**************************************************************************/
/*  ipyhop_test_helpers.h                                                 */
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
#include "../../src/planner_result.h"
#include "../domains/ipyhop_test_domain.h"
#include "core/variant/array.h"
#include "core/variant/variant.h"

// Helper to create initial state for IPyHOP tests
Dictionary create_init_state_1() {
	Dictionary state;
	Dictionary flag;
	flag[0] = true;
	for (int i = 1; i < 20; i++) {
		flag[i] = false;
	}
	state["flag"] = flag;
	return state;
}

// Helper to create domain for IPyHOP tests
Ref<PlannerDomain> create_ipyhop_test_domain() {
	Ref<PlannerDomain> domain = memnew(PlannerDomain);

	// Register actions
	// Note: In tools-disabled builds, use add_command with explicit names
	// For now, actions are registered inline (requires IPyHOPTestDomainCallable to be defined)
	// TypedArray<Callable> actions;
	// actions.push_back(callable_mp_static(&IPyHOPTestDomainCallable::action_transfer_flag));
	// actions.push_back(callable_mp_static(&IPyHOPTestDomainCallable::action_putv));
	// actions.push_back(callable_mp_static(&IPyHOPTestDomainCallable::action_getv));
	// For explicit naming: domain->add_command("name", callable);

	return domain;
}

// Helper to validate plan result
bool validate_plan_result(Ref<PlannerResult> result, const Array &expected_actions) {
	if (!result.is_valid() || !result->get_success()) {
		return false;
	}
	Array plan = result->extract_plan();
	if (plan.size() != expected_actions.size()) {
		return false;
	}
	for (int i = 0; i < plan.size(); i++) {
		Array plan_action = plan[i];
		Array expected_action = expected_actions[i];
		if (plan_action.size() != expected_action.size()) {
			return false;
		}
		for (int j = 0; j < plan_action.size(); j++) {
			if (plan_action[j] != expected_action[j]) {
				return false;
			}
		}
	}
	return true;
}
