/**************************************************************************/
/*  bt_execute_plan.cpp                                                   */
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

/**
 * bt_execute_plan.cpp
 */

#include "bt_execute_plan.h"

#include "../../../planning/src/domain.h"
#include "../../../planning/src/plan.h"
#include "../../../planning/src/planner_state.h"

void BTExecutePlan::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_planner_plan", "plan"), &BTExecutePlan::set_planner_plan);
	ClassDB::bind_method(D_METHOD("get_planner_plan"), &BTExecutePlan::get_planner_plan);
	ClassDB::bind_method(D_METHOD("set_planner_state", "state"), &BTExecutePlan::set_planner_state);
	ClassDB::bind_method(D_METHOD("get_planner_state"), &BTExecutePlan::get_planner_state);
	ClassDB::bind_method(D_METHOD("set_plan_var", "var"), &BTExecutePlan::set_plan_var);
	ClassDB::bind_method(D_METHOD("get_plan_var"), &BTExecutePlan::get_plan_var);
	ClassDB::bind_method(D_METHOD("set_plan_index_var", "var"), &BTExecutePlan::set_plan_index_var);
	ClassDB::bind_method(D_METHOD("get_plan_index_var"), &BTExecutePlan::get_plan_index_var);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "planner_plan", PROPERTY_HINT_RESOURCE_TYPE, "PlannerPlan"), "set_planner_plan", "get_planner_plan");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "planner_state", PROPERTY_HINT_RESOURCE_TYPE, "PlannerState"), "set_planner_state", "get_planner_state");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "plan_var"), "set_plan_var", "get_plan_var");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "plan_index_var"), "set_plan_index_var", "get_plan_index_var");
}

BTExecutePlan::BTExecutePlan() {
	plan_var = StringName("plan");
	plan_index_var = StringName("plan_index");
}

void BTExecutePlan::set_planner_plan(const Ref<PlannerPlan> &p_plan) {
	planner_plan = p_plan;
	emit_changed();
}

void BTExecutePlan::set_planner_state(const Ref<PlannerState> &p_state) {
	planner_state = p_state;
	emit_changed();
}

void BTExecutePlan::set_plan_var(const StringName &p_var) {
	plan_var = p_var;
	emit_changed();
}

void BTExecutePlan::set_plan_index_var(const StringName &p_var) {
	plan_index_var = p_var;
	emit_changed();
}

PackedStringArray BTExecutePlan::get_configuration_warnings() {
	PackedStringArray w = BTAction::get_configuration_warnings();
	if (planner_plan.is_null()) {
		w.append("PlannerPlan is not set.");
	}
	return w;
}

String BTExecutePlan::_generate_name() {
	return "Execute Plan";
}

BTTask::Status BTExecutePlan::_tick(double p_delta) {
	(void)p_delta;
	Ref<Blackboard> bb = get_blackboard();
	if (bb.is_null()) {
		return FAILURE;
	}
	if (planner_plan.is_null() || !planner_plan->get_current_domain().is_valid()) {
		return FAILURE;
	}

	Array plan = bb->get_var(plan_var, Array(), false);
	int idx = bb->get_var(plan_index_var, 0);
	if (idx < 0 || idx >= plan.size()) {
		return FAILURE;
	}

	Ref<PlannerState> state = planner_state;
	if (state.is_null()) {
		state.instantiate();
		state->set_blackboard(bb);
	} else if (state->get_blackboard() != bb) {
		state->set_blackboard(bb);
	}

	Dictionary state_dict = state->to_plan_dictionary();
	Ref<PlannerDomain> domain = planner_plan->get_current_domain();
	Dictionary command_dict = domain->command_dictionary;

	Variant action_v = plan[idx];
	if (action_v.get_type() == Variant::DICTIONARY) {
		Dictionary d = action_v;
		if (d.has("item")) {
			action_v = d["item"];
		}
	}
	if (action_v.get_type() != Variant::ARRAY) {
		bb->set_var(plan_index_var, idx + 1);
		return idx + 1 >= plan.size() ? SUCCESS : RUNNING;
	}
	Array action = action_v;
	if (action.is_empty()) {
		bb->set_var(plan_index_var, idx + 1);
		return idx + 1 >= plan.size() ? SUCCESS : RUNNING;
	}

	String command_name = action[0];
	if (!command_dict.has(command_name)) {
		return FAILURE;
	}
	Callable cmd = command_dict[command_name];
	Array args;
	args.push_back(state_dict);
	for (int i = 1; i < action.size(); i++) {
		args.push_back(action[i]);
	}
	Variant result = cmd.callv(args);
	if (result.get_type() != Variant::DICTIONARY) {
		return FAILURE;
	}
	state->apply_plan_state(result);
	bb->set_var(plan_index_var, idx + 1);
	return (idx + 1 >= plan.size()) ? SUCCESS : RUNNING;
}

Dictionary BTExecutePlan::get_debug_plan_detail() const {
	Dictionary d;
	Ref<Blackboard> bb = get_blackboard();
	if (bb.is_null()) {
		return d;
	}
	Variant plan = bb->get_var(plan_var, Array(), false);
	Variant plan_index = bb->get_var(plan_index_var, 0, false);
	Variant solution_graph = bb->get_var(StringName("solution_graph"), Dictionary(), false);
	if (plan.get_type() != Variant::NIL) {
		d["plan"] = plan;
	}
	if (plan_index.get_type() != Variant::NIL) {
		d["plan_index"] = plan_index;
	}
	if (solution_graph.get_type() != Variant::NIL) {
		d["solution_graph"] = solution_graph;
	}
	return d;
}
