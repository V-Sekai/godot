/**************************************************************************/
/*  bt_compound_task.cpp                                                  */
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

#include "bt_compound_task.h"

#include "../../../planning/src/domain.h"
#include "../../../planning/src/plan.h"
#include "../../../planning/src/planner_result.h"
#include "../../../planning/src/planner_state.h"

void BTCompoundTask::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_planner_plan", "plan"), &BTCompoundTask::set_planner_plan);
	ClassDB::bind_method(D_METHOD("get_planner_plan"), &BTCompoundTask::get_planner_plan);
	ClassDB::bind_method(D_METHOD("set_planner_state", "state"), &BTCompoundTask::set_planner_state);
	ClassDB::bind_method(D_METHOD("get_planner_state"), &BTCompoundTask::get_planner_state);
	ClassDB::bind_method(D_METHOD("set_compound_task_name", "name"), &BTCompoundTask::set_compound_task_name);
	ClassDB::bind_method(D_METHOD("get_compound_task_name"), &BTCompoundTask::get_compound_task_name);
	ClassDB::bind_method(D_METHOD("set_task_args", "args"), &BTCompoundTask::set_task_args);
	ClassDB::bind_method(D_METHOD("get_task_args"), &BTCompoundTask::get_task_args);
	ClassDB::bind_method(D_METHOD("set_todo_list_var", "var"), &BTCompoundTask::set_todo_list_var);
	ClassDB::bind_method(D_METHOD("get_todo_list_var"), &BTCompoundTask::get_todo_list_var);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "planner_plan", PROPERTY_HINT_RESOURCE_TYPE, "PlannerPlan"), "set_planner_plan", "get_planner_plan");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "planner_state", PROPERTY_HINT_RESOURCE_TYPE, "PlannerState"), "set_planner_state", "get_planner_state");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "compound_task_name"), "set_compound_task_name", "get_compound_task_name");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "task_args"), "set_task_args", "get_task_args");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "todo_list_var"), "set_todo_list_var", "get_todo_list_var");
}

BTCompoundTask::BTCompoundTask() {
	todo_list_var = StringName("todo_list");
}

void BTCompoundTask::set_planner_plan(const Ref<PlannerPlan> &p_plan) {
	planner_plan = p_plan;
	emit_changed();
}
void BTCompoundTask::set_planner_state(const Ref<PlannerState> &p_state) {
	planner_state = p_state;
	emit_changed();
}
void BTCompoundTask::set_compound_task_name(const String &p_name) {
	compound_task_name = p_name;
	emit_changed();
}
void BTCompoundTask::set_task_args(const Array &p_args) {
	task_args = p_args;
	emit_changed();
}
void BTCompoundTask::set_todo_list_var(const StringName &p_var) {
	todo_list_var = p_var;
	emit_changed();
}

PackedStringArray BTCompoundTask::get_configuration_warnings() {
	PackedStringArray w = BTAction::get_configuration_warnings();
	if (planner_plan.is_null())
		w.append("PlannerPlan is not set.");
	if (compound_task_name.is_empty())
		w.append("Compound task name is not set.");
	return w;
}

String BTCompoundTask::_generate_name() {
	return compound_task_name.is_empty() ? "Compound Task" : String("Task: ") + compound_task_name;
}

BTTask::Status BTCompoundTask::_tick(double p_delta) {
	(void)p_delta;
	Ref<Blackboard> bb = get_blackboard();
	if (bb.is_null())
		return FAILURE;
	if (planner_plan.is_null() || !planner_plan->get_current_domain().is_valid())
		return FAILURE;

	Ref<PlannerState> state = planner_state;
	if (state.is_null()) {
		state.instantiate();
		state->set_blackboard(bb);
	} else if (state->get_blackboard() != bb) {
		state->set_blackboard(bb);
	}
	Array todo_list;
	todo_list.push_back(compound_task_name);
	todo_list.append_array(task_args);
	Dictionary state_dict = state->to_plan_dictionary();
	Ref<PlannerResult> result = planner_plan->find_plan(state_dict, todo_list);
	if (result.is_null() || !result->get_success())
		return FAILURE;

	Array plan = result->extract_plan(0);
	bb->set_var(StringName("plan"), plan);
	bb->set_var(StringName("plan_index"), 0);
	bb->set_var(StringName("solution_graph"), result->get_solution_graph());
	state->apply_plan_state(result->get_final_state());
	return SUCCESS;
}

Dictionary BTCompoundTask::get_debug_plan_detail() const {
	Dictionary d;
	Ref<Blackboard> bb = get_blackboard();
	if (bb.is_null()) {
		return d;
	}
	Variant plan = bb->get_var(StringName("plan"), Array(), false);
	Variant plan_index = bb->get_var(StringName("plan_index"), 0, false);
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
