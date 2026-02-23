/**************************************************************************/
/*  bt_command.cpp                                                        */
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

#include "bt_command.h"

#include "../../../planning/src/domain.h"
#include "../../../planning/src/plan.h"
#include "../../../planning/src/planner_state.h"

void BTCommand::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_planner_plan", "plan"), &BTCommand::set_planner_plan);
	ClassDB::bind_method(D_METHOD("get_planner_plan"), &BTCommand::get_planner_plan);
	ClassDB::bind_method(D_METHOD("set_planner_state", "state"), &BTCommand::set_planner_state);
	ClassDB::bind_method(D_METHOD("get_planner_state"), &BTCommand::get_planner_state);
	ClassDB::bind_method(D_METHOD("set_command_name", "name"), &BTCommand::set_command_name);
	ClassDB::bind_method(D_METHOD("get_command_name"), &BTCommand::get_command_name);
	ClassDB::bind_method(D_METHOD("set_command_args", "args"), &BTCommand::set_command_args);
	ClassDB::bind_method(D_METHOD("get_command_args"), &BTCommand::get_command_args);
	ClassDB::bind_method(D_METHOD("set_args_var", "var"), &BTCommand::set_args_var);
	ClassDB::bind_method(D_METHOD("get_args_var"), &BTCommand::get_args_var);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "planner_plan", PROPERTY_HINT_RESOURCE_TYPE, "PlannerPlan"), "set_planner_plan", "get_planner_plan");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "planner_state", PROPERTY_HINT_RESOURCE_TYPE, "PlannerState"), "set_planner_state", "get_planner_state");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "command_name"), "set_command_name", "get_command_name");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "command_args"), "set_command_args", "get_command_args");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "args_var"), "set_args_var", "get_args_var");
}

void BTCommand::set_planner_plan(const Ref<PlannerPlan> &p_plan) {
	planner_plan = p_plan;
	emit_changed();
}

void BTCommand::set_planner_state(const Ref<PlannerState> &p_state) {
	planner_state = p_state;
	emit_changed();
}

BTCommand::BTCommand() {
	args_var = StringName("command_args");
}

void BTCommand::set_command_name(const String &p_name) {
	command_name = p_name;
	emit_changed();
}

void BTCommand::set_command_args(const Array &p_args) {
	command_args = p_args;
	emit_changed();
}

void BTCommand::set_args_var(const StringName &p_var) {
	args_var = p_var;
	emit_changed();
}

PackedStringArray BTCommand::get_configuration_warnings() {
	PackedStringArray w = BTAction::get_configuration_warnings();
	if (command_name.is_empty()) {
		w.append("Command name is not set.");
	}
	return w;
}

Dictionary BTCommand::get_debug_plan_detail() const {
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

String BTCommand::_generate_name() {
	return command_name.is_empty() ? "Command" : String("Command: ") + command_name;
}

BTTask::Status BTCommand::_tick(double p_delta) {
	(void)p_delta;
	Ref<Blackboard> bb = get_blackboard();
	if (bb.is_null()) {
		return FAILURE;
	}
	Ref<PlannerPlan> plan = planner_plan;
	if (plan.is_null() || !plan->get_current_domain().is_valid()) {
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
	Ref<PlannerDomain> domain = plan->get_current_domain();
	if (!domain->command_dictionary.has(command_name)) {
		return FAILURE;
	}
	Variant args_v = bb->has_var(args_var) ? bb->get_var(args_var, Array(), false) : Variant(command_args);
	Array args = args_v.get_type() == Variant::ARRAY ? (Array)args_v : command_args;
	Array call_args;
	call_args.push_back(state_dict);
	call_args.append_array(args);
	Callable cmd = domain->command_dictionary[command_name];
	Variant result = cmd.callv(call_args);
	if (result.get_type() != Variant::DICTIONARY) {
		return FAILURE;
	}
	state->apply_plan_state(result);
	return SUCCESS;
}
