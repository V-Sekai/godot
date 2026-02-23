/**************************************************************************/
/*  bt_command.h                                                          */
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
#include "../bt_action.h"
class PlannerPlan;
class PlannerState;
class BTCommand : public BTAction {
	GDCLASS(BTCommand, BTAction);
	TASK_CATEGORY(Planning);

private:
	Ref<PlannerPlan> planner_plan;
	Ref<PlannerState> planner_state;
	String command_name;
	Array command_args;
	StringName args_var;

protected:
	static void _bind_methods();
	virtual String _generate_name() override;
	virtual Status _tick(double p_delta) override;
	virtual Dictionary get_debug_plan_detail() const override;

public:
	void set_planner_plan(const Ref<PlannerPlan> &p_plan);
	Ref<PlannerPlan> get_planner_plan() const { return planner_plan; }
	void set_planner_state(const Ref<PlannerState> &p_state);
	Ref<PlannerState> get_planner_state() const { return planner_state; }
	void set_command_name(const String &p_name);
	String get_command_name() const { return command_name; }
	void set_command_args(const Array &p_args);
	Array get_command_args() const { return command_args; }
	void set_args_var(const StringName &p_var);
	StringName get_args_var() const { return args_var; }
	virtual PackedStringArray get_configuration_warnings() override;
	BTCommand();
};
