/**************************************************************************/
/*  domain.h                                                              */
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

// SPDX-FileCopyrightText: 2021 University of Maryland
// SPDX-License-Identifier: BSD-3-Clause-Clear
// Author: Dana Nau <nau@umd.edu>, July 7, 2021

#include "multigoal.h"

#include "core/io/resource.h"
#include "core/object/object.h"
#include "core/variant/typed_array.h"
#include "planner_time_range.h"

class PlannerTaskMetadata : public Resource {
	GDCLASS(PlannerTaskMetadata, Resource);

private:
	String task_id;
	PlannerTimeRange time_range;

public:
	PlannerTaskMetadata();
	void set_task_id(String p_id) { task_id = p_id; }
	String get_task_id() const { return task_id; }
	void set_time_range(PlannerTimeRange p_time_range) { time_range = p_time_range; }
	PlannerTimeRange get_time_range() const { return time_range; }
	// p_physical_time is in absolute microseconds since Unix epoch
	void update_metadata(int64_t p_physical_time);

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_task_id", "id"), &PlannerTaskMetadata::set_task_id);
		ClassDB::bind_method(D_METHOD("get_task_id"), &PlannerTaskMetadata::get_task_id);
		ClassDB::bind_method(D_METHOD("update_metadata", "physical_time"), &PlannerTaskMetadata::update_metadata);
	}
};

class PlannerTask : public Resource {
	GDCLASS(PlannerTask, Resource);

private:
	Ref<PlannerTaskMetadata> metadata;

public:
	PlannerTask();
	Ref<PlannerTaskMetadata> get_metadata() const { return metadata; }
	void set_metadata(Ref<PlannerTaskMetadata> p_metadata) { metadata = p_metadata; }

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("get_metadata"), &PlannerTask::get_metadata);
		ClassDB::bind_method(D_METHOD("set_metadata", "metadata"), &PlannerTask::set_metadata);
	}
};

class PlannerPlan;
class PlannerDomain : public Resource {
	GDCLASS(PlannerDomain, Resource);
	friend PlannerPlan;

public:
	Dictionary action_dictionary; // Public for testing
private:
	Dictionary task_method_dictionary;
	Dictionary unigoal_method_dictionary;
	TypedArray<Callable> multigoal_method_list;

public:
	PlannerDomain();

public:
	void add_actions(TypedArray<Callable> p_actions);
	void add_task_methods(String p_task_name, TypedArray<Callable> p_methods);
	void add_unigoal_methods(String p_task_name, TypedArray<Callable> p_methods);
	void add_multigoal_methods(TypedArray<Callable> p_methods);

public:
	static Variant method_verify_goal(Dictionary p_state, String p_method, String p_state_var, String p_arguments, Variant p_desired_values, int p_depth, int verbose);

protected:
	static void _bind_methods();
};
