/**************************************************************************/
/*  temporal_entity_test_domain.h                                         */
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

#include "core/variant/array.h"
#include "core/variant/callable.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"

// Test domain for temporal and entity constraints
class TemporalEntityTestDomainCallable {
public:
	// Action with temporal constraint (duration)
	static Dictionary action_work_task(Dictionary p_state, String p_worker, String p_task);
	// Action that requires entity capability
	static Dictionary action_use_tool(Dictionary p_state, String p_worker, String p_tool);
	// Task method that requires entity capability
	static Variant task_complete_work(Dictionary p_state, String p_worker, String p_task);
};

namespace TemporalEntityTestDomain {

// Action: work_task(worker, task) - worker completes a task (takes time)
Dictionary action_work_task(Dictionary state, String worker, String task) {
	Dictionary new_state = state.duplicate();
	Dictionary worker_state;
	if (new_state.has(worker)) {
		worker_state = new_state[worker];
	} else {
		worker_state = Dictionary();
	}
	Dictionary completed;
	if (worker_state.has("completed_tasks")) {
		completed = worker_state["completed_tasks"];
	} else {
		completed = Dictionary();
	}
	completed[task] = true;
	worker_state["completed_tasks"] = completed;
	new_state[worker] = worker_state;
	return new_state;
}

// Action: use_tool(worker, tool) - worker uses a tool (requires capability)
Dictionary action_use_tool(Dictionary state, String worker, String tool) {
	Dictionary new_state = state.duplicate();
	Dictionary worker_state;
	if (new_state.has(worker)) {
		worker_state = new_state[worker];
	} else {
		worker_state = Dictionary();
	}
	Dictionary tools_used;
	if (worker_state.has("tools_used")) {
		tools_used = worker_state["tools_used"];
	} else {
		tools_used = Dictionary();
	}
	tools_used[tool] = true;
	worker_state["tools_used"] = tools_used;
	new_state[worker] = worker_state;
	return new_state;
}

// Task method: complete_work(worker, task) - worker completes work using a tool
Variant task_complete_work(Dictionary state, String worker, String task) {
	Array subtasks;
	// First use a tool, then complete the work
	Array use_tool_task;
	use_tool_task.push_back("action_use_tool");
	use_tool_task.push_back(worker);
	use_tool_task.push_back("hammer");
	Array work_task;
	work_task.push_back("action_work_task");
	work_task.push_back(worker);
	work_task.push_back(task);
	subtasks.push_back(use_tool_task);
	subtasks.push_back(work_task);
	return subtasks;
}

} // namespace TemporalEntityTestDomain

// Implementations of TemporalEntityTestDomainCallable static methods
inline Dictionary TemporalEntityTestDomainCallable::action_work_task(Dictionary p_state, String p_worker, String p_task) {
	return TemporalEntityTestDomain::action_work_task(p_state, p_worker, p_task);
}

inline Dictionary TemporalEntityTestDomainCallable::action_use_tool(Dictionary p_state, String p_worker, String p_tool) {
	return TemporalEntityTestDomain::action_use_tool(p_state, p_worker, p_tool);
}

inline Variant TemporalEntityTestDomainCallable::task_complete_work(Dictionary p_state, String p_worker, String p_task) {
	return TemporalEntityTestDomain::task_complete_work(p_state, p_worker, p_task);
}
