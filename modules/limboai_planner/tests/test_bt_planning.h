/**************************************************************************/
/*  test_bt_planning.h                                                    */
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

/**
 * test_bt_planning.h
 * Unit tests for BT planning nodes and get_debug_plan_detail / BehaviorTreeData debug_plan_detail.
 *
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/bt_action.h"

#ifdef TOOLS_ENABLED
#include "modules/limboai/editor/debugger/behavior_tree_data.h"
#endif

namespace TestBTPlanning {

TEST_CASE("[Modules][LimboAI][Planner] get_debug_plan_detail") {
	SUBCASE("BTAction returns empty when not initialized") {
		Ref<BTAction> action = memnew(BTAction);
		Dictionary d = action->get_debug_plan_detail();
		CHECK(d.is_empty());
	}
}

#ifdef TOOLS_ENABLED
TEST_CASE("[Modules][LimboAI][Planner] BehaviorTreeData serialize/deserialize with debug_plan_detail") {
	// Build array in serialize format: [bt_instance_id, node_owner_path, source_bt_path, per-task: id, name, is_custom, num_children, status, elapsed, type_name, script_path, debug_plan_detail]
	Array arr;
	arr.push_back(12345); // INT for deserialize
	arr.push_back(NodePath());
	arr.push_back(String("res://test.tres"));
	arr.push_back(1); // task id
	arr.push_back(String("Run Planner"));
	arr.push_back(false);
	arr.push_back(0);
	arr.push_back(int(BTTask::SUCCESS));
	arr.push_back(0.5);
	arr.push_back(String("BTRunPlanner"));
	arr.push_back(String());
	Dictionary debug_detail;
	debug_detail["plan"] = Array();
	debug_detail["plan_index"] = 0;
	debug_detail["solution_graph"] = Dictionary();
	arr.push_back(debug_detail);

	Ref<BehaviorTreeData> data = BehaviorTreeData::deserialize(arr);
	REQUIRE(data.is_valid());
	CHECK_EQ(data->tasks.size(), 1);
	const BehaviorTreeData::TaskData &task0 = data->tasks.get(0);
	CHECK(task0.debug_plan_detail.has("plan"));
	CHECK(task0.debug_plan_detail.has("plan_index"));
	CHECK(task0.debug_plan_detail.has("solution_graph"));
	CHECK_EQ(int(task0.debug_plan_detail["plan_index"]), 0);
}
#endif

} // namespace TestBTPlanning
