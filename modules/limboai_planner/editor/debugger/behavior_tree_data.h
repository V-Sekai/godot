/**************************************************************************/
/*  behavior_tree_data.h                                                  */
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
 * behavior_tree_data.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BEHAVIOR_TREE_DATA_H
#define BEHAVIOR_TREE_DATA_H

#include "../../bt/bt_instance.h"

#ifdef LIMBOAI_MODULE
#include "core/variant/dictionary.h"
#endif

class BehaviorTreeData : public RefCounted {
	GDCLASS(BehaviorTreeData, RefCounted);

protected:
	static void _bind_methods();

public:
	struct TaskData {
		uint64_t id = 0;
		String name;
		bool is_custom_name = false;
		int num_children = 0;
		int status = 0;
		double elapsed_time = 0.0;
		String type_name;
		String script_path;
		Dictionary debug_plan_detail;

		TaskData(uint64_t p_id, const String &p_name, bool p_is_custom_name, int p_num_children, int p_status, double p_elapsed_time, const String &p_type_name, const String &p_script_path, const Dictionary &p_debug_plan_detail = Dictionary()) {
			id = p_id;
			name = p_name;
			is_custom_name = p_is_custom_name;
			num_children = p_num_children;
			status = p_status;
			elapsed_time = p_elapsed_time;
			type_name = p_type_name;
			script_path = p_script_path;
			debug_plan_detail = p_debug_plan_detail;
		}

		TaskData() {}
	};

	List<TaskData> tasks;
	uint64_t bt_instance_id = 0;
	NodePath node_owner_path;
	String source_bt_path;

public:
	static Array serialize(const Ref<BTInstance> &p_instance);
	static Ref<BehaviorTreeData> deserialize(const Array &p_array);
	static Ref<BehaviorTreeData> create_from_bt_instance(const Ref<BTInstance> &p_bt_instance);

	BehaviorTreeData();
};

#endif // BEHAVIOR_TREE_DATA_H
