/**************************************************************************/
/*  behavior_tree.h                                                       */
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
 * behavior_tree.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BEHAVIOR_TREE_H
#define BEHAVIOR_TREE_H

#include "../blackboard/blackboard_plan.h"
#include "bt_instance.h"
#include "tasks/bt_task.h"

#ifdef LIMBOAI_MODULE
#include "core/io/resource.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/resource.hpp>
using namespace godot;
#endif // LIMBOAI_GDEXTENSION

class BehaviorTree : public Resource {
	GDCLASS(BehaviorTree, Resource);

private:
	String description;
	Ref<BlackboardPlan> blackboard_plan;
	Ref<BTTask> root_task;

	void _plan_changed();

#ifdef TOOLS_ENABLED
	void _set_editor_behavior_tree_hint();
	void _unset_editor_behavior_tree_hint();
#endif // TOOLS_ENABLED

protected:
	static void _bind_methods();

#ifdef LIMBOAI_GDEXTENSION
	String _to_string() const { return "<" + get_class() + "#" + itos(get_instance_id()) + ">"; }
#endif

public:
#ifdef LIMBOAI_MODULE
	virtual bool editor_can_reload_from_file() override { return false; }
#endif

	void set_description(const String &p_value);
	String get_description() const { return description; }

	void set_blackboard_plan(const Ref<BlackboardPlan> &p_plan);
	Ref<BlackboardPlan> get_blackboard_plan() const { return blackboard_plan; }

	void set_root_task(const Ref<BTTask> &p_value);
	Ref<BTTask> get_root_task() const { return root_task; }

	Ref<BehaviorTree> clone() const;
	void copy_other(const Ref<BehaviorTree> &p_other);
	Ref<BTInstance> instantiate(Node *p_agent, const Ref<Blackboard> &p_blackboard, Node *p_instance_owner, Node *p_custom_scene_root = nullptr) const;

	void emit_branch_changed(const Ref<BTTask> &p_branch);

	BehaviorTree();
	~BehaviorTree();
};

#endif // BEHAVIOR_TREE_H
