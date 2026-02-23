/**************************************************************************/
/*  bt_state.h                                                            */
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
 * bt_state.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_STATE_H
#define BT_STATE_H

#include "../hsm/limbo_state.h"

#include "../bt/behavior_tree.h"
#include "../bt/bt_instance.h"

class BTState : public LimboState {
	GDCLASS(BTState, LimboState);

private:
	Ref<BehaviorTree> behavior_tree;
	Ref<BTInstance> bt_instance;
	StringName success_event;
	StringName failure_event;
	Node *scene_root_hint = nullptr;
	bool monitor_performance = false;

	_FORCE_INLINE_ Node *_get_scene_root() const { return scene_root_hint ? scene_root_hint : get_owner(); }

protected:
	static void _bind_methods();

	void _notification(int p_notification);

	virtual bool _should_use_new_scope() const override { return true; }
	virtual void _update_blackboard_plan() override;
	virtual Node *_get_prefetch_root_for_base_plan() override;

	virtual void _setup() override;
	virtual void _exit() override;
	virtual void _update(double p_delta) override;

public:
	void set_behavior_tree(const Ref<BehaviorTree> &p_value);
	Ref<BehaviorTree> get_behavior_tree() const { return behavior_tree; }

	Ref<BTInstance> get_bt_instance() const { return bt_instance; }

	void set_success_event(const StringName &p_success_event) { success_event = p_success_event; }
	StringName get_success_event() const { return success_event; }

	void set_failure_event(const StringName &p_failure_event) { failure_event = p_failure_event; }
	StringName get_failure_event() const { return failure_event; }

	void set_monitor_performance(bool p_monitor);
	bool get_monitor_performance() const { return monitor_performance; }

	void set_scene_root_hint(Node *p_node);

	BTState();
};

#endif // BT_STATE_H
