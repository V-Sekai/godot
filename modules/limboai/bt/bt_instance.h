/**************************************************************************/
/*  bt_instance.h                                                         */
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
 * bt_instance.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
#ifndef BT_INSTANCE_H
#define BT_INSTANCE_H

#include "tasks/bt_task.h"

class BTInstance : public RefCounted {
	GDCLASS(BTInstance, RefCounted);

private:
	Ref<BTTask> root_task;
	uint64_t owner_node_id = 0;
	String source_bt_path;
	BT::Status last_status = BT::FRESH;

#ifdef DEBUG_ENABLED
	bool monitor_performance = false;
	StringName monitor_id;
	double update_time_acc = 0.0;
	double update_time_n = 0.0;

	double _get_mean_update_time_msec_and_reset();
	void _add_custom_monitor();
	void _remove_custom_monitor();

#endif // * DEBUG_ENABLED

protected:
	static void _bind_methods();

#ifdef LIMBOAI_GDEXTENSION
	String _to_string() const { return "<" + get_class() + "#" + itos(get_instance_id()) + ">"; }
#endif

public:
	_FORCE_INLINE_ Ref<BTTask> get_root_task() const { return root_task; }
	Node *get_owner_node() const;
	_FORCE_INLINE_ BT::Status get_last_status() const { return last_status; }
	_FORCE_INLINE_ String get_source_bt_path() const { return source_bt_path; }
	_FORCE_INLINE_ Node *get_agent() const { return root_task.is_valid() ? root_task->get_agent() : nullptr; }
	_FORCE_INLINE_ Ref<Blackboard> get_blackboard() const { return root_task.is_valid() ? root_task->get_blackboard() : Ref<Blackboard>(); }

	_FORCE_INLINE_ bool is_instance_valid() const { return root_task.is_valid(); }

	BT::Status update(double p_delta);

	void set_monitor_performance(bool p_monitor);
	bool get_monitor_performance() const;

	void register_with_debugger();
	void unregister_with_debugger();

	static Ref<BTInstance> create(Ref<BTTask> p_root_task, String p_source_bt_path, Node *p_owner_node);

	BTInstance() = default;
	~BTInstance();
};

#endif // BT_INSTANCE_H
