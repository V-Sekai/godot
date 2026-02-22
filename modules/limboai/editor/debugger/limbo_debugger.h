/**************************************************************************/
/*  limbo_debugger.h                                                      */
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
 * limbo_debugger.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef LIMBO_DEBUGGER_H
#define LIMBO_DEBUGGER_H

#ifdef LIMBOAI_MODULE
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/templates/hash_set.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/templates/hash_set.hpp>
using namespace godot;
#endif // LIMBOAI_GDEXTENSION

class LimboDebugger : public Object {
	GDCLASS(LimboDebugger, Object);

private:
	static LimboDebugger *singleton;

	LimboDebugger();

public:
	static void initialize();
	static void deinitialize();
	_FORCE_INLINE_ static LimboDebugger *get_singleton() { return singleton; }

	~LimboDebugger();

protected:
	static void _bind_methods();

#ifdef DEBUG_ENABLED
private:
	HashSet<uint64_t> active_bt_instances;
	uint64_t tracked_instance_id = 0;
	bool session_active = false;

	void _track_tree(uint64_t p_instance_id);
	void _untrack_tree();
	void _send_active_bt_players();

	void _on_bt_instance_updated(int status, uint64_t p_instance_id);

public:
	static Error parse_message(void *p_user, const String &p_msg, const Array &p_args, bool &r_captured);
#ifdef LIMBOAI_GDEXTENSION
	bool parse_message_gdext(const String &p_msg, const Array &p_args);
#endif

	void register_bt_instance(uint64_t p_instance_id);
	void unregister_bt_instance(uint64_t p_instance_id);
	bool is_active() const;

#endif // ! DEBUG_ENABLED
};
#endif // LIMBO_DEBUGGER_H
