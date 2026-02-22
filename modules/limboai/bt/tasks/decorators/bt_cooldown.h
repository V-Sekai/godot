/**************************************************************************/
/*  bt_cooldown.h                                                         */
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
 * bt_cooldown.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_COOLDOWN_H
#define BT_COOLDOWN_H

#include "../bt_decorator.h"

#ifdef LIMBOAI_MODULE
#include "scene/main/scene_tree.h"
#endif

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/scene_tree_timer.hpp>
#endif

class BTCooldown : public BTDecorator {
	GDCLASS(BTCooldown, BTDecorator);
	TASK_CATEGORY(Decorators);

private:
	double duration = 10.0;
	bool process_pause = false;
	bool start_cooled = false;
	bool trigger_on_failure = false;
	StringName cooldown_state_var = "";

	Ref<SceneTreeTimer> timer = nullptr;

	void _chill();
	void _on_timeout();

protected:
	static void _bind_methods();

	virtual String _generate_name() override;
	virtual void _setup() override;
	virtual Status _tick(double p_delta) override;

public:
	void set_duration(double p_value);
	double get_duration() const { return duration; }

	void set_process_pause(bool p_value);
	bool get_process_pause() const { return process_pause; }

	void set_start_cooled(bool p_value);
	bool get_start_cooled() const { return start_cooled; }

	void set_trigger_on_failure(bool p_value);
	bool get_trigger_on_failure() const { return trigger_on_failure; }

	void set_cooldown_state_var(const StringName &p_value);
	StringName get_cooldown_state_var() const { return cooldown_state_var; }
};

#endif // BT_COOLDOWN_H
