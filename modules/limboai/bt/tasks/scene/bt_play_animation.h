/**************************************************************************/
/*  bt_play_animation.h                                                   */
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
 * bt_play_animation.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_PLAY_ANIMATION_H
#define BT_PLAY_ANIMATION_H

#include "../bt_action.h"

#include "../../../blackboard/bb_param/bb_node.h"

#ifdef LIMBOAI_MODULE
#include "scene/animation/animation_player.h"
#endif

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/animation_player.hpp>
#endif

class BTPlayAnimation : public BTAction {
	GDCLASS(BTPlayAnimation, BTAction);
	TASK_CATEGORY(Scene);

private:
	Ref<BBNode> animation_player_param;
	StringName animation_name;
	double await_completion = 0.0;
	double blend = -1.0;
	double speed = 1.0;
	bool from_end = false;

	AnimationPlayer *animation_player = nullptr;
	bool setup_failed = false;

protected:
	static void _bind_methods();

	virtual String _generate_name() override;
	virtual void _setup() override;
	virtual void _enter() override;
	virtual Status _tick(double p_delta) override;

public:
	void set_animation_player(Ref<BBNode> p_animation_player);
	Ref<BBNode> get_animation_player() const { return animation_player_param; }

	void set_animation_name(StringName p_animation_name);
	StringName get_animation_name() const { return animation_name; }

	void set_await_completion(double p_await_completion);
	double get_await_completion() const { return await_completion; }

	void set_blend(double p_blend);
	double get_blend() const { return blend; }

	void set_speed(double p_speed);
	double get_speed() const { return speed; }

	void set_from_end(bool p_from_end);
	bool get_from_end() const { return from_end; }

	virtual PackedStringArray get_configuration_warnings() override;
};

#endif // BT_PLAY_ANIMATION_H
