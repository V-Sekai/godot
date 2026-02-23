/**************************************************************************/
/*  mode_switch_button.h                                                  */
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
 * mode_switch_button.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifdef TOOLS_ENABLED

#ifndef MODE_SWITCH_BUTTON
#define MODE_SWITCH_BUTTON

#ifdef LIMBOAI_MODULE
#include "scene/gui/button.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/button.hpp>
using namespace godot;
#endif // LIMBOAI_GDEXTENSION

class ModeSwitchButton : public Button {
	GDCLASS(ModeSwitchButton, Button);

private:
	struct Mode {
		int id = 0;
		Ref<Texture2D> icon = nullptr;
		String tooltip = "";
	};
	int current_mode_index = -1;

	Vector<Mode> modes;

	_FORCE_INLINE_ void _set_mode_by_index(int p_index) {
		current_mode_index = p_index;
		this->set_button_icon(modes[current_mode_index].icon);
		if (!modes[current_mode_index].tooltip.is_empty()) {
			set_tooltip_text(modes[current_mode_index].tooltip);
		}
	}

protected:
	static void _bind_methods();

	void _notification(int p_what);

public:
	void add_mode(int p_id, const Ref<Texture2D> &p_icon, const String &p_tooltip = "");
	int get_mode() const { return modes.size() > 0 ? modes[current_mode_index].id : -1; }
	void set_mode(int p_id, bool p_no_signal = false);
	_FORCE_INLINE_ void next_mode() { set_mode((current_mode_index + 1) % modes.size()); };
	void clear();

	ModeSwitchButton();
};

#endif // MODE_SWITCH_BUTTON_H

#endif // ! TOOLS_ENABLED
