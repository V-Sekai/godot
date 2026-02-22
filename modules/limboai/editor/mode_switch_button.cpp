/**************************************************************************/
/*  mode_switch_button.cpp                                                */
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

/**
 * mode_switch_button.cpp
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifdef TOOLS_ENABLED

#include "mode_switch_button.h"

#include "../util/limbo_string_names.h"

void ModeSwitchButton::add_mode(int p_id, const Ref<Texture2D> &p_icon, const String &p_tooltip) {
	bool unique_id = true;
	for (int i = 0; i < modes.size(); i++) {
		if (modes[i].id == p_id) {
			unique_id = false;
			break;
		}
	}
	ERR_FAIL_COND_MSG(unique_id == false, "ID is already in use by another button state: " + itos(p_id));

	Mode mode;
	mode.id = p_id;
	mode.icon = p_icon;
	mode.tooltip = p_tooltip;
	modes.append(mode);

	if (current_mode_index == -1) {
		_set_mode_by_index(0);
	}
}

void ModeSwitchButton::set_mode(int p_id, bool p_no_signal) {
	ERR_FAIL_COND_MSG(modes.is_empty(), "Cannot set button state with zero states.");

	int idx = -1;
	for (int i = 0; i < modes.size(); i++) {
		if (modes[i].id == p_id) {
			idx = i;
			break;
		}
	}
	ERR_FAIL_COND_MSG(idx == -1, "Button state not found with such id: " + itos(p_id));

	_set_mode_by_index(idx);
	if (!p_no_signal) {
		emit_signal(LW_NAME(mode_changed));
	}
}

void ModeSwitchButton::clear() {
	current_mode_index = -1;
	modes.clear();
}

void ModeSwitchButton::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			connect(LW_NAME(pressed), callable_mp(this, &ModeSwitchButton::next_mode));
		} break;
	}
}

void ModeSwitchButton::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_mode", "id", "icon", "tooltip"), &ModeSwitchButton::add_mode);
	ClassDB::bind_method(D_METHOD("get_mode"), &ModeSwitchButton::get_mode);
	ClassDB::bind_method(D_METHOD("set_mode", "id"), &ModeSwitchButton::set_mode);
	ClassDB::bind_method(D_METHOD("next_mode"), &ModeSwitchButton::next_mode);

	ADD_SIGNAL(MethodInfo("mode_changed"));
}

ModeSwitchButton::ModeSwitchButton() {
}

#endif // ! TOOLS_ENABLED
