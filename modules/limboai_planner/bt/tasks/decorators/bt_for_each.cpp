/**************************************************************************/
/*  bt_for_each.cpp                                                       */
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
 * bt_for_each.cpp
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_for_each.h"

#include "../../../blackboard/blackboard.h"
#include "../../../util/limbo_utility.h"

//**** Setters / Getters

void BTForEach::set_array_var(const StringName &p_value) {
	array_var = p_value;
	emit_changed();
}

void BTForEach::set_save_var(const StringName &p_value) {
	save_var = p_value;
	emit_changed();
}

//**** Task Implementation

String BTForEach::_generate_name() {
	return vformat("ForEach %s in %s",
			LimboUtility::get_singleton()->decorate_var(save_var),
			LimboUtility::get_singleton()->decorate_var(array_var));
}

void BTForEach::_enter() {
	current_idx = 0;
}

BT::Status BTForEach::_tick(double p_delta) {
	ERR_FAIL_COND_V_MSG(get_child_count() == 0, FAILURE, "BTForEach: Decorator has no child.");
	ERR_FAIL_COND_V_MSG(save_var == StringName(), FAILURE, "BTForEach: Save variable is not set.");
	ERR_FAIL_COND_V_MSG(array_var == StringName(), FAILURE, "BTForEach: Array variable is not set.");

	Array arr = get_blackboard()->get_var(array_var, Variant());
	if (current_idx >= arr.size()) {
		if (current_idx != 0) {
			WARN_PRINT("BTForEach: Array size changed during iteration.");
		}
		return SUCCESS;
	}
	Variant elem = arr[current_idx];
	get_blackboard()->set_var(save_var, elem);

	Status status = get_child(0)->execute(p_delta);
	if (status == RUNNING) {
		return RUNNING;
	} else if (status == FAILURE) {
		return FAILURE;
	} else if (current_idx == (arr.size() - 1)) {
		return SUCCESS;
	} else {
		current_idx += 1;
		return RUNNING;
	}
}

//**** Godot

void BTForEach::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_array_var", "variable"), &BTForEach::set_array_var);
	ClassDB::bind_method(D_METHOD("get_array_var"), &BTForEach::get_array_var);
	ClassDB::bind_method(D_METHOD("set_save_var", "variable"), &BTForEach::set_save_var);
	ClassDB::bind_method(D_METHOD("get_save_var"), &BTForEach::get_save_var);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "array_var"), "set_array_var", "get_array_var");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "save_var"), "set_save_var", "get_save_var");
}
