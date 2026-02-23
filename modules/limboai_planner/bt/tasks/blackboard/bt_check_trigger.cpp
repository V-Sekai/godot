/**************************************************************************/
/*  bt_check_trigger.cpp                                                  */
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
 * bt_check_trigger.cpp
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_check_trigger.h"

#include "../../../util/limbo_utility.h"

void BTCheckTrigger::set_variable(const StringName &p_variable) {
	variable = p_variable;
	emit_changed();
}

PackedStringArray BTCheckTrigger::get_configuration_warnings() {
	PackedStringArray warnings = BTCondition::get_configuration_warnings();
	if (variable == StringName()) {
		warnings.append("Variable is not set.");
	}
	return warnings;
}

String BTCheckTrigger::_generate_name() {
	if (variable == StringName()) {
		return "CheckTrigger ???";
	}
	return "CheckTrigger " + LimboUtility::get_singleton()->decorate_var(variable);
}

BT::Status BTCheckTrigger::_tick(double p_delta) {
	ERR_FAIL_COND_V_MSG(variable == StringName(), FAILURE, "BBCheckVar: `variable` is not set.");
	Variant trigger_value = get_blackboard()->get_var(variable, false);
	if (trigger_value == Variant(true)) {
		get_blackboard()->set_var(variable, false);
		return SUCCESS;
	}
	return FAILURE;
}

void BTCheckTrigger::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_variable", "variable"), &BTCheckTrigger::set_variable);
	ClassDB::bind_method(D_METHOD("get_variable"), &BTCheckTrigger::get_variable);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "variable"), "set_variable", "get_variable");
}
