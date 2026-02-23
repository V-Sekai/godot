/**************************************************************************/
/*  bt_run_limit.cpp                                                      */
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
 * bt_run_limit.cpp
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_run_limit.h"

void BTRunLimit::set_run_limit(int p_value) {
	run_limit = p_value;
	emit_changed();
}

void BTRunLimit::set_count_policy(CountPolicy p_policy) {
	count_policy = p_policy;
	emit_changed();
}

String BTRunLimit::_generate_name() {
	return vformat("RunLimit x%d", run_limit);
}

BT::Status BTRunLimit::_tick(double p_delta) {
	ERR_FAIL_COND_V_MSG(get_child_count() == 0, FAILURE, "BT decorator has no child.");
	if (num_runs >= run_limit) {
		return FAILURE;
	}
	Status child_status = get_child(0)->execute(p_delta);
	if ((count_policy == COUNT_SUCCESSFUL && child_status == SUCCESS) ||
			(count_policy == COUNT_FAILED && child_status == FAILURE) ||
			(count_policy == COUNT_ALL && child_status != RUNNING)) {
		num_runs += 1;
	}
	return child_status;
}

void BTRunLimit::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_run_limit", "max_runs"), &BTRunLimit::set_run_limit);
	ClassDB::bind_method(D_METHOD("get_run_limit"), &BTRunLimit::get_run_limit);
	ClassDB::bind_method(D_METHOD("set_count_policy", "count_policy"), &BTRunLimit::set_count_policy);
	ClassDB::bind_method(D_METHOD("get_count_policy"), &BTRunLimit::get_count_policy);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "run_limit"), "set_run_limit", "get_run_limit");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "count_policy", PROPERTY_HINT_ENUM, "COUNT_SUCCESSFUL,COUNT_FAILED,COUNT_ALL"), "set_count_policy", "get_count_policy");

	BIND_ENUM_CONSTANT(COUNT_SUCCESSFUL);
	BIND_ENUM_CONSTANT(COUNT_FAILED);
	BIND_ENUM_CONSTANT(COUNT_ALL);
}
