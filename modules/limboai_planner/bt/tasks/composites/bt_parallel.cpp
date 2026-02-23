/**************************************************************************/
/*  bt_parallel.cpp                                                       */
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
 * bt_parallel.cpp
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_parallel.h"

void BTParallel::_enter() {
	for (int i = 0; i < get_child_count(); i++) {
		get_child(i)->abort();
	}
}

BT::Status BTParallel::_tick(double p_delta) {
	int num_succeeded = 0;
	int num_failed = 0;
	BT::Status return_status = RUNNING;
	for (int i = 0; i < get_child_count(); i++) {
		Status status = BT::FRESH;
		Ref<BTTask> child = get_child(i);
		if (!repeat && (child->get_status() == FAILURE || child->get_status() == SUCCESS)) {
			status = child->get_status();
		} else {
			status = child->execute(p_delta);
		}
		if (status == FAILURE) {
			num_failed += 1;
			if (num_failed >= num_failures_required && return_status == RUNNING) {
				return_status = FAILURE;
			}
		} else if (status == SUCCESS) {
			num_succeeded += 1;
			if (num_succeeded >= num_successes_required && return_status == RUNNING) {
				return_status = SUCCESS;
			}
		}
	}
	if (!repeat && (num_failed + num_succeeded) == get_child_count() && return_status == RUNNING) {
		return_status = FAILURE;
	}
	return return_status;
}

void BTParallel::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_num_successes_required"), &BTParallel::get_num_successes_required);
	ClassDB::bind_method(D_METHOD("set_num_successes_required", "value"), &BTParallel::set_num_successes_required);
	ClassDB::bind_method(D_METHOD("get_num_failures_required"), &BTParallel::get_num_failures_required);
	ClassDB::bind_method(D_METHOD("set_num_failures_required", "value"), &BTParallel::set_num_failures_required);
	ClassDB::bind_method(D_METHOD("get_repeat"), &BTParallel::get_repeat);
	ClassDB::bind_method(D_METHOD("set_repeat", "enable"), &BTParallel::set_repeat);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "num_successes_required"), "set_num_successes_required", "get_num_successes_required");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "num_failures_required"), "set_num_failures_required", "get_num_failures_required");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "repeat"), "set_repeat", "get_repeat");
}
