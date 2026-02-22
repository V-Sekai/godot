/**************************************************************************/
/*  bt_time_limit.cpp                                                     */
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
 * bt_time_limit.cpp
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_time_limit.h"

void BTTimeLimit::set_time_limit(double p_value) {
	time_limit = p_value;
	emit_changed();
}

String BTTimeLimit::_generate_name() {
	return vformat("TimeLimit %s sec", Math::snapped(time_limit, 0.001));
}

BT::Status BTTimeLimit::_tick(double p_delta) {
	ERR_FAIL_COND_V_MSG(get_child_count() == 0, FAILURE, "BT decorator has no child.");
	Status status = get_child(0)->execute(p_delta);
	if (status == RUNNING && get_elapsed_time() >= time_limit) {
		get_child(0)->abort();
		return FAILURE;
	}
	return status;
}

void BTTimeLimit::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_time_limit", "duration_sec"), &BTTimeLimit::set_time_limit);
	ClassDB::bind_method(D_METHOD("get_time_limit"), &BTTimeLimit::get_time_limit);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "time_limit"), "set_time_limit", "get_time_limit");
}
