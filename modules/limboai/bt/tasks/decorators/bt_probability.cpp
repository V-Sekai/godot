/**************************************************************************/
/*  bt_probability.cpp                                                    */
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
 * bt_probability.cpp
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_probability.h"

#include "../../../compat/math.h"

void BTProbability::set_run_chance(float p_value) {
	run_chance = p_value;
	emit_changed();
}

String BTProbability::_generate_name() {
	return vformat("Probability %.1f%%", run_chance * 100.0);
}

BT::Status BTProbability::_tick(double p_delta) {
	ERR_FAIL_COND_V_MSG(get_child_count() == 0, FAILURE, "BT decorator has no child.");
	if (get_child(0)->get_status() == RUNNING || RANDF() <= run_chance) {
		return get_child(0)->execute(p_delta);
	}
	return FAILURE;
}

void BTProbability::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_run_chance", "probability"), &BTProbability::set_run_chance);
	ClassDB::bind_method(D_METHOD("get_run_chance"), &BTProbability::get_run_chance);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "run_chance", PROPERTY_HINT_RANGE, "0.0,1.0"), "set_run_chance", "get_run_chance");
}
