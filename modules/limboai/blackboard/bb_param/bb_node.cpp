/**************************************************************************/
/*  bb_node.cpp                                                           */
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
 * bb_node.cpp
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bb_node.h"

Variant BBNode::get_value(Node *p_scene_root, const Ref<Blackboard> &p_blackboard, const Variant &p_default) {
	ERR_FAIL_NULL_V_MSG(p_scene_root, Variant(), "BBNode: get_value() failed - scene_root is null.");
	ERR_FAIL_COND_V_MSG(p_blackboard.is_null(), Variant(), "BBNode: get_value() failed - blackboard is null.");

	Variant val;
	if (get_value_source() == SAVED_VALUE) {
		val = get_saved_value();
	} else {
		val = p_blackboard->get_var(get_variable(), p_default);
	}

	if (val.get_type() == Variant::NODE_PATH) {
		return p_scene_root->get_node_or_null(val);
	} else if (val.get_type() == Variant::OBJECT || val.get_type() == Variant::NIL) {
		return val;
	} else {
		WARN_PRINT("BBNode: Unexpected variant type: " + Variant::get_type_name(val.get_type()) + ". Returning default value.");
		return p_default;
	}
}
