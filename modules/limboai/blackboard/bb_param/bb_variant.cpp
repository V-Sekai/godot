/**************************************************************************/
/*  bb_variant.cpp                                                        */
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
 * bb_variant.cpp
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bb_variant.h"

void BBVariant::set_type(Variant::Type p_type) {
	if (type != p_type) {
		type = p_type;
		if (get_saved_value().get_type() != p_type) {
			_assign_default_value();
		}
		emit_changed();
		notify_property_list_changed();
	}
}

Variant::Type BBVariant::get_type() const {
	return type;
}

void BBVariant::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_type", "type"), &BBVariant::set_type);

	String vtypes;
	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		if (i > 0) {
			vtypes += ",";
		}
		vtypes += Variant::get_type_name(Variant::Type(i));
	}
	ADD_PROPERTY(PropertyInfo(Variant::INT, "type", PROPERTY_HINT_ENUM, vtypes), "set_type", "get_type");
}

BBVariant::BBVariant(const Variant &p_value) {
	set_type(p_value.get_type());
	set_saved_value(p_value);
}

BBVariant::BBVariant() {
}
