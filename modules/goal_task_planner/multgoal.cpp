/**************************************************************************/
/*  multgoal.cpp                                                          */
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

#include "multigoal.h"

void Multigoal::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_to_string"), &Multigoal::_to_string);
	ClassDB::bind_method(D_METHOD("get_state"), &Multigoal::get_state);
	ClassDB::bind_method(D_METHOD("set_state", "value"), &Multigoal::set_state);
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "state"), "set_state", "get_state");

	ClassDB::bind_method(D_METHOD("_init", "multigoal_name", "state_variables"), &Multigoal::_init);
	ClassDB::bind_method(D_METHOD("state_vars"), &Multigoal::state_vars);
}

String Multigoal::_to_string() {
	return get_name();
}

Dictionary Multigoal::get_state() const {
	return _state;
}

void Multigoal::set_state(Dictionary value) {
	_state = value;
}

void Multigoal::_init(String multigoal_name, Dictionary state_variables) {
	set_name(multigoal_name);
	_state = state_variables;
}

Array Multigoal::state_vars() {
	return _state.keys();
}
