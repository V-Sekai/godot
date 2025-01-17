/**************************************************************************/
/*  planner_state.cpp                                                     */
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

#include "planner_state.h"

#include "core/error/error_macros.h"
#include "core/object/class_db.h"
#include "core/variant/variant.h"

void PlannerState::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_value", "predicate", "subject"), &PlannerState::get_object);
	ClassDB::bind_method(D_METHOD("set_value", "predicate", "subject", "object"), &PlannerState::set_object);
	ClassDB::bind_method(D_METHOD("get_value_property_list"), &PlannerState::get_subject_property_list);

	ClassDB::bind_method(D_METHOD("has_value_variable", "variable"), &PlannerState::has_subject_variable);
	ClassDB::bind_method(D_METHOD("has_value", "variable", "arguments"), &PlannerState::has_subject);
}

Variant PlannerState::get_object(const String &p_variable, const String &p_arguments) const {
	if (data.has(p_variable)) {
		Dictionary variable_data = data[p_variable];
		if (variable_data.has(p_arguments)) {
			return variable_data[p_arguments];
		}
	}
	return Variant();
}

void PlannerState::set_object(const String &p_variable, const String &p_arguments, Variant p_value) {
	bool is_float = p_value.get_type() == Variant::FLOAT;
	bool is_int = p_value.get_type() == Variant::INT;
	bool is_string = p_value.get_type() == Variant::STRING;
	bool is_bool = p_value.get_type() == Variant::BOOL;
	ERR_FAIL_COND(!(is_float || is_int || is_string || is_bool));
	if (data.has(p_variable)) {
		Dictionary variable_data = data[p_variable];
		variable_data[p_arguments] = p_value;
		data[p_variable] = variable_data;
	} else {
		Dictionary variable_data;
		variable_data[p_arguments] = p_value;
		data[p_variable] = variable_data;
	}
}

Array PlannerState::get_subject_property_list() const {
	return data.keys();
}

bool PlannerState::has_subject_variable(const String &p_variable) const {
	return data.has(p_variable);
}

bool PlannerState::has_subject(const String &p_predicate, const String &p_subject) const {
	if (!data.has(p_predicate)) {
		return false;
	}
	Dictionary variable_data = data[p_predicate];
	return variable_data.has(p_subject);
}
