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
	ClassDB::bind_method(D_METHOD("get_predicate", "subject", "predicate"), &PlannerState::get_predicate);
	ClassDB::bind_method(D_METHOD("set_predicate", "subject", "predicate", "value"), &PlannerState::set_predicate);
	ClassDB::bind_method(D_METHOD("get_subject_predicate_list"), &PlannerState::get_subject_predicate_list);

	ClassDB::bind_method(D_METHOD("has_subject_variable", "variable"), &PlannerState::has_subject_variable);
	ClassDB::bind_method(D_METHOD("has_predicate", "subject", "predicate"), &PlannerState::has_predicate);

	// Entity capabilities methods
	ClassDB::bind_method(D_METHOD("get_entity_capability", "entity_id", "capability"), &PlannerState::get_entity_capability);
	ClassDB::bind_method(D_METHOD("set_entity_capability", "entity_id", "capability", "value"), &PlannerState::set_entity_capability);
	ClassDB::bind_method(D_METHOD("has_entity", "entity_id"), &PlannerState::has_entity);
	ClassDB::bind_method(D_METHOD("get_all_entities"), &PlannerState::get_all_entities);
}

Variant PlannerState::get_predicate(const String &p_subject, const String &p_predicate) const {
	if (data.has(p_subject)) {
		Dictionary subject_data = data[p_subject];
		if (subject_data.has(p_predicate)) {
			return subject_data[p_predicate];
		}
	}
	return Variant();
}

void PlannerState::set_predicate(const String &p_subject, const String &p_predicate, Variant p_value) {
	bool is_float = p_value.get_type() == Variant::FLOAT;
	bool is_int = p_value.get_type() == Variant::INT;
	bool is_string = p_value.get_type() == Variant::STRING;
	bool is_bool = p_value.get_type() == Variant::BOOL;
	ERR_FAIL_COND(!(is_float || is_int || is_string || is_bool));
	if (data.has(p_subject)) {
		Dictionary subject_data = data[p_subject];
		subject_data[p_predicate] = p_value;
		data[p_subject] = subject_data;
	} else {
		Dictionary subject_data;
		subject_data[p_predicate] = p_value;
		data[p_subject] = subject_data;
	}
}

Array PlannerState::get_subject_predicate_list() const {
	return data.keys();
}

bool PlannerState::has_subject_variable(const String &p_variable) const {
	return data.has(p_variable);
}

bool PlannerState::has_predicate(const String &p_subject, const String &p_predicate) const {
	if (!data.has(p_subject)) {
		return false;
	}
	Dictionary subject_data = data[p_subject];
	return subject_data.has(p_predicate);
}

Variant PlannerState::get_entity_capability(const String &p_entity_id, const String &p_capability) const {
	if (entity_capabilities.has(p_entity_id)) {
		Dictionary entity_data = entity_capabilities[p_entity_id];
		if (entity_data.has(p_capability)) {
			return entity_data[p_capability];
		}
	}
	return Variant();
}

void PlannerState::set_entity_capability(const String &p_entity_id, const String &p_capability, Variant p_value) {
	if (entity_capabilities.has(p_entity_id)) {
		Dictionary entity_data = entity_capabilities[p_entity_id];
		entity_data[p_capability] = p_value;
		entity_capabilities[p_entity_id] = entity_data;
	} else {
		Dictionary entity_data;
		entity_data[p_capability] = p_value;
		entity_capabilities[p_entity_id] = entity_data;
	}
}

bool PlannerState::has_entity(const String &p_entity_id) const {
	return entity_capabilities.has(p_entity_id);
}

Array PlannerState::get_all_entities() const {
	return entity_capabilities.keys();
}
