#include "planner_state.h"

#include "core/error/error_macros.h"
#include "core/object/class_db.h"
#include "core/variant/variant.h"

void PlannerState::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_value", "predicate", "subject"), &PlannerState::get_value);
	ClassDB::bind_method(D_METHOD("set_value", "predicate", "subject", "object"), &PlannerState::set_value);
	ClassDB::bind_method(D_METHOD("get_value_property_list"), &PlannerState::get_value_property_list);

	ClassDB::bind_method(D_METHOD("has_value_variable", "variable"), &PlannerState::has_value_variable);
	ClassDB::bind_method(D_METHOD("has_value", "variable", "arguments"), &PlannerState::has_value);

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_data", "get_data");
}

Variant PlannerState::get_value(const String &p_variable, const String &p_arguments) const {
	if (data.has(p_variable)) {
		Dictionary variable_data = data[p_variable];
		if (variable_data.has(p_arguments)) {
			return variable_data[p_arguments];
		}
	}
	return Variant();
}

void PlannerState::set_value(const String &p_variable, const String &p_arguments, Variant p_value) {
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

Array PlannerState::get_value_property_list() const {
	return data.keys();
}

bool PlannerState::has_value_variable(const String &p_variable) const {
	return data.has(p_variable);
}

bool PlannerState::has_value(const String &p_variable, const String &p_arguments) const {
	if (data.has(p_variable)) {
		Dictionary variable_data = data[p_variable];
		return variable_data.has(p_arguments);
	}
	return false;
}
