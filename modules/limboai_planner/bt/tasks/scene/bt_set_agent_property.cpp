/**************************************************************************/
/*  bt_set_agent_property.cpp                                             */
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
 * bt_set_agent_property.cpp
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_set_agent_property.h"

#include "../../../util/limbo_string_names.h"

#ifdef LIMBOAI_MODULE
#include "core/config/engine.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/engine.hpp>
#endif // LIMBOAI_GDEXTENSION

void BTSetAgentProperty::set_property(StringName p_prop) {
	property = p_prop;
	emit_changed();
}

void BTSetAgentProperty::set_value(Ref<BBVariant> p_value) {
	value = p_value;
	emit_changed();
	if (Engine::get_singleton()->is_editor_hint() && value.is_valid() &&
			!value->is_connected(LW_NAME(changed), callable_mp((Resource *)this, &Resource::emit_changed))) {
		value->connect(LW_NAME(changed), callable_mp((Resource *)this, &Resource::emit_changed));
	}
}

void BTSetAgentProperty::set_operation(LimboUtility::Operation p_operation) {
	operation = p_operation;
	emit_changed();
}

PackedStringArray BTSetAgentProperty::get_configuration_warnings() {
	PackedStringArray warnings = BTAction::get_configuration_warnings();
	if (property == StringName()) {
		warnings.append("`property` should be assigned.");
	}
	if (!value.is_valid()) {
		warnings.append("`value` should be assigned.");
	}
	return warnings;
}

String BTSetAgentProperty::_generate_name() {
	if (property == StringName()) {
		return "SetAgentProperty ???";
	}

	return vformat("Set agent.%s %s= %s",
			property,
			LimboUtility::get_singleton()->get_operation_string(operation),
			value.is_valid() ? Variant(value) : Variant("???"));
}

BT::Status BTSetAgentProperty::_tick(double p_delta) {
	ERR_FAIL_COND_V_MSG(property == StringName(), FAILURE, "BTSetAgentProperty: `property` is not set.");
	ERR_FAIL_COND_V_MSG(!value.is_valid(), FAILURE, "BTSetAgentProperty: `value` is not set.");

	Variant result;
	StringName error_value = LW_NAME(error_value);
	Variant right_value = value->get_value(get_scene_root(), get_blackboard(), error_value);
	ERR_FAIL_COND_V_MSG(right_value == Variant(error_value), FAILURE, "BTSetAgentProperty: Couldn't get value of value-parameter.");
	bool r_valid;
	if (operation == LimboUtility::OPERATION_NONE) {
		result = right_value;
	} else {
#ifdef LIMBOAI_MODULE
		Variant left_value = get_agent()->get(property, &r_valid);
		ERR_FAIL_COND_V_MSG(!r_valid, FAILURE, vformat("BTSetAgentProperty: Failed to get agent's \"%s\" property. Returning FAILURE.", property));
#elif LIMBOAI_GDEXTENSION
		Variant left_value = get_agent()->get(property);
#endif
		result = LimboUtility::get_singleton()->perform_operation(operation, left_value, right_value);
		ERR_FAIL_COND_V_MSG(result == Variant(), FAILURE, "BTSetAgentProperty: Operation not valid. Returning FAILURE.");
	}

#ifdef LIMBOAI_MODULE
	get_agent()->set(property, result, &r_valid);
	ERR_FAIL_COND_V_MSG(!r_valid, FAILURE, vformat("BTSetAgentProperty: Couldn't set property \"%s\" with value \"%s\"", property, result));
#elif LIMBOAI_GDEXTENSION
	get_agent()->set(property, result);
#endif
	return SUCCESS;
}

void BTSetAgentProperty::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_property", "property"), &BTSetAgentProperty::set_property);
	ClassDB::bind_method(D_METHOD("get_property"), &BTSetAgentProperty::get_property);
	ClassDB::bind_method(D_METHOD("set_value", "value"), &BTSetAgentProperty::set_value);
	ClassDB::bind_method(D_METHOD("get_value"), &BTSetAgentProperty::get_value);
	ClassDB::bind_method(D_METHOD("set_operation", "operation"), &BTSetAgentProperty::set_operation);
	ClassDB::bind_method(D_METHOD("get_operation"), &BTSetAgentProperty::get_operation);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "property"), "set_property", "get_property");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "value", PROPERTY_HINT_RESOURCE_TYPE, "BBVariant"), "set_value", "get_value");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "operation", PROPERTY_HINT_ENUM, "None,Addition,Subtraction,Multiplication,Division,Modulo,Power,Bitwise Shift Left,Bitwise Shift Right,Bitwise AND,Bitwise OR,Bitwise XOR"), "set_operation", "get_operation");
}
