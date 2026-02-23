/**************************************************************************/
/*  bt_evaluate_expression.h                                              */
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

#pragma once

/**
 * bt_evaluate_expression.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_EVALUATE_EXPRESSION_H
#define BT_EVALUATE_EXPRESSION_H

#include "../bt_action.h"

#include "../../../blackboard/bb_param/bb_node.h"
#include "../../../blackboard/bb_param/bb_variant.h"

#ifdef LIMBOAI_MODULE
#include "core/math/expression.h"
#endif

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/expression.hpp>
#endif

class BTEvaluateExpression : public BTAction {
	GDCLASS(BTEvaluateExpression, BTAction);
	TASK_CATEGORY(Utility);

private:
	Ref<Expression> expression;
	Error is_parsed = FAILED;
	Ref<BBNode> node_param;
	String expression_string;
	PackedStringArray input_names;
	TypedArray<BBVariant> input_values;
	bool input_include_delta = false;
	Array processed_input_values;
	StringName result_var;

protected:
	static void _bind_methods();

	virtual String _generate_name() override;
	virtual void _setup() override;
	virtual Status _tick(double p_delta) override;

public:
	Error parse();

	void set_expression_string(const String &p_expression_string);
	String get_expression_string() const { return expression_string; }

	void set_node_param(Ref<BBNode> p_object);
	Ref<BBNode> get_node_param() const { return node_param; }

	void set_input_names(const PackedStringArray &p_input_names);
	PackedStringArray get_input_names() const { return input_names; }

	void set_input_values(const TypedArray<BBVariant> &p_input_values);
	TypedArray<BBVariant> get_input_values() const { return input_values; }

	void set_input_include_delta(bool p_input_include_delta);
	bool is_input_delta_included() const { return input_include_delta; }

	void set_result_var(const StringName &p_result_var);
	StringName get_result_var() const { return result_var; }

	virtual PackedStringArray get_configuration_warnings() override;

	BTEvaluateExpression();
};

#endif // BT_EVALUATE_EXPRESSION_H
