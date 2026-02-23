/**************************************************************************/
/*  bt_call_method.h                                                      */
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
 * bt_call_method.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_CALL_METHOD_H
#define BT_CALL_METHOD_H

#include "../bt_action.h"

#include "../../../blackboard/bb_param/bb_node.h"
#include "../../../blackboard/bb_param/bb_variant.h"

class BTCallMethod : public BTAction {
	GDCLASS(BTCallMethod, BTAction);
	TASK_CATEGORY(Utility);

private:
	StringName method;
	Ref<BBNode> node_param;
	TypedArray<BBVariant> args;
	bool include_delta = false;
	StringName result_var;

protected:
	static void _bind_methods();

	virtual String _generate_name() override;
	virtual Status _tick(double p_delta) override;

public:
	void set_method(const StringName &p_method_name);
	StringName get_method() const { return method; }

	void set_node_param(const Ref<BBNode> &p_object);
	Ref<BBNode> get_node_param() const { return node_param; }

	void set_args(TypedArray<BBVariant> p_args);
	TypedArray<BBVariant> get_args() const { return args; }

	void set_include_delta(bool p_include_delta);
	bool is_delta_included() const { return include_delta; }

	void set_result_var(const StringName &p_result_var);
	StringName get_result_var() const { return result_var; }

	virtual PackedStringArray get_configuration_warnings() override;

	BTCallMethod();
};

#endif // BT_CALL_METHOD_H
