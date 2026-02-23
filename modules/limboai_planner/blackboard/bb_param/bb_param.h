/**************************************************************************/
/*  bb_param.h                                                            */
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
 * bb_param.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BB_PARAM_H
#define BB_PARAM_H

#include "../../blackboard/blackboard.h"
#include "../../util/limbo_utility.h"

#ifdef LIMBOAI_MODULE
#include "core/io/resource.h"
#include "scene/main/node.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/resource.hpp>
#endif // LIMBOAI_GDEXTENSION

class BBParam : public Resource {
	GDCLASS(BBParam, Resource);

public:
	enum ValueSource : unsigned int {
		SAVED_VALUE,
		BLACKBOARD_VAR
	};

private:
	ValueSource value_source;
	Variant saved_value;
	StringName variable;

	_FORCE_INLINE_ void _update_name() {
		set_name((value_source == SAVED_VALUE) ? String(saved_value) : LimboUtility::get_singleton()->decorate_var(variable));
	}

protected:
	static void _bind_methods();

	void _assign_default_value();

	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	void set_value_source(ValueSource p_value);
	ValueSource get_value_source() const { return value_source; }

	void set_saved_value(Variant p_value);
	Variant get_saved_value();

	void set_variable(const StringName &p_variable);
	StringName get_variable() const { return variable; }

#ifdef LIMBOAI_MODULE
	virtual String _to_string() override;
#elif LIMBOAI_GDEXTENSION
	virtual String _to_string();
#endif

	virtual Variant::Type get_type() const { return Variant::NIL; }
	virtual Variant::Type get_variable_expected_type() const { return get_type(); }
	virtual Variant get_value(Node *p_scene_root, const Ref<Blackboard> &p_blackboard, const Variant &p_default = Variant());

	BBParam();
};

#endif // BB_PARAM_H
