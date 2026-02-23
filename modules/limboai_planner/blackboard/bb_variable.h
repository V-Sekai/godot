/**************************************************************************/
/*  bb_variable.h                                                         */
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
 * bb_variable.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BB_VARIABLE_H
#define BB_VARIABLE_H

#ifdef LIMBOAI_MODULE
#include "core/object/object.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/templates/safe_refcount.hpp>
#include <godot_cpp/variant/variant.hpp>
using namespace godot;
#endif // LIMBOAI_GDEXTENSION

class BBVariable {
private:
	struct Data {
		// Is used to decide if the value needs to be synced in a derived plan.
		bool value_changed = false;

		SafeRefCount refcount;
		Variant value;
		Variant::Type type = Variant::NIL;
		PropertyHint hint = PropertyHint::PROPERTY_HINT_NONE;
		String hint_string;

		NodePath binding_path;
		uint64_t bound_object = 0;
		StringName bound_property;
	};

	Data *data = nullptr;
	void unref();

public:
	void set_value(const Variant &p_value);
	Variant get_value() const;

	void set_type(Variant::Type p_type);
	Variant::Type get_type() const;

	void set_hint(PropertyHint p_hint);
	PropertyHint get_hint() const;

	void set_hint_string(const String &p_hint_string);
	String get_hint_string() const;

	BBVariable duplicate(bool p_deep = false) const;

	_FORCE_INLINE_ bool is_value_changed() const { return data->value_changed; }
	_FORCE_INLINE_ void reset_value_changed() { data->value_changed = false; }

	bool is_same_prop_info(const BBVariable &p_other) const;
	void copy_prop_info(const BBVariable &p_other);

	// * Editor binding methods
	NodePath get_binding_path() const { return data->binding_path; }
	void set_binding_path(const NodePath &p_binding_path) { data->binding_path = p_binding_path; }
	bool has_binding() { return data->binding_path.is_empty(); }

	// * Runtime binding methods
	_FORCE_INLINE_ bool is_bound() const { return data->bound_object != 0; }
	void bind(Object *p_object, const StringName &p_property);
	void unbind();

	bool operator==(const BBVariable &p_var) const;
	bool operator!=(const BBVariable &p_var) const;
	void operator=(const BBVariable &p_var);

	BBVariable(const BBVariable &p_var);
	BBVariable(Variant::Type p_type = Variant::Type::NIL, PropertyHint p_hint = PROPERTY_HINT_NONE, const String &p_hint_string = "");
	~BBVariable();
};

#endif // BB_VARIABLE_H
