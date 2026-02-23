/**************************************************************************/
/*  multigoal.h                                                           */
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

// SPDX-FileCopyrightText: 2021 University of Maryland
// SPDX-License-Identifier: BSD-3-Clause-Clear
// Author: Dana Nau <nau@umd.edu>, July 7, 2021

#include "core/object/object.h"
#include "core/variant/dictionary.h"

// PlannerMultigoal is a utility class for working with Array-based multigoals
// Multigoals are represented as Array of unigoal arrays: [[predicate, subject, value], ...]
// Each unigoal is [predicate, subject, value] where predicate is the state variable name
class PlannerMultigoal : public Object {
	GDCLASS(PlannerMultigoal, Object);

public:
	// Check if a Variant is an Array multigoal (Array of unigoal arrays)
	// A multigoal is an Array where the first element is also an Array (unigoal)
	static bool is_multigoal_array(const Variant &p_variant);

	// Multigoal tag support
	// Get goal_tag from multigoal (supports both Array and Dictionary-wrapped formats)
	static String get_goal_tag(const Variant &p_multigoal);
	// Set goal_tag on multigoal (wraps in Dictionary if needed)
	static Variant set_goal_tag(const Variant &p_multigoal, const String &p_tag);

	// Static methods for multigoal operations
	static Array method_goals_not_achieved(const Dictionary &p_state, const Array &p_multigoal_array);
	static Variant method_verify_multigoal(const Dictionary &p_state, const String &p_method, const Array &p_multigoal_array, int p_depth, int p_verbose);

protected:
	static void _bind_methods();
};
