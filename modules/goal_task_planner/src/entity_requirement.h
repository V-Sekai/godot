/**************************************************************************/
/*  entity_requirement.h                                                  */
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

#include "core/string/ustring.h"
#include "core/templates/local_vector.h"
#include "core/variant/array.h"
#include "core/variant/dictionary.h"

// Entity requirement for goal solving
// Matches entities by type and required capabilities
class PlannerEntityRequirement {
public:
	String type; // Entity type (e.g., "agent", "robot")
	LocalVector<String> capabilities; // Required capabilities (e.g., ["cooking", "cleaning"])

	PlannerEntityRequirement() {}

	PlannerEntityRequirement(const String &p_type, const LocalVector<String> &p_capabilities) :
			type(p_type), capabilities(p_capabilities) {}

	// Validation
	bool is_valid() const {
		return !type.is_empty() && capabilities.size() > 0;
	}

	// Convert to Dictionary for GDScript interface
	Dictionary to_dictionary() const {
		Dictionary dict;
		dict["type"] = type;
		Array caps_array;
		for (uint32_t i = 0; i < capabilities.size(); i++) {
			caps_array.push_back(capabilities[i]);
		}
		dict["capabilities"] = caps_array;
		return dict;
	}

	// Convert from Dictionary (GDScript interface)
	static PlannerEntityRequirement from_dictionary(const Dictionary &p_dict) {
		PlannerEntityRequirement req;
		req.type = p_dict.get("type", "");

		Variant caps_var = p_dict.get("capabilities", Array());
		if (caps_var.get_type() == Variant::ARRAY) {
			Array caps_array = caps_var;
			req.capabilities.resize(caps_array.size());
			for (int i = 0; i < caps_array.size(); i++) {
				req.capabilities[i] = caps_array[i];
			}
		}
		return req;
	}
};
