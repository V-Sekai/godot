/**************************************************************************/
/*  planner_state.h                                                       */
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

#include "core/io/resource.h"
#include "core/object/object_id.h" // Include for ObjectID
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"

class PlannerState : public Resource {
	GDCLASS(PlannerState, Resource);

	Dictionary data;
	Dictionary entity_capabilities; // entity_id -> Dictionary of capabilities

protected:
	static void _bind_methods();

public:
	Variant get_predicate(const String &p_subject, const String &p_predicate) const;
	void set_predicate(const String &p_subject, const String &p_predicate, Variant p_value);
	Array get_subject_predicate_list() const;
	bool has_subject_variable(const String &p_variable) const;
	bool has_predicate(const String &p_subject, const String &p_predicate) const;

	// Entity capabilities methods
	Variant get_entity_capability(const String &p_entity_id, const String &p_capability) const;
	void set_entity_capability(const String &p_entity_id, const String &p_capability, Variant p_value);
	bool has_entity(const String &p_entity_id) const;
	Array get_all_entities() const;

	PlannerState() {}
	~PlannerState() {}
};
