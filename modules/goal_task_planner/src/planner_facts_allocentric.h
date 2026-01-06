/**************************************************************************/
/*  planner_facts_allocentric.h                                           */
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

// SPDX-FileCopyrightText: 2025-present K. S. Ernest (iFire) Lee
// SPDX-License-Identifier: MIT

#include "core/io/resource.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"

// Allocentric Facts - Shared ground truth observable by all personas
// Represents terrain, shared objects, public events that all personas can observe
class PlannerFactsAllocentric : public Resource {
	GDCLASS(PlannerFactsAllocentric, Resource);

private:
	Dictionary terrain_facts; // Terrain information (locations, paths, obstacles)
	Dictionary shared_objects; // Shared objects that all personas can observe
	Dictionary public_events; // Public events observable by all personas
	Dictionary entity_positions; // Observable entity positions (public information)
	Dictionary entity_capabilities_public; // Publicly observable entity capabilities

protected:
	static void _bind_methods();

public:
	PlannerFactsAllocentric();
	~PlannerFactsAllocentric();

	// Terrain facts
	void set_terrain_fact(const String &p_location, const String &p_fact_key, const Variant &p_value);
	Variant get_terrain_fact(const String &p_location, const String &p_fact_key) const;
	bool has_terrain_fact(const String &p_location, const String &p_fact_key) const;
	Dictionary get_all_terrain_facts() const { return terrain_facts; }

	// Shared objects
	void add_shared_object(const String &p_object_id, const Dictionary &p_object_data);
	void remove_shared_object(const String &p_object_id);
	Dictionary get_shared_object(const String &p_object_id) const;
	bool has_shared_object(const String &p_object_id) const;
	TypedArray<String> get_all_shared_object_ids() const;
	Dictionary get_all_shared_objects() const { return shared_objects; }

	// Public events
	void add_public_event(const String &p_event_id, const Dictionary &p_event_data);
	void remove_public_event(const String &p_event_id);
	Dictionary get_public_event(const String &p_event_id) const;
	bool has_public_event(const String &p_event_id) const;
	TypedArray<String> get_all_public_event_ids() const;
	Dictionary get_all_public_events() const { return public_events; }

	// Entity positions (publicly observable)
	void set_entity_position(const String &p_entity_id, const Variant &p_position);
	Variant get_entity_position(const String &p_entity_id) const;
	bool has_entity_position(const String &p_entity_id) const;
	Dictionary get_all_entity_positions() const { return entity_positions; }

	// Public entity capabilities (observable by all)
	void set_entity_capability_public(const String &p_entity_id, const String &p_capability, const Variant &p_value);
	Variant get_entity_capability_public(const String &p_entity_id, const String &p_capability) const;
	bool has_entity_capability_public(const String &p_entity_id, const String &p_capability) const;
	Dictionary get_all_entity_capabilities_public() const { return entity_capabilities_public; }

	// Observation interface - personas can observe allocentric facts
	Dictionary observe_terrain(const String &p_location) const;
	Dictionary observe_shared_objects(const String &p_location = "") const; // Empty string = all objects
	Dictionary observe_public_events() const;
	Dictionary observe_entity_positions() const;
	Dictionary observe_entity_capabilities() const;

	// Clear all facts (for testing/reset)
	void clear_all();
};
