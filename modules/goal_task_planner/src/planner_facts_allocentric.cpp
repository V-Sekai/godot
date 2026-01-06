/**************************************************************************/
/*  planner_facts_allocentric.cpp                                         */
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

// SPDX-FileCopyrightText: 2025-present K. S. Ernest (iFire) Lee
// SPDX-License-Identifier: MIT

#include "planner_facts_allocentric.h"

#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"

void PlannerFactsAllocentric::_bind_methods() {
	// Terrain facts
	ClassDB::bind_method(D_METHOD("set_terrain_fact", "location", "fact_key", "value"), &PlannerFactsAllocentric::set_terrain_fact);
	ClassDB::bind_method(D_METHOD("get_terrain_fact", "location", "fact_key"), &PlannerFactsAllocentric::get_terrain_fact);
	ClassDB::bind_method(D_METHOD("has_terrain_fact", "location", "fact_key"), &PlannerFactsAllocentric::has_terrain_fact);
	ClassDB::bind_method(D_METHOD("get_all_terrain_facts"), &PlannerFactsAllocentric::get_all_terrain_facts);

	// Shared objects
	ClassDB::bind_method(D_METHOD("add_shared_object", "object_id", "object_data"), &PlannerFactsAllocentric::add_shared_object);
	ClassDB::bind_method(D_METHOD("remove_shared_object", "object_id"), &PlannerFactsAllocentric::remove_shared_object);
	ClassDB::bind_method(D_METHOD("get_shared_object", "object_id"), &PlannerFactsAllocentric::get_shared_object);
	ClassDB::bind_method(D_METHOD("has_shared_object", "object_id"), &PlannerFactsAllocentric::has_shared_object);
	ClassDB::bind_method(D_METHOD("get_all_shared_object_ids"), &PlannerFactsAllocentric::get_all_shared_object_ids);
	ClassDB::bind_method(D_METHOD("get_all_shared_objects"), &PlannerFactsAllocentric::get_all_shared_objects);

	// Public events
	ClassDB::bind_method(D_METHOD("add_public_event", "event_id", "event_data"), &PlannerFactsAllocentric::add_public_event);
	ClassDB::bind_method(D_METHOD("remove_public_event", "event_id"), &PlannerFactsAllocentric::remove_public_event);
	ClassDB::bind_method(D_METHOD("get_public_event", "event_id"), &PlannerFactsAllocentric::get_public_event);
	ClassDB::bind_method(D_METHOD("has_public_event", "event_id"), &PlannerFactsAllocentric::has_public_event);
	ClassDB::bind_method(D_METHOD("get_all_public_event_ids"), &PlannerFactsAllocentric::get_all_public_event_ids);
	ClassDB::bind_method(D_METHOD("get_all_public_events"), &PlannerFactsAllocentric::get_all_public_events);

	// Entity positions
	ClassDB::bind_method(D_METHOD("set_entity_position", "entity_id", "position"), &PlannerFactsAllocentric::set_entity_position);
	ClassDB::bind_method(D_METHOD("get_entity_position", "entity_id"), &PlannerFactsAllocentric::get_entity_position);
	ClassDB::bind_method(D_METHOD("has_entity_position", "entity_id"), &PlannerFactsAllocentric::has_entity_position);
	ClassDB::bind_method(D_METHOD("get_all_entity_positions"), &PlannerFactsAllocentric::get_all_entity_positions);

	// Entity capabilities (public)
	ClassDB::bind_method(D_METHOD("set_entity_capability_public", "entity_id", "capability", "value"), &PlannerFactsAllocentric::set_entity_capability_public);
	ClassDB::bind_method(D_METHOD("get_entity_capability_public", "entity_id", "capability"), &PlannerFactsAllocentric::get_entity_capability_public);
	ClassDB::bind_method(D_METHOD("has_entity_capability_public", "entity_id", "capability"), &PlannerFactsAllocentric::has_entity_capability_public);
	ClassDB::bind_method(D_METHOD("get_all_entity_capabilities_public"), &PlannerFactsAllocentric::get_all_entity_capabilities_public);

	// Observation interface
	ClassDB::bind_method(D_METHOD("observe_terrain", "location"), &PlannerFactsAllocentric::observe_terrain);
	ClassDB::bind_method(D_METHOD("observe_shared_objects", "location"), &PlannerFactsAllocentric::observe_shared_objects, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("observe_public_events"), &PlannerFactsAllocentric::observe_public_events);
	ClassDB::bind_method(D_METHOD("observe_entity_positions"), &PlannerFactsAllocentric::observe_entity_positions);
	ClassDB::bind_method(D_METHOD("observe_entity_capabilities"), &PlannerFactsAllocentric::observe_entity_capabilities);

	// Utility
	ClassDB::bind_method(D_METHOD("clear_all"), &PlannerFactsAllocentric::clear_all);
}

PlannerFactsAllocentric::PlannerFactsAllocentric() {
	terrain_facts = Dictionary();
	shared_objects = Dictionary();
	public_events = Dictionary();
	entity_positions = Dictionary();
	entity_capabilities_public = Dictionary();
}

PlannerFactsAllocentric::~PlannerFactsAllocentric() {
}

void PlannerFactsAllocentric::set_terrain_fact(const String &p_location, const String &p_fact_key, const Variant &p_value) {
	if (!terrain_facts.has(p_location)) {
		terrain_facts[p_location] = Dictionary();
	}
	Dictionary location_facts = terrain_facts[p_location];
	location_facts[p_fact_key] = p_value;
	terrain_facts[p_location] = location_facts;
}

Variant PlannerFactsAllocentric::get_terrain_fact(const String &p_location, const String &p_fact_key) const {
	if (terrain_facts.has(p_location)) {
		Dictionary location_facts = terrain_facts[p_location];
		if (location_facts.has(p_fact_key)) {
			return location_facts[p_fact_key];
		}
	}
	return Variant();
}

bool PlannerFactsAllocentric::has_terrain_fact(const String &p_location, const String &p_fact_key) const {
	if (terrain_facts.has(p_location)) {
		Dictionary location_facts = terrain_facts[p_location];
		return location_facts.has(p_fact_key);
	}
	return false;
}

void PlannerFactsAllocentric::add_shared_object(const String &p_object_id, const Dictionary &p_object_data) {
	shared_objects[p_object_id] = p_object_data;
}

void PlannerFactsAllocentric::remove_shared_object(const String &p_object_id) {
	shared_objects.erase(p_object_id);
}

Dictionary PlannerFactsAllocentric::get_shared_object(const String &p_object_id) const {
	if (shared_objects.has(p_object_id)) {
		return shared_objects[p_object_id];
	}
	return Dictionary();
}

bool PlannerFactsAllocentric::has_shared_object(const String &p_object_id) const {
	return shared_objects.has(p_object_id);
}

TypedArray<String> PlannerFactsAllocentric::get_all_shared_object_ids() const {
	TypedArray<String> ids;
	Array keys = shared_objects.keys();
	for (int i = 0; i < keys.size(); i++) {
		ids.push_back(keys[i]);
	}
	return ids;
}

void PlannerFactsAllocentric::add_public_event(const String &p_event_id, const Dictionary &p_event_data) {
	public_events[p_event_id] = p_event_data;
}

void PlannerFactsAllocentric::remove_public_event(const String &p_event_id) {
	public_events.erase(p_event_id);
}

Dictionary PlannerFactsAllocentric::get_public_event(const String &p_event_id) const {
	if (public_events.has(p_event_id)) {
		return public_events[p_event_id];
	}
	return Dictionary();
}

bool PlannerFactsAllocentric::has_public_event(const String &p_event_id) const {
	return public_events.has(p_event_id);
}

TypedArray<String> PlannerFactsAllocentric::get_all_public_event_ids() const {
	TypedArray<String> ids;
	Array keys = public_events.keys();
	for (int i = 0; i < keys.size(); i++) {
		ids.push_back(keys[i]);
	}
	return ids;
}

void PlannerFactsAllocentric::set_entity_position(const String &p_entity_id, const Variant &p_position) {
	entity_positions[p_entity_id] = p_position;
}

Variant PlannerFactsAllocentric::get_entity_position(const String &p_entity_id) const {
	if (entity_positions.has(p_entity_id)) {
		return entity_positions[p_entity_id];
	}
	return Variant();
}

bool PlannerFactsAllocentric::has_entity_position(const String &p_entity_id) const {
	return entity_positions.has(p_entity_id);
}

void PlannerFactsAllocentric::set_entity_capability_public(const String &p_entity_id, const String &p_capability, const Variant &p_value) {
	if (!entity_capabilities_public.has(p_entity_id)) {
		entity_capabilities_public[p_entity_id] = Dictionary();
	}
	Dictionary entity_caps = entity_capabilities_public[p_entity_id];
	entity_caps[p_capability] = p_value;
	entity_capabilities_public[p_entity_id] = entity_caps;
}

Variant PlannerFactsAllocentric::get_entity_capability_public(const String &p_entity_id, const String &p_capability) const {
	if (entity_capabilities_public.has(p_entity_id)) {
		Dictionary entity_caps = entity_capabilities_public[p_entity_id];
		if (entity_caps.has(p_capability)) {
			return entity_caps[p_capability];
		}
	}
	return Variant();
}

bool PlannerFactsAllocentric::has_entity_capability_public(const String &p_entity_id, const String &p_capability) const {
	if (entity_capabilities_public.has(p_entity_id)) {
		Dictionary entity_caps = entity_capabilities_public[p_entity_id];
		return entity_caps.has(p_capability);
	}
	return false;
}

Dictionary PlannerFactsAllocentric::observe_terrain(const String &p_location) const {
	if (terrain_facts.has(p_location)) {
		return terrain_facts[p_location];
	}
	return Dictionary();
}

Dictionary PlannerFactsAllocentric::observe_shared_objects(const String &p_location) const {
	if (p_location.is_empty()) {
		return shared_objects.duplicate(true);
	}

	// Filter objects by location if specified
	Dictionary filtered;
	Array keys = shared_objects.keys();
	for (int i = 0; i < keys.size(); i++) {
		String object_id = keys[i];
		Dictionary object_data = shared_objects[object_id];
		if (object_data.has("location") && object_data["location"] == p_location) {
			filtered[object_id] = object_data;
		}
	}
	return filtered;
}

Dictionary PlannerFactsAllocentric::observe_public_events() const {
	return public_events.duplicate(true);
}

Dictionary PlannerFactsAllocentric::observe_entity_positions() const {
	return entity_positions.duplicate(true);
}

Dictionary PlannerFactsAllocentric::observe_entity_capabilities() const {
	return entity_capabilities_public.duplicate(true);
}

void PlannerFactsAllocentric::clear_all() {
	terrain_facts.clear();
	shared_objects.clear();
	public_events.clear();
	entity_positions.clear();
	entity_capabilities_public.clear();
}
