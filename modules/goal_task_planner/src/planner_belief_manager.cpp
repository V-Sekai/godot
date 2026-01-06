/**************************************************************************/
/*  planner_belief_manager.cpp                                            */
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

#include "planner_belief_manager.h"

#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"

void PlannerBeliefManager::_bind_methods() {
	ClassDB::bind_method(D_METHOD("register_persona", "persona"), &PlannerBeliefManager::register_persona);
	ClassDB::bind_method(D_METHOD("unregister_persona", "persona_id"), &PlannerBeliefManager::unregister_persona);
	ClassDB::bind_method(D_METHOD("get_persona", "persona_id"), &PlannerBeliefManager::get_persona);
	ClassDB::bind_method(D_METHOD("has_persona", "persona_id"), &PlannerBeliefManager::has_persona);
	ClassDB::bind_method(D_METHOD("get_all_persona_ids"), &PlannerBeliefManager::get_all_persona_ids);

	ClassDB::bind_method(D_METHOD("get_beliefs_about", "persona", "target_persona_id"), &PlannerBeliefManager::get_beliefs_about);
	ClassDB::bind_method(D_METHOD("get_planner_state", "target_persona_id", "requesting_persona_id"), &PlannerBeliefManager::get_planner_state);

	ClassDB::bind_method(D_METHOD("process_observation_for_persona", "persona_id", "observation"), &PlannerBeliefManager::process_observation_for_persona);
	ClassDB::bind_method(D_METHOD("process_communication_for_persona", "persona_id", "communication"), &PlannerBeliefManager::process_communication_for_persona);

	ClassDB::bind_method(D_METHOD("clear_all"), &PlannerBeliefManager::clear_all);
}

PlannerBeliefManager::PlannerBeliefManager() {
	persona_registry = Dictionary();
}

PlannerBeliefManager::~PlannerBeliefManager() {
}

void PlannerBeliefManager::register_persona(Ref<PlannerPersona> p_persona) {
	if (p_persona.is_valid()) {
		persona_registry[p_persona->get_persona_id()] = p_persona;
	}
}

void PlannerBeliefManager::unregister_persona(const String &p_persona_id) {
	persona_registry.erase(p_persona_id);
}

Ref<PlannerPersona> PlannerBeliefManager::get_persona(const String &p_persona_id) const {
	if (persona_registry.has(p_persona_id)) {
		Variant persona_var = persona_registry[p_persona_id];
		if (persona_var.get_type() == Variant::OBJECT) {
			Ref<PlannerPersona> persona = persona_var;
			return persona;
		}
	}
	return Ref<PlannerPersona>();
}

bool PlannerBeliefManager::has_persona(const String &p_persona_id) const {
	return persona_registry.has(p_persona_id);
}

TypedArray<String> PlannerBeliefManager::get_all_persona_ids() const {
	TypedArray<String> ids;
	Array keys = persona_registry.keys();
	for (int i = 0; i < keys.size(); i++) {
		ids.push_back(keys[i]);
	}
	return ids;
}

Dictionary PlannerBeliefManager::get_beliefs_about(Ref<PlannerPersona> p_persona, const String &p_target_persona_id) const {
	if (p_persona.is_valid()) {
		return p_persona->get_beliefs_about(p_target_persona_id);
	}
	return Dictionary();
}

Dictionary PlannerBeliefManager::get_planner_state(const String &p_target_persona_id, const String &p_requesting_persona_id) const {
	// Information asymmetry: Personas cannot directly access each other's internal states
	return PlannerPersona::get_planner_state(p_target_persona_id, p_requesting_persona_id);
}

void PlannerBeliefManager::process_observation_for_persona(const String &p_persona_id, const Dictionary &p_observation) {
	Ref<PlannerPersona> persona = get_persona(p_persona_id);
	if (persona.is_valid()) {
		persona->process_observation(p_observation);
	}
}

void PlannerBeliefManager::process_communication_for_persona(const String &p_persona_id, const Dictionary &p_communication) {
	Ref<PlannerPersona> persona = get_persona(p_persona_id);
	if (persona.is_valid()) {
		persona->process_communication(p_communication);
	}
}

void PlannerBeliefManager::clear_all() {
	persona_registry.clear();
}
