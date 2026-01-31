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

// SPDX-FileCopyrightText: 2021 University of Maryland
// SPDX-License-Identifier: BSD-3-Clause-Clear
// Author: Dana Nau <nau@umd.edu>, July 7, 2021

#include "planner_belief_manager.h"

#include "core/os/os.h"

void PlannerBeliefManager::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_persona", "persona_id"), &PlannerBeliefManager::get_persona);
	ClassDB::bind_method(D_METHOD("has_persona", "persona_id"), &PlannerBeliefManager::has_persona);
	ClassDB::bind_method(D_METHOD("register_persona", "persona"), &PlannerBeliefManager::register_persona);
	ClassDB::bind_method(D_METHOD("process_observation_for_persona", "persona_id", "observation"), &PlannerBeliefManager::process_observation_for_persona);
	ClassDB::bind_method(D_METHOD("get_planner_state", "target_persona_id", "requester_persona_id"), &PlannerBeliefManager::get_planner_state);
}

PlannerBeliefManager::PlannerBeliefManager() {
}

PlannerBeliefManager::~PlannerBeliefManager() {
}

Ref<PlannerPersona> PlannerBeliefManager::get_persona(const String &p_persona_id) {
	if (!personas.has(p_persona_id)) {
		Ref<PlannerPersona> persona;
		persona.instantiate();
		persona->set_persona_id(p_persona_id);
		personas[p_persona_id] = persona;
	}
	return personas[p_persona_id];
}

bool PlannerBeliefManager::has_persona(const String &p_persona_id) const {
	return personas.has(p_persona_id);
}

void PlannerBeliefManager::register_persona(const Ref<PlannerPersona> &p_persona) {
	if (p_persona.is_valid()) {
		personas[p_persona->get_persona_id()] = p_persona;
	}
}

void PlannerBeliefManager::process_observation_for_persona(const String &p_persona_id, const Dictionary &p_observation) {
	Ref<PlannerPersona> persona = get_persona(p_persona_id);
	if (persona.is_valid()) {
		// Add observation as a belief or fact
		// Assuming observation is a dictionary with keys like "subject", "predicate", "value"
		String subject = p_observation.get("subject", "");
		String predicate = p_observation.get("predicate", "");
		Variant value = p_observation.get("value", Variant());
		bool is_fact = p_observation.get("is_fact", false);

		if (!subject.is_empty() && !predicate.is_empty()) {
			Dictionary belief_metadata;
			belief_metadata["type"] = is_fact ? "fact" : "belief";
			belief_metadata["source"] = is_fact ? "allocentric" : p_persona_id;
			belief_metadata["confidence"] = 1.0;
			belief_metadata["timestamp"] = OS::get_singleton()->get_ticks_msec();
			belief_metadata["accessibility"] = "public"; // or based on observation

			persona->get_belief_state()->set_predicate(subject, predicate, value, belief_metadata);
		}
	}
}

Dictionary PlannerBeliefManager::get_planner_state(const String &p_target_persona_id, const String &p_requester_persona_id) {
	Ref<PlannerPersona> persona = get_persona(p_target_persona_id);
	if (persona.is_valid()) {
		return persona->get_planner_state(p_target_persona_id, p_requester_persona_id);
	}
	return Dictionary();
}
