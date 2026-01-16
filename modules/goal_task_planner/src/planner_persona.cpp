/**************************************************************************/
/*  planner_persona.cpp                                                   */
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

#include "planner_persona.h"

#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/string/ustring.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"
#include "planner_time_range.h"

// Capability constants
const String PlannerPersona::CAPABILITY_MOVABLE = "movable";
const String PlannerPersona::CAPABILITY_INVENTORY = "inventory";
const String PlannerPersona::CAPABILITY_CRAFT = "craft";
const String PlannerPersona::CAPABILITY_MINE = "mine";
const String PlannerPersona::CAPABILITY_BUILD = "build";
const String PlannerPersona::CAPABILITY_INTERACT = "interact";
const String PlannerPersona::CAPABILITY_COMPUTE = "compute";
const String PlannerPersona::CAPABILITY_OPTIMIZE = "optimize";
const String PlannerPersona::CAPABILITY_PREDICT = "predict";
const String PlannerPersona::CAPABILITY_LEARN = "learn";
const String PlannerPersona::CAPABILITY_NAVIGATE = "navigate";

void PlannerPersona::_bind_methods() {
	ClassDB::bind_method(D_METHOD("initialize", "id", "name"), &PlannerPersona::initialize);
	ClassDB::bind_static_method("PlannerPersona", D_METHOD("create_basic", "id", "name"), &PlannerPersona::create_basic);
	ClassDB::bind_static_method("PlannerPersona", D_METHOD("create_human", "id", "name"), &PlannerPersona::create_human);
	ClassDB::bind_static_method("PlannerPersona", D_METHOD("create_ai", "id", "name"), &PlannerPersona::create_ai);
	ClassDB::bind_static_method("PlannerPersona", D_METHOD("create_hybrid", "id", "name"), &PlannerPersona::create_hybrid);

	ClassDB::bind_method(D_METHOD("get_persona_id"), &PlannerPersona::get_persona_id);
	ClassDB::bind_method(D_METHOD("get_persona_name"), &PlannerPersona::get_persona_name);
	ClassDB::bind_method(D_METHOD("get_identity_type"), &PlannerPersona::get_identity_type);
	ClassDB::bind_method(D_METHOD("is_active"), &PlannerPersona::is_active);
	ClassDB::bind_method(D_METHOD("get_metadata"), &PlannerPersona::get_metadata);
	ClassDB::bind_method(D_METHOD("get_beliefs_about_others"), &PlannerPersona::get_beliefs_about_others);
	ClassDB::bind_method(D_METHOD("get_belief_confidence"), &PlannerPersona::get_belief_confidence);
	ClassDB::bind_method(D_METHOD("get_capabilities"), &PlannerPersona::get_capabilities);

	ClassDB::bind_method(D_METHOD("set_persona_id", "id"), &PlannerPersona::set_persona_id);
	ClassDB::bind_method(D_METHOD("set_persona_name", "name"), &PlannerPersona::set_persona_name);
	ClassDB::bind_method(D_METHOD("set_active", "active"), &PlannerPersona::set_active);
	ClassDB::bind_method(D_METHOD("set_metadata", "metadata"), &PlannerPersona::set_metadata);

	ClassDB::bind_method(D_METHOD("has_capability", "capability"), &PlannerPersona::has_capability);
	ClassDB::bind_method(D_METHOD("add_capability", "capability"), &PlannerPersona::add_capability);
	ClassDB::bind_method(D_METHOD("remove_capability", "capability"), &PlannerPersona::remove_capability);
	ClassDB::bind_method(D_METHOD("enable_human_capabilities"), &PlannerPersona::enable_human_capabilities);
	ClassDB::bind_method(D_METHOD("enable_ai_capabilities"), &PlannerPersona::enable_ai_capabilities);

	ClassDB::bind_method(D_METHOD("get_beliefs_about", "target_persona_id"), &PlannerPersona::get_beliefs_about);
	ClassDB::bind_method(D_METHOD("set_belief_about", "target_persona_id", "belief_key", "belief_value", "confidence", "timestamp"), &PlannerPersona::set_belief_about, DEFVAL(1.0), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_belief_confidence_for", "target_persona_id", "belief_key"), &PlannerPersona::get_belief_confidence_for);
	ClassDB::bind_method(D_METHOD("get_belief_timestamp_for", "target_persona_id", "belief_key"), &PlannerPersona::get_belief_timestamp_for);
	ClassDB::bind_method(D_METHOD("update_belief_confidence", "target_persona_id", "belief_key", "confidence"), &PlannerPersona::update_belief_confidence);

	ClassDB::bind_method(D_METHOD("get_planner_state", "target_persona_id", "requesting_persona_id"), &PlannerPersona::get_planner_state);

	ClassDB::bind_method(D_METHOD("process_observation", "observation"), &PlannerPersona::process_observation);
	ClassDB::bind_method(D_METHOD("process_communication", "communication"), &PlannerPersona::process_communication);

	BIND_ENUM_CONSTANT(IDENTITY_BASIC);
	BIND_ENUM_CONSTANT(IDENTITY_HUMAN);
	BIND_ENUM_CONSTANT(IDENTITY_AI);
	BIND_ENUM_CONSTANT(IDENTITY_HUMAN_AND_AI);
}

VARIANT_ENUM_CAST(PlannerPersonaIdentity);

PlannerPersona::PlannerPersona() {
	persona_id = "";
	persona_name = "";
	identity_type = PlannerPersonaIdentity::IDENTITY_BASIC;
	active = true;
	metadata = Dictionary();
	belief_state = Ref<PlannerState>(memnew(PlannerState));
	capabilities = TypedArray<String>();
}

PlannerPersona::~PlannerPersona() {
}

void PlannerPersona::initialize(const String &p_id, const String &p_name) {
	persona_id = p_id;
	persona_name = p_name;
	active = true;
	_update_identity_type();
}

Ref<PlannerPersona> PlannerPersona::create_basic(const String &p_id, const String &p_name) {
	Ref<PlannerPersona> persona = memnew(PlannerPersona);
	persona->initialize(p_id, p_name);
	return persona;
}

Ref<PlannerPersona> PlannerPersona::create_human(const String &p_id, const String &p_name) {
	Ref<PlannerPersona> persona = memnew(PlannerPersona);
	persona->initialize(p_id, p_name);
	persona->enable_human_capabilities();
	return persona;
}

Ref<PlannerPersona> PlannerPersona::create_ai(const String &p_id, const String &p_name) {
	Ref<PlannerPersona> persona = memnew(PlannerPersona);
	persona->initialize(p_id, p_name);
	persona->enable_ai_capabilities();
	return persona;
}

Ref<PlannerPersona> PlannerPersona::create_hybrid(const String &p_id, const String &p_name) {
	Ref<PlannerPersona> persona = memnew(PlannerPersona);
	persona->initialize(p_id, p_name);
	persona->enable_human_capabilities();
	persona->enable_ai_capabilities();
	return persona;
}

void PlannerPersona::_update_identity_type() {
	bool has_human = has_capability(CAPABILITY_INVENTORY) || has_capability(CAPABILITY_CRAFT) ||
			has_capability(CAPABILITY_MINE) || has_capability(CAPABILITY_BUILD) || has_capability(CAPABILITY_INTERACT);
	bool has_ai = has_capability(CAPABILITY_COMPUTE) || has_capability(CAPABILITY_OPTIMIZE) ||
			has_capability(CAPABILITY_PREDICT) || has_capability(CAPABILITY_LEARN) || has_capability(CAPABILITY_NAVIGATE);

	if (has_human && has_ai) {
		identity_type = PlannerPersonaIdentity::IDENTITY_HUMAN_AND_AI;
	} else if (has_human) {
		identity_type = PlannerPersonaIdentity::IDENTITY_HUMAN;
	} else if (has_ai) {
		identity_type = PlannerPersonaIdentity::IDENTITY_AI;
	} else {
		identity_type = PlannerPersonaIdentity::IDENTITY_BASIC;
	}
}

bool PlannerPersona::has_capability(const String &p_capability) const {
	for (int i = 0; i < capabilities.size(); i++) {
		if (capabilities[i] == p_capability) {
			return true;
		}
	}
	return false;
}

void PlannerPersona::add_capability(const String &p_capability) {
	if (!has_capability(p_capability)) {
		capabilities.push_back(p_capability);
		_update_identity_type();
	}
}

void PlannerPersona::remove_capability(const String &p_capability) {
	for (int i = 0; i < capabilities.size(); i++) {
		if (capabilities[i] == p_capability) {
			capabilities.remove_at(i);
			_update_identity_type();
			return;
		}
	}
}

void PlannerPersona::enable_human_capabilities() {
	add_capability(CAPABILITY_MOVABLE);
	add_capability(CAPABILITY_INVENTORY);
	add_capability(CAPABILITY_CRAFT);
	add_capability(CAPABILITY_MINE);
	add_capability(CAPABILITY_BUILD);
	add_capability(CAPABILITY_INTERACT);
}

void PlannerPersona::enable_ai_capabilities() {
	add_capability(CAPABILITY_MOVABLE);
	add_capability(CAPABILITY_COMPUTE);
	add_capability(CAPABILITY_OPTIMIZE);
	add_capability(CAPABILITY_PREDICT);
	add_capability(CAPABILITY_LEARN);
	add_capability(CAPABILITY_NAVIGATE);
}

Dictionary PlannerPersona::get_beliefs_about(const String &p_target_persona_id) const {
	return belief_state->get_beliefs_about(persona_id, p_target_persona_id);
}

void PlannerPersona::set_belief_about(const String &p_target_persona_id, const String &p_belief_key, const Variant &p_belief_value, double p_confidence, int64_t p_timestamp) {
	Dictionary belief_metadata;
	belief_metadata["confidence"] = p_confidence;
	belief_metadata["type"] = "belief";
	belief_metadata["source"] = persona_id;
	belief_metadata["timestamp"] = p_timestamp == 0 ? OS::get_singleton()->get_ticks_msec() : p_timestamp;
	belief_metadata["accessibility"] = "private";
	belief_state->set_belief_about(persona_id, p_target_persona_id, p_belief_key, p_belief_value, belief_metadata);
}

double PlannerPersona::get_belief_confidence_for(const String &p_target_persona_id, const String &p_belief_key) const {
	return belief_state->get_belief_confidence(persona_id, p_target_persona_id, p_belief_key);
}

int64_t PlannerPersona::get_belief_timestamp_for(const String &p_target_persona_id, const String &p_belief_key) const {
	return belief_state->get_belief_timestamp(persona_id, p_target_persona_id, p_belief_key);
}

void PlannerPersona::update_belief_confidence(const String &p_target_persona_id, const String &p_belief_key, double p_confidence) {
	belief_state->update_belief_confidence(persona_id, p_target_persona_id, p_belief_key, p_confidence);
}

Dictionary PlannerPersona::get_planner_state(const String &p_target_persona_id, const String &p_requesting_persona_id) const {
	// Information asymmetry: Only allow access if requesting persona is the target
	if (p_requesting_persona_id != p_target_persona_id || p_target_persona_id != persona_id) {
		Dictionary error_dict;
		error_dict["error"] = "hidden";
		return error_dict;
	}

	// Return triples as Array of Dictionaries
	Array result;
	for (const auto &triple : belief_state->get_triples()) {
		Dictionary triple_dict;
		triple_dict["subject"] = triple.subject;
		triple_dict["predicate"] = triple.predicate;
		triple_dict["object"] = triple.object;
		triple_dict["metadata"] = triple.metadata;
		result.push_back(triple_dict);
	}

	Dictionary state_dict;
	state_dict["triples"] = result;
	return state_dict;
}

void PlannerPersona::process_observation(const Dictionary &p_observation) {
	// Process an observation to update beliefs
	// Expected format: {"entity": String, "command": String, "confidence": float, "time": int64_t, ...}
	if (!p_observation.has("entity")) {
		return;
	}

	String entity_id = p_observation.get("entity", "");
	String command = p_observation.get("command", "");
	double confidence = p_observation.get("confidence", 1.0);

	if (entity_id.is_empty()) {
		return;
	}

	// Extract timestamp from observation or use current time
	int64_t timestamp = 0;
	if (p_observation.has("time")) {
		Variant time_var = p_observation["time"];
		if (time_var.get_type() == Variant::INT || time_var.get_type() == Variant::FLOAT) {
			// If time is provided as seconds (float) or microseconds (int), convert appropriately
			if (time_var.get_type() == Variant::FLOAT) {
				timestamp = PlannerTimeRange::unix_time_to_microseconds(time_var);
			} else {
				timestamp = time_var;
			}
		}
	}
	if (timestamp == 0) {
		timestamp = PlannerTimeRange::now_microseconds();
	}

	// Update belief about observed entity with temporal metadata
	String belief_key = vformat("observed_%s", command);
	set_belief_about(entity_id, belief_key, p_observation, confidence, timestamp);

	// Increase confidence with consistent observations
	double current_confidence = get_belief_confidence_for(entity_id, belief_key);
	double new_confidence = MIN(1.0, current_confidence + (confidence * 0.1));
	update_belief_confidence(entity_id, belief_key, new_confidence);

	// Only form location preference beliefs if we observe them at a location AND we're at that same location
	// This requires both "location" (where entity was observed) and "observer_location" (where we are) fields
	if (p_observation.has("location") && p_observation.has("observer_location")) {
		String entity_location = p_observation.get("location", "");
		String observer_location = p_observation.get("observer_location", "");

		// Only form belief if we're at the same location as the observed entity
		if (entity_location == observer_location && !entity_location.is_empty()) {
			String likes_key = vformat("likes_%s", entity_location);
			set_belief_about(entity_id, likes_key, true, confidence, timestamp);
			update_belief_confidence(entity_id, likes_key, confidence);
		}
	}
}

void PlannerPersona::process_communication(const Dictionary &p_communication) {
	// Process communication to update beliefs
	// Expected format: {"sender"/"from": String, "content"/"message": String, "type"/"topic": String, "time": int64_t, ...}
	String sender_id;
	if (p_communication.has("sender")) {
		sender_id = p_communication.get("sender", "");
	} else if (p_communication.has("from")) {
		sender_id = p_communication.get("from", "");
	} else {
		return;
	}

	String content;
	if (p_communication.has("content")) {
		content = p_communication.get("content", "");
	} else if (p_communication.has("message")) {
		content = p_communication.get("message", "");
	}

	String comm_type = "general";
	if (p_communication.has("type")) {
		comm_type = p_communication.get("type", "general");
	} else if (p_communication.has("topic")) {
		comm_type = p_communication.get("topic", "general");
	}

	if (sender_id.is_empty()) {
		return;
	}

	// Extract timestamp from communication or use current time
	int64_t timestamp = 0;
	if (p_communication.has("time")) {
		Variant time_var = p_communication["time"];
		if (time_var.get_type() == Variant::INT || time_var.get_type() == Variant::FLOAT) {
			// If time is provided as seconds (float) or microseconds (int), convert appropriately
			if (time_var.get_type() == Variant::FLOAT) {
				timestamp = PlannerTimeRange::unix_time_to_microseconds(time_var);
			} else {
				timestamp = time_var;
			}
		}
	}
	if (timestamp == 0) {
		timestamp = PlannerTimeRange::now_microseconds();
	}

	// Update belief about sender based on communication with temporal metadata
	String belief_key = vformat("communication_%s", comm_type);
	double confidence = p_communication.get("confidence", 0.8);
	set_belief_about(sender_id, belief_key, p_communication, confidence, timestamp);

	// Communication alone does not form location preference beliefs
	// Location preferences are only formed through direct observation at that location
}
