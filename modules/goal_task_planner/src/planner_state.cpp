/**************************************************************************/
/*  planner_state.cpp                                                     */
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

#include "planner_state.h"

#include "core/error/error_macros.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "core/templates/vector.h"
#include "core/variant/variant.h"

void PlannerState::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_predicate", "subject", "predicate"), &PlannerState::get_predicate);
	ClassDB::bind_method(D_METHOD("set_predicate", "subject", "predicate", "value", "metadata"), &PlannerState::set_predicate, DEFVAL(Dictionary()));
	ClassDB::bind_method(D_METHOD("get_triples_as_array"), &PlannerState::get_triples_as_array);
	ClassDB::bind_method(D_METHOD("get_subject_predicate_list"), &PlannerState::get_subject_predicate_list);

	ClassDB::bind_method(D_METHOD("has_subject_variable", "variable"), &PlannerState::has_subject_variable);
	ClassDB::bind_method(D_METHOD("has_predicate", "subject", "predicate"), &PlannerState::has_predicate);

	// Entity capabilities methods
	ClassDB::bind_method(D_METHOD("get_entity_capability", "entity_id", "capability"), &PlannerState::get_entity_capability);
	ClassDB::bind_method(D_METHOD("set_entity_capability", "entity_id", "capability", "value"), &PlannerState::set_entity_capability);
	ClassDB::bind_method(D_METHOD("has_entity", "entity_id"), &PlannerState::has_entity);
	ClassDB::bind_method(D_METHOD("get_entity_capabilities", "entity_id"), &PlannerState::get_entity_capabilities);
	ClassDB::bind_method(D_METHOD("get_all_entity_capabilities"), &PlannerState::get_all_entity_capabilities);
	ClassDB::bind_method(D_METHOD("get_all_entities"), &PlannerState::get_all_entities);

	// Terrain facts (allocentric)
	ClassDB::bind_method(D_METHOD("set_terrain_fact", "location", "fact_key", "value"), &PlannerState::set_terrain_fact);
	ClassDB::bind_method(D_METHOD("get_terrain_fact", "location", "fact_key"), &PlannerState::get_terrain_fact);
	ClassDB::bind_method(D_METHOD("has_terrain_fact", "location", "fact_key"), &PlannerState::has_terrain_fact);
	ClassDB::bind_method(D_METHOD("get_all_terrain_facts"), &PlannerState::get_all_terrain_facts);

	// Shared objects
	ClassDB::bind_method(D_METHOD("add_shared_object", "object_id", "object_data"), &PlannerState::add_shared_object);
	ClassDB::bind_method(D_METHOD("remove_shared_object", "object_id"), &PlannerState::remove_shared_object);
	ClassDB::bind_method(D_METHOD("get_shared_object", "object_id"), &PlannerState::get_shared_object);
	ClassDB::bind_method(D_METHOD("has_shared_object", "object_id"), &PlannerState::has_shared_object);
	ClassDB::bind_method(D_METHOD("get_all_shared_object_ids"), &PlannerState::get_all_shared_object_ids);
	ClassDB::bind_method(D_METHOD("get_all_shared_objects"), &PlannerState::get_all_shared_objects);

	// Public events
	ClassDB::bind_method(D_METHOD("add_public_event", "event_id", "event_data"), &PlannerState::add_public_event);
	ClassDB::bind_method(D_METHOD("remove_public_event", "event_id"), &PlannerState::remove_public_event);
	ClassDB::bind_method(D_METHOD("get_public_event", "event_id"), &PlannerState::get_public_event);
	ClassDB::bind_method(D_METHOD("has_public_event", "event_id"), &PlannerState::has_public_event);
	ClassDB::bind_method(D_METHOD("get_all_public_event_ids"), &PlannerState::get_all_public_event_ids);
	ClassDB::bind_method(D_METHOD("get_all_public_events"), &PlannerState::get_all_public_events);

	// Entity positions
	ClassDB::bind_method(D_METHOD("set_entity_position", "entity_id", "position"), &PlannerState::set_entity_position);
	ClassDB::bind_method(D_METHOD("get_entity_position", "entity_id"), &PlannerState::get_entity_position);
	ClassDB::bind_method(D_METHOD("has_entity_position", "entity_id"), &PlannerState::has_entity_position);
	ClassDB::bind_method(D_METHOD("get_all_entity_positions"), &PlannerState::get_all_entity_positions);

	// Public entity capabilities
	ClassDB::bind_method(D_METHOD("set_entity_capability_public", "entity_id", "capability", "value"), &PlannerState::set_entity_capability_public);
	ClassDB::bind_method(D_METHOD("get_entity_capability_public", "entity_id", "capability"), &PlannerState::get_entity_capability_public);
	ClassDB::bind_method(D_METHOD("has_entity_capability_public", "entity_id", "capability"), &PlannerState::has_entity_capability_public);
	ClassDB::bind_method(D_METHOD("get_all_entity_capabilities_public"), &PlannerState::get_all_entity_capabilities_public);

	// Observation methods
	ClassDB::bind_method(D_METHOD("observe_terrain", "location"), &PlannerState::observe_terrain);
	ClassDB::bind_method(D_METHOD("observe_shared_objects", "location"), &PlannerState::observe_shared_objects);
	ClassDB::bind_method(D_METHOD("observe_public_events"), &PlannerState::observe_public_events);
	ClassDB::bind_method(D_METHOD("observe_entity_positions"), &PlannerState::observe_entity_positions);
	ClassDB::bind_method(D_METHOD("observe_entity_capabilities"), &PlannerState::observe_entity_capabilities);

	// Clear methods
	ClassDB::bind_method(D_METHOD("clear_allocentric_facts"), &PlannerState::clear_allocentric_facts);

	// Belief management with metadata
	ClassDB::bind_method(D_METHOD("get_beliefs_about", "persona_id", "target"), &PlannerState::get_beliefs_about);
	ClassDB::bind_method(D_METHOD("set_belief_about", "persona_id", "target", "predicate", "value", "metadata"), &PlannerState::set_belief_about, DEFVAL(Dictionary()));
	ClassDB::bind_method(D_METHOD("get_belief_confidence", "persona_id", "target", "predicate"), &PlannerState::get_belief_confidence);
	ClassDB::bind_method(D_METHOD("get_belief_timestamp", "persona_id", "target", "predicate"), &PlannerState::get_belief_timestamp);
	ClassDB::bind_method(D_METHOD("update_belief_confidence", "persona_id", "target", "predicate", "confidence"), &PlannerState::update_belief_confidence);
}

Variant PlannerState::get_predicate(const String &p_subject, const String &p_predicate) const {
	for (const auto &triple : triples) {
		if (triple.subject == p_subject && triple.predicate == p_predicate) {
			return triple.object;
		}
	}
	return Variant();
}

void PlannerState::set_predicate(const String &p_subject, const String &p_predicate, const Variant &p_value, const Dictionary &p_metadata) {
	Dictionary predicate_metadata = p_metadata;
	if (predicate_metadata.is_empty()) {
		predicate_metadata["type"] = "state";
		predicate_metadata["confidence"] = 1.0;
		predicate_metadata["timestamp"] = OS::get_singleton()->get_ticks_usec();
		predicate_metadata["accessibility"] = "private";
		predicate_metadata["source"] = "planner";
	}

	// Check if triple exists, update it
	for (auto &triple : triples) {
		if (triple.subject == p_subject && triple.predicate == p_predicate) {
			triple.object = p_value;
			triple.metadata = predicate_metadata;
			return;
		}
	}
	// Add new triple
	KnowledgeTriple new_triple;
	new_triple.subject = p_subject;
	new_triple.predicate = p_predicate;
	new_triple.object = p_value;
	new_triple.metadata = predicate_metadata;
	triples.push_back(new_triple);
}

TypedArray<Dictionary> PlannerState::get_triples_as_array() const {
	TypedArray<Dictionary> result;
	for (const auto &triple : triples) {
		Dictionary dict;
		dict["subject"] = triple.subject;
		dict["predicate"] = triple.predicate;
		dict["object"] = triple.object;
		dict["metadata"] = triple.metadata;
		result.push_back(dict);
	}
	return result;
}

TypedArray<String> PlannerState::get_subject_predicate_list() const {
	TypedArray<String> subjects;
	for (const auto &triple : triples) {
		if (!subjects.has(triple.subject)) {
			subjects.push_back(triple.subject);
		}
	}
	return subjects;
}

bool PlannerState::has_subject_variable(const String &p_variable) const {
	for (const auto &triple : triples) {
		if (triple.subject == p_variable) {
			return true;
		}
	}
	return false;
}

bool PlannerState::has_predicate(const String &p_subject, const String &p_predicate) const {
	for (const auto &triple : triples) {
		if (triple.subject == p_subject && triple.predicate == p_predicate) {
			return true;
		}
	}
	return false;
}

Variant PlannerState::get_entity_capability(const String &p_entity_id, const String &p_capability) const {
	String subject = "entity_" + p_entity_id;
	String predicate = "capability_" + p_capability;
	for (const auto &triple : triples) {
		if (triple.subject == subject && triple.predicate == predicate && triple.metadata["type"] == "state") {
			return triple.object;
		}
	}
	return Variant();
}

void PlannerState::set_entity_capability(const String &p_entity_id, const String &p_capability, const Variant &p_value) {
	String subject = "entity_" + p_entity_id;
	String predicate = "capability_" + p_capability;
	for (auto &triple : triples) {
		if (triple.subject == subject && triple.predicate == predicate && triple.metadata["type"] == "state") {
			triple.object = p_value;
			triple.metadata["timestamp"] = OS::get_singleton()->get_ticks_usec();
			return;
		}
	}
	KnowledgeTriple new_triple;
	new_triple.subject = subject;
	new_triple.predicate = predicate;
	new_triple.object = p_value;
	new_triple.metadata["type"] = "state";
	new_triple.metadata["timestamp"] = OS::get_singleton()->get_ticks_usec();
	triples.push_back(new_triple);
}

bool PlannerState::has_entity(const String &p_entity_id) const {
	String subject = "entity_" + p_entity_id;
	for (const auto &triple : triples) {
		if (triple.subject == subject) {
			return true;
		}
	}
	return false;
}

Array PlannerState::get_all_entities() const {
	Array entities;
	for (const auto &triple : triples) {
		if (triple.subject.begins_with("entity_") && triple.metadata["type"] == "state") {
			String entity_id = triple.subject.substr(7); // Remove "entity_"
			if (!entities.has(entity_id)) {
				entities.push_back(entity_id);
			}
		}
	}
	return entities;
}

Dictionary PlannerState::get_entity_capabilities(const String &p_entity_id) const {
	Dictionary result;
	String subject = "entity_" + p_entity_id;
	for (const auto &triple : triples) {
		if (triple.subject == subject && triple.metadata["type"] == "state") {
			String capability = triple.predicate.substr(11); // Remove "capability_"
			result[capability] = triple.object;
		}
	}
	return result;
}

Dictionary PlannerState::get_all_entity_capabilities() const {
	Dictionary result;
	for (const auto &triple : triples) {
		if (triple.subject.begins_with("entity_") && triple.metadata["type"] == "state") {
			String entity_id = triple.subject.substr(7);
			if (!result.has(entity_id)) {
				result[entity_id] = Dictionary();
			}
			String capability = triple.predicate.substr(11);
			Dictionary entity_caps = result[entity_id];
			entity_caps[capability] = triple.object;
			result[entity_id] = entity_caps;
		}
	}
	return result;
}

// Allocentric facts implementations

void PlannerState::set_terrain_fact(const String &p_location, const String &p_fact_key, const Variant &p_value) {
	String subject = "terrain_" + p_location;
	Dictionary terrain_metadata;
	terrain_metadata["type"] = "fact";
	terrain_metadata["source"] = "allocentric";
	terrain_metadata["accessibility"] = "public";
	terrain_metadata["timestamp"] = OS::get_singleton()->get_ticks_usec();
	set_predicate(subject, p_fact_key, p_value, terrain_metadata);
}

Variant PlannerState::get_terrain_fact(const String &p_location, const String &p_fact_key) const {
	String subject = "terrain_" + p_location;
	return get_predicate(subject, p_fact_key);
}

bool PlannerState::has_terrain_fact(const String &p_location, const String &p_fact_key) const {
	String subject = "terrain_" + p_location;
	return has_predicate(subject, p_fact_key);
}

Dictionary PlannerState::get_all_terrain_facts() const {
	Dictionary result;
	for (const auto &triple : triples) {
		if (triple.metadata["type"] == "fact" && triple.subject.begins_with("terrain_")) {
			String location = triple.subject.substr(8); // Remove "terrain_"
			if (!result.has(location)) {
				result[location] = Dictionary();
			}
			Dictionary loc_data = result[location];
			loc_data[triple.predicate] = triple.object;
			result[location] = loc_data;
		}
	}
	return result;
}

void PlannerState::add_shared_object(const String &p_object_id, const Dictionary &p_object_data) {
	String subject = "shared_object_" + p_object_id;
	String predicate = "data";
	// Store the entire dictionary as object
	for (auto &triple : triples) {
		if (triple.subject == subject && triple.predicate == predicate && triple.metadata["type"] == "fact") {
			triple.object = p_object_data;
			triple.metadata["timestamp"] = OS::get_singleton()->get_ticks_usec();
			return;
		}
	}
	KnowledgeTriple new_triple;
	new_triple.subject = subject;
	new_triple.predicate = predicate;
	new_triple.object = p_object_data;
	new_triple.metadata["type"] = "fact";
	new_triple.metadata["source"] = "allocentric";
	new_triple.metadata["confidence"] = 1.0;
	new_triple.metadata["timestamp"] = OS::get_singleton()->get_ticks_usec();
	new_triple.metadata["accessibility"] = "public";
	triples.push_back(new_triple);
}

void PlannerState::remove_shared_object(const String &p_object_id) {
	String subject = "shared_object_" + p_object_id;
	Vector<KnowledgeTriple> new_triples;
	for (const auto &t : triples) {
		if (!(t.subject == subject && t.metadata["type"] == "fact")) {
			new_triples.push_back(t);
		}
	}
	triples = new_triples;
}

Dictionary PlannerState::get_shared_object(const String &p_object_id) const {
	String subject = "shared_object_" + p_object_id;
	return get_predicate(subject, "data");
}

bool PlannerState::has_shared_object(const String &p_object_id) const {
	String subject = "shared_object_" + p_object_id;
	return has_predicate(subject, "data");
}

Array PlannerState::get_all_shared_object_ids() const {
	Array ids;
	for (const auto &triple : triples) {
		if (triple.subject.begins_with("shared_object_") && triple.predicate == "data" && triple.metadata["type"] == "fact") {
			String id = triple.subject.substr(14); // Remove "shared_object_"
			ids.push_back(id);
		}
	}
	return ids;
}

Dictionary PlannerState::get_all_shared_objects() const {
	Dictionary result;
	for (const auto &triple : triples) {
		if (triple.subject.begins_with("shared_object_") && triple.predicate == "data" && triple.metadata["type"] == "fact") {
			String id = triple.subject.substr(14);
			result[id] = triple.object;
		}
	}
	return result;
}

void PlannerState::add_public_event(const String &p_event_id, const Dictionary &p_event_data) {
	String subject = "public_event_" + p_event_id;
	String predicate = "data";
	for (auto &triple : triples) {
		if (triple.subject == subject && triple.predicate == predicate && triple.metadata["type"] == "fact") {
			triple.object = p_event_data;
			triple.metadata["timestamp"] = OS::get_singleton()->get_ticks_usec();
			return;
		}
	}
	KnowledgeTriple new_triple;
	new_triple.subject = subject;
	new_triple.predicate = predicate;
	new_triple.object = p_event_data;
	new_triple.metadata["type"] = "fact";
	new_triple.metadata["source"] = "allocentric";
	new_triple.metadata["confidence"] = 1.0;
	new_triple.metadata["timestamp"] = OS::get_singleton()->get_ticks_usec();
	new_triple.metadata["accessibility"] = "public";
	triples.push_back(new_triple);
}

void PlannerState::remove_public_event(const String &p_event_id) {
	String subject = "public_event_" + p_event_id;
	Vector<KnowledgeTriple> new_triples;
	for (const auto &t : triples) {
		if (!(t.subject == subject && t.metadata["type"] == "fact")) {
			new_triples.push_back(t);
		}
	}
	triples = new_triples;
}

Dictionary PlannerState::get_public_event(const String &p_event_id) const {
	String subject = "public_event_" + p_event_id;
	return get_predicate(subject, "data");
}

bool PlannerState::has_public_event(const String &p_event_id) const {
	String subject = "public_event_" + p_event_id;
	return has_predicate(subject, "data");
}

Array PlannerState::get_all_public_event_ids() const {
	Array ids;
	for (const auto &triple : triples) {
		if (triple.subject.begins_with("public_event_") && triple.predicate == "data" && triple.metadata["type"] == "fact") {
			String id = triple.subject.substr(13); // Remove "public_event_"
			ids.push_back(id);
		}
	}
	return ids;
}

Dictionary PlannerState::get_all_public_events() const {
	Dictionary result;
	for (const auto &triple : triples) {
		if (triple.subject.begins_with("public_event_") && triple.predicate == "data" && triple.metadata["type"] == "fact") {
			String id = triple.subject.substr(13);
			result[id] = triple.object;
		}
	}
	return result;
}

void PlannerState::set_entity_position(const String &p_entity_id, const Variant &p_position) {
	String subject = "entity_" + p_entity_id;
	String predicate = "position";
	for (auto &triple : triples) {
		if (triple.subject == subject && triple.predicate == predicate && triple.metadata["type"] == "fact") {
			triple.object = p_position;
			triple.metadata["timestamp"] = OS::get_singleton()->get_ticks_usec();
			return;
		}
	}
	KnowledgeTriple new_triple;
	new_triple.subject = subject;
	new_triple.predicate = predicate;
	new_triple.object = p_position;
	new_triple.metadata["type"] = "fact";
	new_triple.metadata["source"] = "allocentric";
	new_triple.metadata["confidence"] = 1.0;
	new_triple.metadata["timestamp"] = OS::get_singleton()->get_ticks_usec();
	new_triple.metadata["accessibility"] = "public";
	triples.push_back(new_triple);
}

Variant PlannerState::get_entity_position(const String &p_entity_id) const {
	String subject = "entity_" + p_entity_id;
	return get_predicate(subject, "position");
}

bool PlannerState::has_entity_position(const String &p_entity_id) const {
	String subject = "entity_" + p_entity_id;
	return has_predicate(subject, "position");
}

Dictionary PlannerState::get_all_entity_positions() const {
	Dictionary result;
	for (const auto &triple : triples) {
		if (triple.predicate == "position" && triple.metadata["type"] == "fact") {
			String entity_id = triple.subject.substr(7); // Remove "entity_"
			result[entity_id] = triple.object;
		}
	}
	return result;
}

void PlannerState::set_entity_capability_public(const String &p_entity_id, const String &p_capability, const Variant &p_value) {
	String subject = "entity_" + p_entity_id;
	String predicate = "capability_" + p_capability;
	for (auto &triple : triples) {
		if (triple.subject == subject && triple.predicate == predicate && triple.metadata["type"] == "fact") {
			triple.object = p_value;
			triple.metadata["timestamp"] = OS::get_singleton()->get_ticks_usec();
			return;
		}
	}
	KnowledgeTriple new_triple;
	new_triple.subject = subject;
	new_triple.predicate = predicate;
	new_triple.object = p_value;
	new_triple.metadata["type"] = "fact";
	new_triple.metadata["source"] = "allocentric";
	new_triple.metadata["confidence"] = 1.0;
	new_triple.metadata["timestamp"] = OS::get_singleton()->get_ticks_usec();
	new_triple.metadata["accessibility"] = "public";
	triples.push_back(new_triple);
}

Variant PlannerState::get_entity_capability_public(const String &p_entity_id, const String &p_capability) const {
	String subject = "entity_" + p_entity_id;
	String predicate = "capability_" + p_capability;
	return get_predicate(subject, predicate);
}

bool PlannerState::has_entity_capability_public(const String &p_entity_id, const String &p_capability) const {
	String subject = "entity_" + p_entity_id;
	String predicate = "capability_" + p_capability;
	return has_predicate(subject, predicate);
}

Dictionary PlannerState::get_all_entity_capabilities_public() const {
	Dictionary result;
	for (const auto &triple : triples) {
		if (triple.subject.begins_with("entity_") && triple.predicate.begins_with("capability_") && triple.metadata["type"] == "fact") {
			String entity_id = triple.subject.substr(7);
			String capability = triple.predicate.substr(11); // Remove "capability_"
			if (!result.has(entity_id)) {
				result[entity_id] = Dictionary();
			}
			Dictionary cap_data = result[entity_id];
			cap_data[capability] = triple.object;
			result[entity_id] = cap_data;
		}
	}
	return result;
}

Dictionary PlannerState::observe_terrain(const String &p_location) const {
	Dictionary result;
	String prefix = "terrain_" + p_location;
	for (const auto &triple : triples) {
		if (triple.subject == prefix && triple.metadata["type"] == "fact" && triple.metadata.get("accessibility", "") == "public") {
			result[triple.predicate] = triple.object;
		}
	}
	return result;
}

Dictionary PlannerState::observe_shared_objects(const String &p_location) const {
	Dictionary result;
	for (const auto &triple : triples) {
		if (triple.subject.begins_with("shared_object_") && triple.predicate == "data" && triple.metadata["type"] == "fact" && triple.metadata.get("accessibility", "") == "public") {
			String id = triple.subject.substr(14);
			result[id] = triple.object;
		}
	}
	return result;
}

Dictionary PlannerState::observe_public_events() const {
	return get_all_public_events();
}

Dictionary PlannerState::observe_entity_positions() const {
	return get_all_entity_positions();
}

Dictionary PlannerState::observe_entity_capabilities() const {
	return get_all_entity_capabilities_public();
}

void PlannerState::clear_allocentric_facts() {
	Vector<KnowledgeTriple> new_triples;
	for (const auto &t : triples) {
		if (t.metadata["type"] != "fact") {
			new_triples.push_back(t);
		}
	}
	triples = new_triples;
}

// Belief methods implementations

int64_t PlannerState::get_belief_timestamp(const String &p_persona_id, const String &p_target, const String &p_predicate) const {
	for (const auto &triple : triples) {
		if (triple.subject == p_target && triple.predicate == p_predicate && triple.metadata.get("source", "") == p_persona_id && triple.metadata.get("type", "") == "belief") {
			return triple.metadata.get("timestamp", 0);
		}
	}
	return 0;
}

Dictionary PlannerState::get_beliefs_about(const String &p_persona_id, const String &p_target) const {
	Dictionary dict;
	for (const auto &triple : triples) {
		if (triple.subject == p_target && triple.metadata.get("source", "") == p_persona_id && triple.metadata.get("type", "") == "belief") {
			dict[triple.predicate] = triple.object;
		}
	}
	return dict;
}

void PlannerState::set_belief_about(const String &p_persona_id, const String &p_target, const String &p_predicate, const Variant &p_value, const Dictionary &p_metadata) {
	Dictionary belief_metadata = p_metadata;
	if (belief_metadata.is_empty()) {
		belief_metadata["type"] = "belief";
		belief_metadata["source"] = p_persona_id;
		belief_metadata["confidence"] = 1.0;
		belief_metadata["timestamp"] = OS::get_singleton()->get_ticks_usec();
		belief_metadata["accessibility"] = "private";
	}
	set_predicate(p_target, p_predicate, p_value, belief_metadata);
}

double PlannerState::get_belief_confidence(const String &p_persona_id, const String &p_target, const String &p_predicate) const {
	for (const auto &triple : triples) {
		if (triple.subject == p_target && triple.predicate == p_predicate && triple.metadata.get("source", "") == p_persona_id && triple.metadata.get("type", "") == "belief") {
			return triple.metadata.get("confidence", 1.0);
		}
	}
	return 1.0;
}

void PlannerState::update_belief_confidence(const String &p_persona_id, const String &p_target, const String &p_predicate, double p_confidence) {
	for (auto &triple : triples) {
		if (triple.subject == p_target && triple.predicate == p_predicate && triple.metadata.get("source", "") == p_persona_id && triple.metadata.get("type", "") == "belief") {
			triple.metadata["confidence"] = p_confidence;
			break;
		}
	}
}

PlannerState::PlannerState() {}

PlannerState::~PlannerState() {}
