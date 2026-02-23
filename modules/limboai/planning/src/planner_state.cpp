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

#include "blackboard/blackboard.h"
#include "core/error/error_macros.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "core/templates/vector.h"
#include "core/variant/variant.h"

// Blackboard key names for planner state (avoid string literals in hot paths)
#define _BB_PLANNER_METADATA StringName("planner_metadata")
#define _BB_ENTITY_CAPABILITIES StringName("entity_capabilities")
#define _BB_ENTITY_CAPABILITIES_PUBLIC StringName("entity_capabilities_public")
#define _BB_TERRAIN_FACTS StringName("terrain_facts")
#define _BB_SHARED_OBJECTS StringName("shared_objects")
#define _BB_PUBLIC_EVENTS StringName("public_events")
#define _BB_ENTITY_POSITIONS StringName("entity_positions")
#define _BB_BELIEFS StringName("beliefs")
#define _BB_BELIEFS_METADATA StringName("beliefs_metadata")

namespace {

struct KnowledgeTriple {
	String predicate;
	String subject;
	Variant object;
	Dictionary metadata;
};

Vector<KnowledgeTriple> collect_triples(const PlannerState *state) {
	Vector<KnowledgeTriple> result;
	Ref<Blackboard> blackboard = state->get_blackboard();
	if (!blackboard.is_valid()) {
		return result;
	}
	Dictionary all = blackboard->get_vars_as_dict();
	Dictionary meta_all = blackboard->get_var(_BB_PLANNER_METADATA, Dictionary(), false);
	Array keys = all.keys();
	for (int i = 0; i < keys.size(); i++) {
		StringName k = keys[i];
		if (k == _BB_PLANNER_METADATA || k == _BB_BELIEFS || k == _BB_BELIEFS_METADATA ||
				k == _BB_ENTITY_CAPABILITIES || k == _BB_ENTITY_CAPABILITIES_PUBLIC ||
				k == _BB_TERRAIN_FACTS || k == _BB_SHARED_OBJECTS || k == _BB_PUBLIC_EVENTS || k == _BB_ENTITY_POSITIONS) {
			continue;
		}
		Variant v = all[k];
		if (v.get_type() != Variant::DICTIONARY) {
			continue;
		}
		Dictionary dict = v;
		Dictionary pred_meta = meta_all.has(k) ? Dictionary(meta_all[k]) : Dictionary();
		Array subj_keys = dict.keys();
		for (int j = 0; j < subj_keys.size(); j++) {
			Variant subj = subj_keys[j];
			String subj_str = subj;
			KnowledgeTriple t;
			t.predicate = String(k);
			t.subject = subj_str;
			t.object = dict[subj];
			t.metadata = pred_meta.has(subj_str) ? Dictionary(pred_meta[subj_str]) : Dictionary();
			result.push_back(t);
		}
	}
	// Entity capabilities (state)
	Dictionary entity_caps = blackboard->get_var(_BB_ENTITY_CAPABILITIES, Dictionary(), false);
	keys = entity_caps.keys();
	for (int i = 0; i < keys.size(); i++) {
		String eid = keys[i];
		Dictionary caps = entity_caps[keys[i]];
		Array cap_keys = caps.keys();
		for (int j = 0; j < cap_keys.size(); j++) {
			String cap = cap_keys[j];
			KnowledgeTriple t;
			t.subject = "entity_" + eid;
			t.predicate = "capability_" + cap;
			t.object = caps[cap];
			t.metadata["type"] = "state";
			t.metadata["timestamp"] = OS::get_singleton()->get_ticks_usec();
			result.push_back(t);
		}
	}
	// Entity capabilities (public/fact)
	Dictionary entity_caps_pub = blackboard->get_var(_BB_ENTITY_CAPABILITIES_PUBLIC, Dictionary(), false);
	keys = entity_caps_pub.keys();
	for (int i = 0; i < keys.size(); i++) {
		String eid = keys[i];
		Dictionary caps = entity_caps_pub[keys[i]];
		Array cap_keys = caps.keys();
		for (int j = 0; j < cap_keys.size(); j++) {
			String cap = cap_keys[j];
			KnowledgeTriple t;
			t.subject = "entity_" + eid;
			t.predicate = "capability_" + cap;
			t.object = caps[cap];
			t.metadata["type"] = "fact";
			t.metadata["accessibility"] = "public";
			result.push_back(t);
		}
	}
	// Terrain facts
	Dictionary terrain = blackboard->get_var(_BB_TERRAIN_FACTS, Dictionary(), false);
	keys = terrain.keys();
	for (int i = 0; i < keys.size(); i++) {
		String loc = keys[i];
		Dictionary facts = terrain[keys[i]];
		Array fk = facts.keys();
		for (int j = 0; j < fk.size(); j++) {
			KnowledgeTriple t;
			t.subject = "terrain_" + loc;
			t.predicate = fk[j];
			t.object = facts[fk[j]];
			t.metadata["type"] = "fact";
			t.metadata["source"] = "allocentric";
			t.metadata["accessibility"] = "public";
			result.push_back(t);
		}
	}
	// Shared objects
	Dictionary shared = blackboard->get_var(_BB_SHARED_OBJECTS, Dictionary(), false);
	keys = shared.keys();
	for (int i = 0; i < keys.size(); i++) {
		KnowledgeTriple t;
		t.subject = "shared_object_" + String(keys[i]);
		t.predicate = "data";
		t.object = shared[keys[i]];
		t.metadata["type"] = "fact";
		t.metadata["accessibility"] = "public";
		result.push_back(t);
	}
	// Public events
	Dictionary events = blackboard->get_var(_BB_PUBLIC_EVENTS, Dictionary(), false);
	keys = events.keys();
	for (int i = 0; i < keys.size(); i++) {
		KnowledgeTriple t;
		t.subject = "public_event_" + String(keys[i]);
		t.predicate = "data";
		t.object = events[keys[i]];
		t.metadata["type"] = "fact";
		t.metadata["accessibility"] = "public";
		result.push_back(t);
	}
	// Entity positions
	Dictionary positions = blackboard->get_var(_BB_ENTITY_POSITIONS, Dictionary(), false);
	keys = positions.keys();
	for (int i = 0; i < keys.size(); i++) {
		KnowledgeTriple t;
		t.subject = "entity_" + String(keys[i]);
		t.predicate = "position";
		t.object = positions[keys[i]];
		t.metadata["type"] = "fact";
		t.metadata["accessibility"] = "public";
		result.push_back(t);
	}
	// Beliefs
	Dictionary beliefs = blackboard->get_var(_BB_BELIEFS, Dictionary(), false);
	Dictionary beliefs_meta = blackboard->get_var(_BB_BELIEFS_METADATA, Dictionary(), false);
	Array persona_ids = beliefs.keys();
	for (int i = 0; i < persona_ids.size(); i++) {
		String persona_id = persona_ids[i];
		Dictionary by_target = beliefs[persona_id];
		Dictionary meta_by_target = beliefs_meta.has(persona_id) ? Dictionary(beliefs_meta[persona_id]) : Dictionary();
		Array targets = by_target.keys();
		for (int j = 0; j < targets.size(); j++) {
			String target = targets[j];
			Dictionary preds = by_target[targets[j]];
			Dictionary meta_preds = meta_by_target.has(target) ? Dictionary(meta_by_target[target]) : Dictionary();
			Array pred_keys = preds.keys();
			for (int k = 0; k < pred_keys.size(); k++) {
				String pred = pred_keys[k];
				KnowledgeTriple t;
				t.subject = target;
				t.predicate = pred;
				t.object = preds[pred];
				t.metadata = meta_preds.has(pred) ? Dictionary(meta_preds[pred]) : Dictionary();
				t.metadata["type"] = "belief";
				t.metadata["source"] = persona_id;
				result.push_back(t);
			}
		}
	}
	return result;
}

} // namespace

void PlannerState::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_blackboard", "blackboard"), &PlannerState::set_blackboard);
	ClassDB::bind_method(D_METHOD("get_blackboard"), &PlannerState::get_blackboard);
	ClassDB::bind_method(D_METHOD("to_plan_dictionary"), &PlannerState::to_plan_dictionary);
	ClassDB::bind_method(D_METHOD("apply_plan_state", "state"), &PlannerState::apply_plan_state);

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

void PlannerState::set_blackboard(const Ref<Blackboard> &p_blackboard) {
	blackboard = p_blackboard;
}

Ref<Blackboard> PlannerState::get_blackboard() const {
	return blackboard;
}

Dictionary PlannerState::to_plan_dictionary() const {
	Dictionary state;
	if (!blackboard.is_valid()) {
		return state;
	}
	Dictionary all = blackboard->get_vars_as_dict();
	Array keys = all.keys();
	for (int i = 0; i < keys.size(); i++) {
		StringName k = keys[i];
		if (k == _BB_PLANNER_METADATA || k == _BB_BELIEFS || k == _BB_BELIEFS_METADATA) {
			continue;
		}
		Variant v = all[k];
		if (v.get_type() == Variant::DICTIONARY) {
			state[k] = v;
		}
	}
	// Plan expects entity_capabilities at top level if present
	if (!state.has(_BB_ENTITY_CAPABILITIES) && blackboard->has_var(_BB_ENTITY_CAPABILITIES)) {
		state[_BB_ENTITY_CAPABILITIES] = blackboard->get_var(_BB_ENTITY_CAPABILITIES, Dictionary(), false);
	}
	return state;
}

void PlannerState::apply_plan_state(const Dictionary &p_state) {
	if (!blackboard.is_valid() || p_state.is_empty()) {
		return;
	}
	Array keys = p_state.keys();
	for (int i = 0; i < keys.size(); i++) {
		Variant key = keys[i];
		if (key.get_type() != Variant::STRING && key.get_type() != Variant::STRING_NAME) {
			continue;
		}
		blackboard->set_var(key, p_state[key]);
	}
}

Variant PlannerState::get_predicate(const String &p_subject, const String &p_predicate) const {
	if (!blackboard.is_valid()) {
		return Variant();
	}
	StringName pred_sn(p_predicate);
	Dictionary dict = blackboard->get_var(pred_sn, Dictionary(), false);
	if (dict.has(p_subject)) {
		return dict[p_subject];
	}
	return Variant();
}

void PlannerState::set_predicate(const String &p_subject, const String &p_predicate, const Variant &p_value, const Dictionary &p_metadata) {
	if (!blackboard.is_valid()) {
		return;
	}
	Dictionary predicate_metadata = p_metadata;
	if (predicate_metadata.is_empty()) {
		predicate_metadata["type"] = "state";
		predicate_metadata["confidence"] = 1.0;
		predicate_metadata["timestamp"] = OS::get_singleton()->get_ticks_usec();
		predicate_metadata["accessibility"] = "private";
		predicate_metadata["source"] = "planner";
	}
	StringName pred_sn(p_predicate);
	Dictionary dict = blackboard->get_var(pred_sn, Dictionary(), false);
	dict[p_subject] = p_value;
	blackboard->set_var(pred_sn, dict);
	Dictionary meta = blackboard->get_var(_BB_PLANNER_METADATA, Dictionary(), false);
	if (!meta.has(p_predicate)) {
		meta[p_predicate] = Dictionary();
	}
	Dictionary pred_meta = meta[p_predicate];
	pred_meta[p_subject] = predicate_metadata;
	meta[p_predicate] = pred_meta;
	blackboard->set_var(_BB_PLANNER_METADATA, meta);
}

TypedArray<Dictionary> PlannerState::get_triples_as_array() const {
	TypedArray<Dictionary> result;
	for (const KnowledgeTriple &triple : collect_triples(this)) {
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
	Vector<KnowledgeTriple> tr = collect_triples(this);
	for (const auto &triple : tr) {
		if (!subjects.has(triple.subject)) {
			subjects.push_back(triple.subject);
		}
	}
	return subjects;
}

bool PlannerState::has_subject_variable(const String &p_variable) const {
	Vector<KnowledgeTriple> tr = collect_triples(this);
	for (const auto &triple : tr) {
		if (triple.subject == p_variable) {
			return true;
		}
	}
	return false;
}

bool PlannerState::has_predicate(const String &p_subject, const String &p_predicate) const {
	if (!blackboard.is_valid()) {
		return false;
	}
	Dictionary dict = blackboard->get_var(StringName(p_predicate), Dictionary(), false);
	return dict.has(p_subject);
}

Variant PlannerState::get_entity_capability(const String &p_entity_id, const String &p_capability) const {
	if (!blackboard.is_valid()) {
		return Variant();
	}
	Dictionary dict = blackboard->get_var(_BB_ENTITY_CAPABILITIES, Dictionary(), false);
	if (!dict.has(p_entity_id)) {
		return Variant();
	}
	Dictionary caps = dict[p_entity_id];
	return caps.get(p_capability, Variant());
}

void PlannerState::set_entity_capability(const String &p_entity_id, const String &p_capability, const Variant &p_value) {
	if (!blackboard.is_valid()) {
		return;
	}
	Dictionary dict = blackboard->get_var(_BB_ENTITY_CAPABILITIES, Dictionary(), false);
	if (!dict.has(p_entity_id)) {
		dict[p_entity_id] = Dictionary();
	}
	Dictionary caps = dict[p_entity_id];
	caps[p_capability] = p_value;
	dict[p_entity_id] = caps;
	blackboard->set_var(_BB_ENTITY_CAPABILITIES, dict);
}

bool PlannerState::has_entity(const String &p_entity_id) const {
	if (!blackboard.is_valid()) {
		return false;
	}
	Dictionary dict = blackboard->get_var(_BB_ENTITY_CAPABILITIES, Dictionary(), false);
	return dict.has(p_entity_id);
}

Array PlannerState::get_all_entities() const {
	Array entities;
	if (!blackboard.is_valid()) {
		return entities;
	}
	Dictionary dict = blackboard->get_var(_BB_ENTITY_CAPABILITIES, Dictionary(), false);
	Array keys = dict.keys();
	for (int i = 0; i < keys.size(); i++) {
		entities.push_back(keys[i]);
	}
	return entities;
}

Dictionary PlannerState::get_entity_capabilities(const String &p_entity_id) const {
	if (!blackboard.is_valid()) {
		return Dictionary();
	}
	Dictionary dict = blackboard->get_var(_BB_ENTITY_CAPABILITIES, Dictionary(), false);
	if (dict.has(p_entity_id)) {
		return dict[p_entity_id];
	}
	return Dictionary();
}

Dictionary PlannerState::get_all_entity_capabilities() const {
	if (!blackboard.is_valid()) {
		return Dictionary();
	}
	return blackboard->get_var(_BB_ENTITY_CAPABILITIES, Dictionary(), false);
}

// Allocentric facts implementations

void PlannerState::set_terrain_fact(const String &p_location, const String &p_fact_key, const Variant &p_value) {
	if (!blackboard.is_valid()) {
		return;
	}
	Dictionary terrain = blackboard->get_var(_BB_TERRAIN_FACTS, Dictionary(), false);
	if (!terrain.has(p_location)) {
		terrain[p_location] = Dictionary();
	}
	Dictionary loc = terrain[p_location];
	loc[p_fact_key] = p_value;
	terrain[p_location] = loc;
	blackboard->set_var(_BB_TERRAIN_FACTS, terrain);
}

Variant PlannerState::get_terrain_fact(const String &p_location, const String &p_fact_key) const {
	if (!blackboard.is_valid()) {
		return Variant();
	}
	Dictionary terrain = blackboard->get_var(_BB_TERRAIN_FACTS, Dictionary(), false);
	if (!terrain.has(p_location)) {
		return Variant();
	}
	Dictionary loc = terrain[p_location];
	return loc.get(p_fact_key, Variant());
}

bool PlannerState::has_terrain_fact(const String &p_location, const String &p_fact_key) const {
	if (!blackboard.is_valid()) {
		return false;
	}
	Dictionary terrain = blackboard->get_var(_BB_TERRAIN_FACTS, Dictionary(), false);
	return terrain.has(p_location) && Dictionary(terrain[p_location]).has(p_fact_key);
}

Dictionary PlannerState::get_all_terrain_facts() const {
	if (!blackboard.is_valid()) {
		return Dictionary();
	}
	return blackboard->get_var(_BB_TERRAIN_FACTS, Dictionary(), false);
}

void PlannerState::add_shared_object(const String &p_object_id, const Dictionary &p_object_data) {
	if (!blackboard.is_valid()) {
		return;
	}
	Dictionary shared = blackboard->get_var(_BB_SHARED_OBJECTS, Dictionary(), false);
	shared[p_object_id] = p_object_data;
	blackboard->set_var(_BB_SHARED_OBJECTS, shared);
}

void PlannerState::remove_shared_object(const String &p_object_id) {
	if (!blackboard.is_valid()) {
		return;
	}
	Dictionary shared = blackboard->get_var(_BB_SHARED_OBJECTS, Dictionary(), false);
	shared.erase(p_object_id);
	blackboard->set_var(_BB_SHARED_OBJECTS, shared);
}

Dictionary PlannerState::get_shared_object(const String &p_object_id) const {
	if (!blackboard.is_valid()) {
		return Dictionary();
	}
	Dictionary shared = blackboard->get_var(_BB_SHARED_OBJECTS, Dictionary(), false);
	if (shared.has(p_object_id)) {
		return shared[p_object_id];
	}
	return Dictionary();
}

bool PlannerState::has_shared_object(const String &p_object_id) const {
	if (!blackboard.is_valid()) {
		return false;
	}
	Dictionary shared = blackboard->get_var(_BB_SHARED_OBJECTS, Dictionary(), false);
	return shared.has(p_object_id);
}

Array PlannerState::get_all_shared_object_ids() const {
	Array ids;
	if (!blackboard.is_valid()) {
		return ids;
	}
	Dictionary shared = blackboard->get_var(_BB_SHARED_OBJECTS, Dictionary(), false);
	Array keys = shared.keys();
	for (int i = 0; i < keys.size(); i++) {
		ids.push_back(keys[i]);
	}
	return ids;
}

Dictionary PlannerState::get_all_shared_objects() const {
	if (!blackboard.is_valid()) {
		return Dictionary();
	}
	return blackboard->get_var(_BB_SHARED_OBJECTS, Dictionary(), false);
}

void PlannerState::add_public_event(const String &p_event_id, const Dictionary &p_event_data) {
	if (!blackboard.is_valid()) {
		return;
	}
	Dictionary events = blackboard->get_var(_BB_PUBLIC_EVENTS, Dictionary(), false);
	events[p_event_id] = p_event_data;
	blackboard->set_var(_BB_PUBLIC_EVENTS, events);
}

void PlannerState::remove_public_event(const String &p_event_id) {
	if (!blackboard.is_valid()) {
		return;
	}
	Dictionary events = blackboard->get_var(_BB_PUBLIC_EVENTS, Dictionary(), false);
	events.erase(p_event_id);
	blackboard->set_var(_BB_PUBLIC_EVENTS, events);
}

Dictionary PlannerState::get_public_event(const String &p_event_id) const {
	if (!blackboard.is_valid()) {
		return Dictionary();
	}
	Dictionary events = blackboard->get_var(_BB_PUBLIC_EVENTS, Dictionary(), false);
	if (events.has(p_event_id)) {
		return events[p_event_id];
	}
	return Dictionary();
}

bool PlannerState::has_public_event(const String &p_event_id) const {
	if (!blackboard.is_valid()) {
		return false;
	}
	Dictionary events = blackboard->get_var(_BB_PUBLIC_EVENTS, Dictionary(), false);
	return events.has(p_event_id);
}

Array PlannerState::get_all_public_event_ids() const {
	Array ids;
	if (!blackboard.is_valid()) {
		return ids;
	}
	Dictionary events = blackboard->get_var(_BB_PUBLIC_EVENTS, Dictionary(), false);
	Array keys = events.keys();
	for (int i = 0; i < keys.size(); i++) {
		ids.push_back(keys[i]);
	}
	return ids;
}

Dictionary PlannerState::get_all_public_events() const {
	if (!blackboard.is_valid()) {
		return Dictionary();
	}
	return blackboard->get_var(_BB_PUBLIC_EVENTS, Dictionary(), false);
}

void PlannerState::set_entity_position(const String &p_entity_id, const Variant &p_position) {
	if (!blackboard.is_valid()) {
		return;
	}
	Dictionary positions = blackboard->get_var(_BB_ENTITY_POSITIONS, Dictionary(), false);
	positions[p_entity_id] = p_position;
	blackboard->set_var(_BB_ENTITY_POSITIONS, positions);
}

Variant PlannerState::get_entity_position(const String &p_entity_id) const {
	if (!blackboard.is_valid()) {
		return Variant();
	}
	Dictionary positions = blackboard->get_var(_BB_ENTITY_POSITIONS, Dictionary(), false);
	return positions.get(p_entity_id, Variant());
}

bool PlannerState::has_entity_position(const String &p_entity_id) const {
	if (!blackboard.is_valid()) {
		return false;
	}
	Dictionary positions = blackboard->get_var(_BB_ENTITY_POSITIONS, Dictionary(), false);
	return positions.has(p_entity_id);
}

Dictionary PlannerState::get_all_entity_positions() const {
	if (!blackboard.is_valid()) {
		return Dictionary();
	}
	return blackboard->get_var(_BB_ENTITY_POSITIONS, Dictionary(), false);
}

void PlannerState::set_entity_capability_public(const String &p_entity_id, const String &p_capability, const Variant &p_value) {
	if (!blackboard.is_valid()) {
		return;
	}
	Dictionary dict = blackboard->get_var(_BB_ENTITY_CAPABILITIES_PUBLIC, Dictionary(), false);
	if (!dict.has(p_entity_id)) {
		dict[p_entity_id] = Dictionary();
	}
	Dictionary caps = dict[p_entity_id];
	caps[p_capability] = p_value;
	dict[p_entity_id] = caps;
	blackboard->set_var(_BB_ENTITY_CAPABILITIES_PUBLIC, dict);
}

Variant PlannerState::get_entity_capability_public(const String &p_entity_id, const String &p_capability) const {
	if (!blackboard.is_valid()) {
		return Variant();
	}
	Dictionary dict = blackboard->get_var(_BB_ENTITY_CAPABILITIES_PUBLIC, Dictionary(), false);
	if (!dict.has(p_entity_id)) {
		return Variant();
	}
	Dictionary caps = dict[p_entity_id];
	return caps.get(p_capability, Variant());
}

bool PlannerState::has_entity_capability_public(const String &p_entity_id, const String &p_capability) const {
	if (!blackboard.is_valid()) {
		return false;
	}
	Dictionary dict = blackboard->get_var(_BB_ENTITY_CAPABILITIES_PUBLIC, Dictionary(), false);
	if (!dict.has(p_entity_id)) {
		return false;
	}
	return Dictionary(dict[p_entity_id]).has(p_capability);
}

Dictionary PlannerState::get_all_entity_capabilities_public() const {
	if (!blackboard.is_valid()) {
		return Dictionary();
	}
	return blackboard->get_var(_BB_ENTITY_CAPABILITIES_PUBLIC, Dictionary(), false);
}

Dictionary PlannerState::observe_terrain(const String &p_location) const {
	Dictionary all = get_all_terrain_facts();
	if (all.has(p_location)) {
		return all[p_location];
	}
	return Dictionary();
}

Dictionary PlannerState::observe_shared_objects(const String &p_location) const {
	(void)p_location;
	return get_all_shared_objects();
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
	if (!blackboard.is_valid()) {
		return;
	}
	blackboard->erase_var(_BB_TERRAIN_FACTS);
	blackboard->erase_var(_BB_SHARED_OBJECTS);
	blackboard->erase_var(_BB_PUBLIC_EVENTS);
	blackboard->erase_var(_BB_ENTITY_POSITIONS);
	blackboard->erase_var(_BB_ENTITY_CAPABILITIES_PUBLIC);
}

// Belief methods implementations

int64_t PlannerState::get_belief_timestamp(const String &p_persona_id, const String &p_target, const String &p_predicate) const {
	if (!blackboard.is_valid()) {
		return 0;
	}
	Dictionary beliefs_meta = blackboard->get_var(_BB_BELIEFS_METADATA, Dictionary(), false);
	if (!beliefs_meta.has(p_persona_id)) {
		return 0;
	}
	Dictionary by_target = beliefs_meta[p_persona_id];
	if (!by_target.has(p_target)) {
		return 0;
	}
	Dictionary by_pred = by_target[p_target];
	if (!by_pred.has(p_predicate)) {
		return 0;
	}
	Dictionary m = by_pred[p_predicate];
	return m.get("timestamp", 0);
}

Dictionary PlannerState::get_beliefs_about(const String &p_persona_id, const String &p_target) const {
	if (!blackboard.is_valid()) {
		return Dictionary();
	}
	Dictionary beliefs = blackboard->get_var(_BB_BELIEFS, Dictionary(), false);
	if (!beliefs.has(p_persona_id)) {
		return Dictionary();
	}
	Dictionary by_target = beliefs[p_persona_id];
	if (by_target.has(p_target)) {
		return by_target[p_target];
	}
	return Dictionary();
}

void PlannerState::set_belief_about(const String &p_persona_id, const String &p_target, const String &p_predicate, const Variant &p_value, const Dictionary &p_metadata) {
	if (!blackboard.is_valid()) {
		return;
	}
	Dictionary belief_metadata = p_metadata;
	if (belief_metadata.is_empty()) {
		belief_metadata["type"] = "belief";
		belief_metadata["source"] = p_persona_id;
		belief_metadata["confidence"] = 1.0;
		belief_metadata["timestamp"] = OS::get_singleton()->get_ticks_usec();
		belief_metadata["accessibility"] = "private";
	}
	Dictionary beliefs = blackboard->get_var(_BB_BELIEFS, Dictionary(), false);
	if (!beliefs.has(p_persona_id)) {
		beliefs[p_persona_id] = Dictionary();
	}
	Dictionary by_target = beliefs[p_persona_id];
	if (!by_target.has(p_target)) {
		by_target[p_target] = Dictionary();
	}
	Dictionary by_pred = by_target[p_target];
	by_pred[p_predicate] = p_value;
	by_target[p_target] = by_pred;
	beliefs[p_persona_id] = by_target;
	blackboard->set_var(_BB_BELIEFS, beliefs);

	Dictionary beliefs_meta = blackboard->get_var(_BB_BELIEFS_METADATA, Dictionary(), false);
	if (!beliefs_meta.has(p_persona_id)) {
		beliefs_meta[p_persona_id] = Dictionary();
	}
	Dictionary meta_by_target = beliefs_meta[p_persona_id];
	if (!meta_by_target.has(p_target)) {
		meta_by_target[p_target] = Dictionary();
	}
	Dictionary meta_by_pred = meta_by_target[p_target];
	meta_by_pred[p_predicate] = belief_metadata;
	meta_by_target[p_target] = meta_by_pred;
	beliefs_meta[p_persona_id] = meta_by_target;
	blackboard->set_var(_BB_BELIEFS_METADATA, beliefs_meta);
}

double PlannerState::get_belief_confidence(const String &p_persona_id, const String &p_target, const String &p_predicate) const {
	if (!blackboard.is_valid()) {
		return 1.0;
	}
	Dictionary beliefs_meta = blackboard->get_var(_BB_BELIEFS_METADATA, Dictionary(), false);
	if (!beliefs_meta.has(p_persona_id)) {
		return 1.0;
	}
	Dictionary by_target = beliefs_meta[p_persona_id];
	if (!by_target.has(p_target)) {
		return 1.0;
	}
	Dictionary by_pred = by_target[p_target];
	if (!by_pred.has(p_predicate)) {
		return 1.0;
	}
	Dictionary m = by_pred[p_predicate];
	return m.get("confidence", 1.0);
}

void PlannerState::update_belief_confidence(const String &p_persona_id, const String &p_target, const String &p_predicate, double p_confidence) {
	if (!blackboard.is_valid()) {
		return;
	}
	Dictionary beliefs_meta = blackboard->get_var(_BB_BELIEFS_METADATA, Dictionary(), false);
	if (!beliefs_meta.has(p_persona_id)) {
		return;
	}
	Dictionary by_target = beliefs_meta[p_persona_id];
	if (!by_target.has(p_target)) {
		return;
	}
	Dictionary by_pred = by_target[p_target];
	if (!by_pred.has(p_predicate)) {
		return;
	}
	Dictionary m = by_pred[p_predicate];
	m["confidence"] = p_confidence;
	by_pred[p_predicate] = m;
	by_target[p_target] = by_pred;
	beliefs_meta[p_persona_id] = by_target;
	blackboard->set_var(_BB_BELIEFS_METADATA, beliefs_meta);
}

PlannerState::PlannerState() {}

PlannerState::~PlannerState() {}
