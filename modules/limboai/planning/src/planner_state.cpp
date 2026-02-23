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
#include "core/templates/local_vector.h"
#include "core/templates/vector.h"
#include "core/variant/variant.h"

// Blackboard: two keys only — planner_metadata (nested) and entity_capability (state caps)
#define _BB_PLANNER_METADATA StringName("planner_metadata")
#define _BB_ENTITY_CAPABILITY StringName("entity_capability")

// Sub-keys inside planner_metadata (do not use as top-level BB keys)
#define _META_TERRAIN_FACTS StringName("terrain_facts")
#define _META_SHARED_OBJECTS StringName("shared_objects")
#define _META_PUBLIC_EVENTS StringName("public_events")
#define _META_ENTITY_POSITIONS StringName("entity_positions")
#define _META_BELIEFS StringName("beliefs")
#define _META_BELIEFS_METADATA StringName("beliefs_metadata")
#define _META_ENTITY_CAPABILITIES_PUBLIC StringName("entity_capabilities_public")

namespace {

static Dictionary get_meta_sub(const Ref<Blackboard> &bb, const StringName &sub_key) {
	if (!bb.is_valid()) {
		return Dictionary();
	}
	Dictionary meta = bb->get_var(_BB_PLANNER_METADATA, Dictionary(), false);
	return meta.get(sub_key, Dictionary());
}

static void set_meta_sub(const Ref<Blackboard> &bb, const StringName &sub_key, const Dictionary &value) {
	if (!bb.is_valid()) {
		return;
	}
	Dictionary meta = bb->get_var(_BB_PLANNER_METADATA, Dictionary(), false);
	meta[sub_key] = value;
	bb->set_var(_BB_PLANNER_METADATA, meta);
}

struct TripleEntry {
	String subject;
	String predicate;
	Variant object;
	Dictionary metadata;
};

static void push_triple(LocalVector<TripleEntry> &result, const String &subject, const String &predicate, const Variant &object, const Dictionary &metadata) {
	result.push_back({ subject, predicate, object, metadata });
}

static LocalVector<TripleEntry> collect_triples(const PlannerState *state) {
	LocalVector<TripleEntry> result;
	Ref<Blackboard> blackboard = state->get_blackboard();
	if (!blackboard.is_valid()) {
		return result;
	}
	Dictionary all = blackboard->get_vars_as_dict();
	Dictionary meta_all = blackboard->get_var(_BB_PLANNER_METADATA, Dictionary(), false);
	Array keys = all.keys();
	for (int i = 0; i < keys.size(); i++) {
		StringName k = keys[i];
		if (k == _BB_PLANNER_METADATA || k == _BB_ENTITY_CAPABILITY) {
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
			push_triple(result, subj_str, String(k), dict[subj], pred_meta.has(subj_str) ? Dictionary(pred_meta[subj_str]) : Dictionary());
		}
	}
	// Entity capabilities (state)
	Dictionary entity_caps = blackboard->get_var(_BB_ENTITY_CAPABILITY, Dictionary(), false);
	keys = entity_caps.keys();
	for (int i = 0; i < keys.size(); i++) {
		String eid = keys[i];
		Dictionary caps = entity_caps[keys[i]];
		Array cap_keys = caps.keys();
		for (int j = 0; j < cap_keys.size(); j++) {
			String cap = cap_keys[j];
			Dictionary meta;
			meta["type"] = "state";
			meta["timestamp"] = OS::get_singleton()->get_ticks_usec();
			push_triple(result, "entity_" + eid, "capability_" + cap, caps[cap], meta);
		}
	}
	// Entity capabilities (public/fact) — stored under planner_metadata
	Dictionary entity_caps_pub = get_meta_sub(blackboard, _META_ENTITY_CAPABILITIES_PUBLIC);
	keys = entity_caps_pub.keys();
	for (int i = 0; i < keys.size(); i++) {
		String eid = keys[i];
		Dictionary caps = entity_caps_pub[keys[i]];
		Array cap_keys = caps.keys();
		for (int j = 0; j < cap_keys.size(); j++) {
			String cap = cap_keys[j];
			Dictionary meta;
			meta["type"] = "fact";
			meta["accessibility"] = "public";
			push_triple(result, "entity_" + eid, "capability_" + cap, caps[cap], meta);
		}
	}
	// Terrain facts
	Dictionary terrain = get_meta_sub(blackboard, _META_TERRAIN_FACTS);
	keys = terrain.keys();
	for (int i = 0; i < keys.size(); i++) {
		String loc = keys[i];
		Dictionary facts = terrain[keys[i]];
		Array fk = facts.keys();
		for (int j = 0; j < fk.size(); j++) {
			Dictionary meta;
			meta["type"] = "fact";
			meta["source"] = "allocentric";
			meta["accessibility"] = "public";
			push_triple(result, "terrain_" + loc, String(fk[j]), facts[fk[j]], meta);
		}
	}
	// Shared objects
	Dictionary shared = get_meta_sub(blackboard, _META_SHARED_OBJECTS);
	keys = shared.keys();
	for (int i = 0; i < keys.size(); i++) {
		Dictionary meta;
		meta["type"] = "fact";
		meta["accessibility"] = "public";
		push_triple(result, "shared_object_" + String(keys[i]), "data", shared[keys[i]], meta);
	}
	// Public events
	Dictionary events = get_meta_sub(blackboard, _META_PUBLIC_EVENTS);
	keys = events.keys();
	for (int i = 0; i < keys.size(); i++) {
		Dictionary meta;
		meta["type"] = "fact";
		meta["accessibility"] = "public";
		push_triple(result, "public_event_" + String(keys[i]), "data", events[keys[i]], meta);
	}
	// Entity positions
	Dictionary positions = get_meta_sub(blackboard, _META_ENTITY_POSITIONS);
	keys = positions.keys();
	for (int i = 0; i < keys.size(); i++) {
		Dictionary meta;
		meta["type"] = "fact";
		meta["accessibility"] = "public";
		push_triple(result, "entity_" + String(keys[i]), "position", positions[keys[i]], meta);
	}
	// Beliefs
	Dictionary beliefs = get_meta_sub(blackboard, _META_BELIEFS);
	Dictionary beliefs_meta = get_meta_sub(blackboard, _META_BELIEFS_METADATA);
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
				Dictionary meta = meta_preds.has(pred) ? Dictionary(meta_preds[pred]) : Dictionary();
				meta["type"] = "belief";
				meta["source"] = persona_id;
				push_triple(result, target, pred, preds[pred], meta);
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
		if (k == _BB_PLANNER_METADATA || k == _BB_ENTITY_CAPABILITY) {
			continue;
		}
		Variant v = all[k];
		if (v.get_type() == Variant::DICTIONARY) {
			state[k] = v;
		}
	}
	// Plan expects flat state: entity_capabilities and allocentric/belief keys at top level
	Dictionary entity_caps = blackboard->get_var(_BB_ENTITY_CAPABILITY, Dictionary(), false);
	if (!entity_caps.is_empty()) {
		state[StringName("entity_capabilities")] = entity_caps;
	}
	Dictionary meta = blackboard->get_var(_BB_PLANNER_METADATA, Dictionary(), false);
	if (meta.has(_META_TERRAIN_FACTS)) {
		state[_META_TERRAIN_FACTS] = meta[_META_TERRAIN_FACTS];
	}
	if (meta.has(_META_SHARED_OBJECTS)) {
		state[_META_SHARED_OBJECTS] = meta[_META_SHARED_OBJECTS];
	}
	if (meta.has(_META_PUBLIC_EVENTS)) {
		state[_META_PUBLIC_EVENTS] = meta[_META_PUBLIC_EVENTS];
	}
	if (meta.has(_META_ENTITY_POSITIONS)) {
		state[_META_ENTITY_POSITIONS] = meta[_META_ENTITY_POSITIONS];
	}
	if (meta.has(_META_BELIEFS)) {
		state[_META_BELIEFS] = meta[_META_BELIEFS];
	}
	if (meta.has(_META_BELIEFS_METADATA)) {
		state[_META_BELIEFS_METADATA] = meta[_META_BELIEFS_METADATA];
	}
	if (meta.has(_META_ENTITY_CAPABILITIES_PUBLIC)) {
		state[_META_ENTITY_CAPABILITIES_PUBLIC] = meta[_META_ENTITY_CAPABILITIES_PUBLIC];
	}
	return state;
}

void PlannerState::apply_plan_state(const Dictionary &p_state) {
	if (!blackboard.is_valid() || p_state.is_empty()) {
		return;
	}
	static const StringName entity_capabilities_key("entity_capabilities");
	Array keys = p_state.keys();
	for (int i = 0; i < keys.size(); i++) {
		Variant key = keys[i];
		if (key.get_type() != Variant::STRING && key.get_type() != Variant::STRING_NAME) {
			continue;
		}
		StringName k = key;
		Variant v = p_state[key];
		if (k == entity_capabilities_key) {
			if (v.get_type() == Variant::DICTIONARY) {
				blackboard->set_var(_BB_ENTITY_CAPABILITY, v);
			}
			continue;
		}
		if (k == _META_TERRAIN_FACTS || k == _META_SHARED_OBJECTS || k == _META_PUBLIC_EVENTS ||
				k == _META_ENTITY_POSITIONS || k == _META_BELIEFS || k == _META_BELIEFS_METADATA ||
				k == _META_ENTITY_CAPABILITIES_PUBLIC) {
			if (v.get_type() == Variant::DICTIONARY) {
				set_meta_sub(blackboard, k, v);
			}
			continue;
		}
		// Predicate or other top-level state key
		blackboard->set_var(k, v);
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
	if (!meta.has(StringName(p_predicate))) {
		meta[StringName(p_predicate)] = Dictionary();
	}
	Dictionary pred_meta = meta[StringName(p_predicate)];
	pred_meta[p_subject] = predicate_metadata;
	meta[StringName(p_predicate)] = pred_meta;
	blackboard->set_var(_BB_PLANNER_METADATA, meta);
}

TypedArray<Dictionary> PlannerState::get_triples_as_array() const {
	TypedArray<Dictionary> out;
	const LocalVector<TripleEntry> tr = collect_triples(this);
	// Canonical triple dict key order: predicate, subject, object, metadata (same order everywhere in LimboAI).
	for (const TripleEntry &e : tr) {
		Dictionary d;
		d["predicate"] = e.predicate;
		d["subject"] = e.subject;
		d["object"] = e.object;
		d["metadata"] = e.metadata;
		out.push_back(d);
	}
	return out;
}

TypedArray<String> PlannerState::get_subject_predicate_list() const {
	TypedArray<String> subjects;
	const LocalVector<TripleEntry> tr = collect_triples(this);
	for (const TripleEntry &e : tr) {
		if (!e.subject.is_empty() && !subjects.has(e.subject)) {
			subjects.push_back(e.subject);
		}
	}
	return subjects;
}

bool PlannerState::has_subject_variable(const String &p_variable) const {
	const LocalVector<TripleEntry> tr = collect_triples(this);
	for (const TripleEntry &e : tr) {
		if (e.subject == p_variable) {
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
	Dictionary dict = blackboard->get_var(_BB_ENTITY_CAPABILITY, Dictionary(), false);
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
	Dictionary dict = blackboard->get_var(_BB_ENTITY_CAPABILITY, Dictionary(), false);
	if (!dict.has(p_entity_id)) {
		dict[p_entity_id] = Dictionary();
	}
	Dictionary caps = dict[p_entity_id];
	caps[p_capability] = p_value;
	dict[p_entity_id] = caps;
	blackboard->set_var(_BB_ENTITY_CAPABILITY, dict);
}

bool PlannerState::has_entity(const String &p_entity_id) const {
	if (!blackboard.is_valid()) {
		return false;
	}
	Dictionary dict = blackboard->get_var(_BB_ENTITY_CAPABILITY, Dictionary(), false);
	return dict.has(p_entity_id);
}

Array PlannerState::get_all_entities() const {
	Array entities;
	if (!blackboard.is_valid()) {
		return entities;
	}
	Dictionary dict = blackboard->get_var(_BB_ENTITY_CAPABILITY, Dictionary(), false);
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
	Dictionary dict = blackboard->get_var(_BB_ENTITY_CAPABILITY, Dictionary(), false);
	if (dict.has(p_entity_id)) {
		return dict[p_entity_id];
	}
	return Dictionary();
}

Dictionary PlannerState::get_all_entity_capabilities() const {
	if (!blackboard.is_valid()) {
		return Dictionary();
	}
	return blackboard->get_var(_BB_ENTITY_CAPABILITY, Dictionary(), false);
}

// Allocentric facts implementations

void PlannerState::set_terrain_fact(const String &p_location, const String &p_fact_key, const Variant &p_value) {
	if (!blackboard.is_valid()) {
		return;
	}
	Dictionary terrain = get_meta_sub(blackboard, _META_TERRAIN_FACTS);
	if (!terrain.has(p_location)) {
		terrain[p_location] = Dictionary();
	}
	Dictionary loc = terrain[p_location];
	loc[p_fact_key] = p_value;
	terrain[p_location] = loc;
	set_meta_sub(blackboard, _META_TERRAIN_FACTS, terrain);
}

Variant PlannerState::get_terrain_fact(const String &p_location, const String &p_fact_key) const {
	if (!blackboard.is_valid()) {
		return Variant();
	}
	Dictionary terrain = get_meta_sub(blackboard, _META_TERRAIN_FACTS);
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
	Dictionary terrain = get_meta_sub(blackboard, _META_TERRAIN_FACTS);
	return terrain.has(p_location) && Dictionary(terrain[p_location]).has(p_fact_key);
}

Dictionary PlannerState::get_all_terrain_facts() const {
	if (!blackboard.is_valid()) {
		return Dictionary();
	}
	return get_meta_sub(blackboard, _META_TERRAIN_FACTS);
}

void PlannerState::add_shared_object(const String &p_object_id, const Dictionary &p_object_data) {
	if (!blackboard.is_valid()) {
		return;
	}
	Dictionary shared = get_meta_sub(blackboard, _META_SHARED_OBJECTS);
	shared[p_object_id] = p_object_data;
	set_meta_sub(blackboard, _META_SHARED_OBJECTS, shared);
}

void PlannerState::remove_shared_object(const String &p_object_id) {
	if (!blackboard.is_valid()) {
		return;
	}
	Dictionary shared = get_meta_sub(blackboard, _META_SHARED_OBJECTS);
	shared.erase(p_object_id);
	set_meta_sub(blackboard, _META_SHARED_OBJECTS, shared);
}

Dictionary PlannerState::get_shared_object(const String &p_object_id) const {
	if (!blackboard.is_valid()) {
		return Dictionary();
	}
	Dictionary shared = get_meta_sub(blackboard, _META_SHARED_OBJECTS);
	if (shared.has(p_object_id)) {
		return shared[p_object_id];
	}
	return Dictionary();
}

bool PlannerState::has_shared_object(const String &p_object_id) const {
	if (!blackboard.is_valid()) {
		return false;
	}
	Dictionary shared = get_meta_sub(blackboard, _META_SHARED_OBJECTS);
	return shared.has(p_object_id);
}

Array PlannerState::get_all_shared_object_ids() const {
	Array ids;
	if (!blackboard.is_valid()) {
		return ids;
	}
	Dictionary shared = get_meta_sub(blackboard, _META_SHARED_OBJECTS);
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
	return get_meta_sub(blackboard, _META_SHARED_OBJECTS);
}

void PlannerState::add_public_event(const String &p_event_id, const Dictionary &p_event_data) {
	if (!blackboard.is_valid()) {
		return;
	}
	Dictionary events = get_meta_sub(blackboard, _META_PUBLIC_EVENTS);
	events[p_event_id] = p_event_data;
	set_meta_sub(blackboard, _META_PUBLIC_EVENTS, events);
}

void PlannerState::remove_public_event(const String &p_event_id) {
	if (!blackboard.is_valid()) {
		return;
	}
	Dictionary events = get_meta_sub(blackboard, _META_PUBLIC_EVENTS);
	events.erase(p_event_id);
	set_meta_sub(blackboard, _META_PUBLIC_EVENTS, events);
}

Dictionary PlannerState::get_public_event(const String &p_event_id) const {
	if (!blackboard.is_valid()) {
		return Dictionary();
	}
	Dictionary events = get_meta_sub(blackboard, _META_PUBLIC_EVENTS);
	if (events.has(p_event_id)) {
		return events[p_event_id];
	}
	return Dictionary();
}

bool PlannerState::has_public_event(const String &p_event_id) const {
	if (!blackboard.is_valid()) {
		return false;
	}
	Dictionary events = get_meta_sub(blackboard, _META_PUBLIC_EVENTS);
	return events.has(p_event_id);
}

Array PlannerState::get_all_public_event_ids() const {
	Array ids;
	if (!blackboard.is_valid()) {
		return ids;
	}
	Dictionary events = get_meta_sub(blackboard, _META_PUBLIC_EVENTS);
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
	return get_meta_sub(blackboard, _META_PUBLIC_EVENTS);
}

void PlannerState::set_entity_position(const String &p_entity_id, const Variant &p_position) {
	if (!blackboard.is_valid()) {
		return;
	}
	Dictionary positions = get_meta_sub(blackboard, _META_ENTITY_POSITIONS);
	positions[p_entity_id] = p_position;
	set_meta_sub(blackboard, _META_ENTITY_POSITIONS, positions);
}

Variant PlannerState::get_entity_position(const String &p_entity_id) const {
	if (!blackboard.is_valid()) {
		return Variant();
	}
	Dictionary positions = get_meta_sub(blackboard, _META_ENTITY_POSITIONS);
	return positions.get(p_entity_id, Variant());
}

bool PlannerState::has_entity_position(const String &p_entity_id) const {
	if (!blackboard.is_valid()) {
		return false;
	}
	Dictionary positions = get_meta_sub(blackboard, _META_ENTITY_POSITIONS);
	return positions.has(p_entity_id);
}

Dictionary PlannerState::get_all_entity_positions() const {
	if (!blackboard.is_valid()) {
		return Dictionary();
	}
	return get_meta_sub(blackboard, _META_ENTITY_POSITIONS);
}

void PlannerState::set_entity_capability_public(const String &p_entity_id, const String &p_capability, const Variant &p_value) {
	if (!blackboard.is_valid()) {
		return;
	}
	Dictionary dict = get_meta_sub(blackboard, _META_ENTITY_CAPABILITIES_PUBLIC);
	if (!dict.has(p_entity_id)) {
		dict[p_entity_id] = Dictionary();
	}
	Dictionary caps = dict[p_entity_id];
	caps[p_capability] = p_value;
	dict[p_entity_id] = caps;
	set_meta_sub(blackboard, _META_ENTITY_CAPABILITIES_PUBLIC, dict);
}

Variant PlannerState::get_entity_capability_public(const String &p_entity_id, const String &p_capability) const {
	if (!blackboard.is_valid()) {
		return Variant();
	}
	Dictionary dict = get_meta_sub(blackboard, _META_ENTITY_CAPABILITIES_PUBLIC);
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
	Dictionary dict = get_meta_sub(blackboard, _META_ENTITY_CAPABILITIES_PUBLIC);
	if (!dict.has(p_entity_id)) {
		return false;
	}
	return Dictionary(dict[p_entity_id]).has(p_capability);
}

Dictionary PlannerState::get_all_entity_capabilities_public() const {
	if (!blackboard.is_valid()) {
		return Dictionary();
	}
	return get_meta_sub(blackboard, _META_ENTITY_CAPABILITIES_PUBLIC);
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
	Dictionary meta = blackboard->get_var(_BB_PLANNER_METADATA, Dictionary(), false);
	meta[_META_TERRAIN_FACTS] = Dictionary();
	meta[_META_SHARED_OBJECTS] = Dictionary();
	meta[_META_PUBLIC_EVENTS] = Dictionary();
	meta[_META_ENTITY_POSITIONS] = Dictionary();
	meta[_META_ENTITY_CAPABILITIES_PUBLIC] = Dictionary();
	blackboard->set_var(_BB_PLANNER_METADATA, meta);
}

// Belief methods implementations

int64_t PlannerState::get_belief_timestamp(const String &p_persona_id, const String &p_target, const String &p_predicate) const {
	if (!blackboard.is_valid()) {
		return 0;
	}
	Dictionary beliefs_meta = get_meta_sub(blackboard, _META_BELIEFS_METADATA);
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
	Dictionary beliefs = get_meta_sub(blackboard, _META_BELIEFS);
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
	Dictionary beliefs = get_meta_sub(blackboard, _META_BELIEFS);
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
	set_meta_sub(blackboard, _META_BELIEFS, beliefs);

	Dictionary beliefs_meta = get_meta_sub(blackboard, _META_BELIEFS_METADATA);
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
	set_meta_sub(blackboard, _META_BELIEFS_METADATA, beliefs_meta);
}

double PlannerState::get_belief_confidence(const String &p_persona_id, const String &p_target, const String &p_predicate) const {
	if (!blackboard.is_valid()) {
		return 1.0;
	}
	Dictionary beliefs_meta = get_meta_sub(blackboard, _META_BELIEFS_METADATA);
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
	Dictionary beliefs_meta = get_meta_sub(blackboard, _META_BELIEFS_METADATA);
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
	set_meta_sub(blackboard, _META_BELIEFS_METADATA, beliefs_meta);
}

PlannerState::PlannerState() {}

PlannerState::~PlannerState() {}
