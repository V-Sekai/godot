#pragma once

// SPDX-FileCopyrightText: 2025-present K. S. Ernest (iFire) Lee
// SPDX-License-Identifier: MIT

#include "core/io/resource.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"
#include "core/templates/vector.h"

struct KnowledgeTriple {
	String subject;
	String predicate;
	Variant object;
	Dictionary metadata;

	KnowledgeTriple() = default;
	KnowledgeTriple(const String &p_subject, const String &p_predicate, const Variant &p_object, const Dictionary &p_metadata = Dictionary())
		: subject(p_subject), predicate(p_predicate), object(p_object), metadata(p_metadata) {}
};

class PlannerState : public Resource {
	GDCLASS(PlannerState, Resource);

	Vector<KnowledgeTriple> triples;

protected:
	static void _bind_methods();

public:
	PlannerState();
	~PlannerState();

	// Triple-based knowledge representation
	Variant get_predicate(const String &p_subject, const String &p_predicate) const;
	void set_predicate(const String &p_subject, const String &p_predicate, const Variant &p_value, const Dictionary &p_metadata = Dictionary());
	TypedArray<Dictionary> get_triples_as_array() const;
	const Vector<KnowledgeTriple> &get_triples() const { return triples; }

	// Legacy Dictionary-based interface for backward compatibility
	TypedArray<String> get_subject_predicate_list() const;
	bool has_subject_variable(const String &p_variable) const;
	bool has_predicate(const String &p_subject, const String &p_predicate) const;

	// Entity capabilities (migrated to triples)
	Variant get_entity_capability(const String &p_entity_id, const String &p_capability) const;
	void set_entity_capability(const String &p_entity_id, const String &p_capability, const Variant &p_value);
	bool has_entity(const String &p_entity_id) const;
	Dictionary get_entity_capabilities(const String &p_entity_id) const;
	Dictionary get_all_entity_capabilities() const;
	Array get_all_entities() const;

	// Terrain facts (allocentric)
	void set_terrain_fact(const String &p_location, const String &p_fact_key, const Variant &p_value);
	Variant get_terrain_fact(const String &p_location, const String &p_fact_key) const;
	bool has_terrain_fact(const String &p_location, const String &p_fact_key) const;
	Dictionary get_all_terrain_facts() const;

	// Shared objects
	void add_shared_object(const String &p_object_id, const Dictionary &p_object_data);
	void remove_shared_object(const String &p_object_id);
	Dictionary get_shared_object(const String &p_object_id) const;
	bool has_shared_object(const String &p_object_id) const;
	Array get_all_shared_object_ids() const;
	Dictionary get_all_shared_objects() const;

	// Public events
	void add_public_event(const String &p_event_id, const Dictionary &p_event_data);
	void remove_public_event(const String &p_event_id);
	Dictionary get_public_event(const String &p_event_id) const;
	bool has_public_event(const String &p_event_id) const;
	Array get_all_public_event_ids() const;
	Dictionary get_all_public_events() const;

	// Entity positions
	void set_entity_position(const String &p_entity_id, const Variant &p_position);
	Variant get_entity_position(const String &p_entity_id) const;
	bool has_entity_position(const String &p_entity_id) const;
	Dictionary get_all_entity_positions() const;

	// Public entity capabilities
	void set_entity_capability_public(const String &p_entity_id, const String &p_capability, const Variant &p_value);
	Variant get_entity_capability_public(const String &p_entity_id, const String &p_capability) const;
	bool has_entity_capability_public(const String &p_entity_id, const String &p_capability) const;
	Dictionary get_all_entity_capabilities_public() const;

	// Observation methods
	Dictionary observe_terrain(const String &p_location) const;
	Dictionary observe_shared_objects(const String &p_location) const;
	Dictionary observe_public_events() const;
	Dictionary observe_entity_positions() const;
	Dictionary observe_entity_capabilities() const;

	// Clear methods
	void clear_allocentric_facts();

	// Belief management with metadata
	Dictionary get_beliefs_about(const String &p_persona_id, const String &p_target) const;
	void set_belief_about(const String &p_persona_id, const String &p_target, const String &p_predicate, const Variant &p_value, const Dictionary &p_metadata = Dictionary());
	double get_belief_confidence(const String &p_persona_id, const String &p_target, const String &p_predicate) const;
	int64_t get_belief_timestamp(const String &p_persona_id, const String &p_target, const String &p_predicate) const;
	void update_belief_confidence(const String &p_persona_id, const String &p_target, const String &p_predicate, double p_confidence);
};
