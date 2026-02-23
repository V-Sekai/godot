/**************************************************************************/
/*  planner_persona.h                                                     */
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
#include "planner_state.h"

// Persona identity types based on capabilities
enum PlannerPersonaIdentity {
	IDENTITY_BASIC = 0, // No special capabilities
	IDENTITY_HUMAN = 1, // Human capabilities (movable, inventory, craft, mine, build, interact)
	IDENTITY_AI = 2, // AI capabilities (movable, compute, optimize, predict, learn, navigate)
	IDENTITY_HUMAN_AND_AI = 3 // Both human and AI capabilities
};

// Unified Persona Entity for belief-immersed planning architecture
// Personas can be human, AI, or hybrid, differentiated by capabilities
class PlannerPersona : public Resource {
	GDCLASS(PlannerPersona, Resource);

private:
	String persona_id;
	String persona_name;
	PlannerPersonaIdentity identity_type;
	bool active;
	Dictionary metadata; // Stores character, position, capabilities data
	Ref<PlannerState> belief_state; // Unified state for beliefs
	TypedArray<String> capabilities; // Capabilities that determine persona type

	// Human persona capabilities
	static const String CAPABILITY_MOVABLE;
	static const String CAPABILITY_INVENTORY;
	static const String CAPABILITY_CRAFT;
	static const String CAPABILITY_MINE;
	static const String CAPABILITY_BUILD;
	static const String CAPABILITY_INTERACT;

	// AI persona capabilities
	static const String CAPABILITY_COMPUTE;
	static const String CAPABILITY_OPTIMIZE;
	static const String CAPABILITY_PREDICT;
	static const String CAPABILITY_LEARN;
	static const String CAPABILITY_NAVIGATE;

	// Update identity type based on capabilities
	void _update_identity_type();

protected:
	static void _bind_methods();

public:
	PlannerPersona();
	~PlannerPersona();

	// Creation and initialization
	void initialize(const String &p_id, const String &p_name);
	static Ref<PlannerPersona> create_basic(const String &p_id, const String &p_name);
	static Ref<PlannerPersona> create_human(const String &p_id, const String &p_name);
	static Ref<PlannerPersona> create_ai(const String &p_id, const String &p_name);
	static Ref<PlannerPersona> create_hybrid(const String &p_id, const String &p_name);

	// Getters
	String get_persona_id() const { return persona_id; }
	String get_persona_name() const { return persona_name; }
	int get_identity_type() const { return static_cast<int>(identity_type); }
	bool is_active() const { return active; }
	Dictionary get_metadata() const { return metadata; }
	Ref<PlannerState> get_belief_state() const { return belief_state; }
	Dictionary get_beliefs_about_others() const { return Dictionary(); } // Deprecated, use get_beliefs_about
	Dictionary get_belief_confidence() const { return Dictionary(); } // Deprecated
	TypedArray<String> get_capabilities() const { return capabilities; }

	// Setters
	void set_persona_id(const String &p_id) { persona_id = p_id; }
	void set_persona_name(const String &p_name) { persona_name = p_name; }
	void set_active(bool p_active) { active = p_active; }
	void set_metadata(const Dictionary &p_metadata) { metadata = p_metadata; }

	// Capability management
	bool has_capability(const String &p_capability) const;
	void add_capability(const String &p_capability);
	void remove_capability(const String &p_capability);
	void enable_human_capabilities();
	void enable_ai_capabilities();

	// Belief management
	Dictionary get_beliefs_about(const String &p_target_persona_id) const;
	void set_belief_about(const String &p_target_persona_id, const String &p_belief_key, const Variant &p_belief_value, double p_confidence = 1.0, int64_t p_timestamp = 0);
	double get_belief_confidence_for(const String &p_target_persona_id, const String &p_belief_key) const;
	int64_t get_belief_timestamp_for(const String &p_target_persona_id, const String &p_belief_key) const;
	void update_belief_confidence(const String &p_target_persona_id, const String &p_belief_key, double p_confidence);

	// Information asymmetry: Personas cannot directly access each other's internal states
	Dictionary get_planner_state(const String &p_target_persona_id, const String &p_requesting_persona_id) const;

	// Observation and communication for belief formation
	void process_observation(const Dictionary &p_observation);
	void process_communication(const Dictionary &p_communication);
};
