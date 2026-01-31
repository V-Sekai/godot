/**************************************************************************/
/*  test_persona_belief.h                                                 */
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

#include "../../src/plan.h"
#include "../../src/planner_belief_manager.h"
#include "../../src/planner_persona.h"

#include "core/math/vector3.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"
#include "tests/test_macros.h"

namespace TestPersonaBelief {

TEST_CASE("[Modules][Planner][Persona] Create basic persona") {
	Ref<PlannerPersona> persona = PlannerPersona::create_basic("persona_001", "TestPersona");
	CHECK(persona.is_valid());
	CHECK(persona->get_persona_id() == "persona_001");
	CHECK(persona->get_persona_name() == "TestPersona");
	CHECK(persona->get_identity_type() == PlannerPersonaIdentity::IDENTITY_BASIC);
	CHECK(persona->is_active());
}

TEST_CASE("[Modules][Planner][Persona] Create human persona with capabilities") {
	Ref<PlannerPersona> persona = PlannerPersona::create_human("human_001", "Alice");
	CHECK(persona.is_valid());
	CHECK(persona->get_identity_type() == PlannerPersonaIdentity::IDENTITY_HUMAN);
	CHECK(persona->has_capability("inventory"));
	CHECK(persona->has_capability("craft"));
	CHECK(persona->has_capability("mine"));
	CHECK(!persona->has_capability("compute")); // Human doesn't have AI capabilities
}

TEST_CASE("[Modules][Planner][Persona] Create AI persona with capabilities") {
	Ref<PlannerPersona> persona = PlannerPersona::create_ai("ai_001", "Bot");
	CHECK(persona.is_valid());
	CHECK(persona->get_identity_type() == PlannerPersonaIdentity::IDENTITY_AI);
	CHECK(persona->has_capability("compute"));
	CHECK(persona->has_capability("optimize"));
	CHECK(persona->has_capability("predict"));
	CHECK(!persona->has_capability("inventory")); // AI doesn't have human capabilities
}

TEST_CASE("[Modules][Planner][Persona] Create hybrid persona") {
	Ref<PlannerPersona> persona = PlannerPersona::create_hybrid("hybrid_001", "Cyborg");
	CHECK(persona.is_valid());
	CHECK(persona->get_identity_type() == PlannerPersonaIdentity::IDENTITY_HUMAN_AND_AI);
	CHECK(persona->has_capability("inventory")); // Has human capabilities
	CHECK(persona->has_capability("compute")); // Has AI capabilities
}

TEST_CASE("[Modules][Planner][Persona] Belief formation and information asymmetry") {
	Ref<PlannerPersona> persona_a = PlannerPersona::create_human("persona_a", "Alice");
	Ref<PlannerPersona> persona_b = PlannerPersona::create_ai("persona_b", "Bob");

	// Information asymmetry: Cannot directly access planner state
	Dictionary result = persona_b->get_planner_state("persona_b", "persona_a");
	CHECK(result.has("error"));
	CHECK(String(result["error"]) == "hidden");

	// But can form beliefs through observation
	Dictionary observation;
	observation["entity"] = "persona_b";
	observation["command"] = "movement";
	observation["confidence"] = 0.8;
	persona_a->process_observation(observation);

	Dictionary beliefs = persona_a->get_beliefs_about("persona_b");
	CHECK(beliefs.has("observed_movement"));
}

TEST_CASE("[Modules][Planner][BeliefManager] Register and manage personas") {
	Ref<PlannerBeliefManager> manager = memnew(PlannerBeliefManager);
	Ref<PlannerPersona> persona1 = PlannerPersona::create_human("p1", "Alice");
	Ref<PlannerPersona> persona2 = PlannerPersona::create_ai("p2", "Bob");

	manager->register_persona(persona1);
	manager->register_persona(persona2);

	CHECK(manager->has_persona("p1"));
	CHECK(manager->has_persona("p2"));
	CHECK(!manager->has_persona("p3"));

	Ref<PlannerPersona> retrieved = manager->get_persona("p1");
	CHECK(retrieved.is_valid());
	CHECK(retrieved->get_persona_id() == "p1");
}

TEST_CASE("[Modules][Planner][State] Shared ground truth") {
	Ref<PlannerState> facts = memnew(PlannerState);

	// Set terrain facts
	facts->set_terrain_fact("location_a", "type", "forest");
	facts->set_terrain_fact("location_a", "difficulty", 3);
	CHECK(facts->has_terrain_fact("location_a", "type"));
	CHECK(String(facts->get_terrain_fact("location_a", "type")) == "forest");

	// Add shared objects
	Dictionary object_data;
	object_data["type"] = "chest";
	object_data["location"] = "location_a";
	facts->add_shared_object("chest_1", object_data);
	CHECK(facts->has_shared_object("chest_1"));

	// Set entity positions (publicly observable)
	facts->set_entity_position("entity_1", Vector3(10, 5, 2));
	CHECK(facts->has_entity_position("entity_1"));

	// All personas can observe these facts
	Dictionary terrain = facts->observe_terrain("location_a");
	CHECK(terrain.has("type"));
	Dictionary objects = facts->observe_shared_objects("location_a");
	CHECK(objects.has("chest_1"));
}

TEST_CASE("[Modules][Planner][Persona] Integration with PlannerPlan") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerPersona> persona = PlannerPersona::create_human("p1", "Alice");
	Ref<PlannerBeliefManager> manager = memnew(PlannerBeliefManager);

	plan->set_current_persona(persona);
	plan->set_belief_manager(manager);

	CHECK(plan->get_current_persona().is_valid());
	CHECK(plan->get_belief_manager().is_valid());
}

} // namespace TestPersonaBelief
