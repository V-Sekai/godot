#pragma once

// SPDX-FileCopyrightText: 2021 University of Maryland
// SPDX-License-Identifier: BSD-3-Clause-Clear
// Author: Dana Nau <nau@umd.edu>, July 7, 2021

#include "core/io/resource.h"
#include "core/templates/hash_map.h"

#include "planner_persona.h"

class PlannerBeliefManager : public Resource {
	GDCLASS(PlannerBeliefManager, Resource);

	HashMap<String, Ref<PlannerPersona>> personas;

protected:
	static void _bind_methods();

public:
	PlannerBeliefManager();
	~PlannerBeliefManager();

	Ref<PlannerPersona> get_persona(const String &p_persona_id);
	bool has_persona(const String &p_persona_id) const;
	void register_persona(const Ref<PlannerPersona> &p_persona);
	void process_observation_for_persona(const String &p_persona_id, const Dictionary &p_observation);
	Dictionary get_planner_state(const String &p_target_persona_id, const String &p_requester_persona_id);
};
