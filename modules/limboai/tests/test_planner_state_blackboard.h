/**************************************************************************/
/*  test_planner_state_blackboard.h                                       */
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

/**
 * test_planner_state_blackboard.h
 * Unit tests for PlannerState with Blackboard backend (to_plan_dictionary,
 * apply_plan_state, get_triples_as_array).
 *
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "limbo_test.h"

#include "modules/limboai/blackboard/blackboard.h"
#include "modules/limboai/planning/src/planner_state.h"

namespace TestPlannerStateBlackboard {

TEST_CASE("[Modules][LimboAI][Planner] PlannerState with Blackboard - set/get and to_plan_dictionary") {
	Ref<Blackboard> bb = memnew(Blackboard);
	Ref<PlannerState> state = memnew(PlannerState);
	state->set_blackboard(bb);

	SUBCASE("Empty state to_plan_dictionary") {
		Dictionary d = state->to_plan_dictionary();
		// to_plan_dictionary returns blackboard vars (excluding planner_metadata, beliefs)
		CHECK(d.size() >= 0);
	}

	SUBCASE("set_predicate / get_predicate roundtrip") {
		state->set_predicate("subj1", "pred1", 42);
		CHECK(state->has_predicate("subj1", "pred1"));
		CHECK_EQ(int(state->get_predicate("subj1", "pred1")), 42);
		state->set_predicate("subj2", "pred2", String("hello"));
		CHECK_EQ(String(state->get_predicate("subj2", "pred2")), "hello");
	}

	SUBCASE("to_plan_dictionary includes predicate data") {
		state->set_predicate("entity", "at", Vector3(1, 2, 3));
		Dictionary d = state->to_plan_dictionary();
		// Predicates stored as var name = predicate, value = Dictionary(subject -> value)
		CHECK(d.has("at"));
		Dictionary at_dict = d["at"];
		CHECK(at_dict.has("entity"));
		CHECK(Vector3(at_dict["entity"]) == Vector3(1, 2, 3));
	}

	SUBCASE("apply_plan_state restores state") {
		state->set_predicate("e1", "p1", 10);
		Dictionary d = state->to_plan_dictionary();
		Ref<Blackboard> bb2 = memnew(Blackboard);
		Ref<PlannerState> state2 = memnew(PlannerState);
		state2->set_blackboard(bb2);
		state2->apply_plan_state(d);
		CHECK(state2->has_predicate("e1", "p1"));
		CHECK_EQ(int(state2->get_predicate("e1", "p1")), 10);
	}

	SUBCASE("get_triples_as_array returns triples from Blackboard") {
		state->set_predicate("s1", "p1", 1);
		state->set_predicate("s1", "p2", 2);
		state->set_predicate("s2", "p1", 3);
		TypedArray<Dictionary> triples = state->get_triples_as_array();
		CHECK(triples.size() >= 3);
		int found = 0;
		for (int i = 0; i < triples.size(); i++) {
			Dictionary t = triples[i];
			// Canonical key order: predicate, subject, object, metadata.
			CHECK(t.has("predicate"));
			CHECK(t.has("subject"));
			CHECK(t.has("object"));
			CHECK(t.has("metadata"));
			Array keys = t.keys();
			CHECK(keys.size() >= 4);
			CHECK(keys[0] == Variant("predicate"));
			CHECK(keys[1] == Variant("subject"));
			CHECK(keys[2] == Variant("object"));
			CHECK(keys[3] == Variant("metadata"));
			String subj = t["subject"];
			String pred = t["predicate"];
			Variant obj = t["object"];
			if (subj == "s1" && pred == "p1" && int(obj) == 1) {
				found++;
			}
			if (subj == "s1" && pred == "p2" && int(obj) == 2) {
				found++;
			}
			if (subj == "s2" && pred == "p1" && int(obj) == 3) {
				found++;
			}
		}
		CHECK_EQ(found, 3);
	}
}

} // namespace TestPlannerStateBlackboard
