/**************************************************************************/
/*  test_stn_solver.h                                                     */
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

// C++ unit tests for STN (Simple Temporal Network) solver

#pragma once

#include "../stn_constraints.h"
#include "../stn_solver.h"
#include "tests/test_macros.h"

namespace TestSTNSolver {

TEST_CASE("[Modules][STN] PlannerSTNSolver basic functionality") {
	PlannerSTNSolver stn;

	SUBCASE("Empty STN is consistent") {
		CHECK(stn.is_consistent());
	}

	SUBCASE("Add time point") {
		int64_t idx = stn.add_time_point("point1");
		CHECK(idx >= 0);
		CHECK(stn.has_time_point("point1"));
		CHECK(stn.is_consistent());
	}

	SUBCASE("Get time points list") {
		stn.add_time_point("point1");
		stn.add_time_point("point2");
		Array points = stn.get_time_points();
		CHECK(points.size() == 2);
	}

	SUBCASE("Add constraint between time points") {
		stn.add_time_point("a");
		stn.add_time_point("b");

		// Add constraint: a -> b with min=10, max=20 microseconds
		bool success = stn.add_constraint("a", "b", 10LL, 20LL);
		CHECK(success);
		CHECK(stn.is_consistent());
		CHECK(stn.has_constraint("a", "b"));

		// Check reverse constraint was added automatically
		CHECK(stn.has_constraint("b", "a"));
	}

	SUBCASE("Get constraint") {
		stn.add_time_point("a");
		stn.add_time_point("b");
		stn.add_constraint("a", "b", 10LL, 20LL);

		PlannerSTNSolver::Constraint constraint = stn.get_constraint("a", "b");
		CHECK(constraint.min_distance == 10LL);
		CHECK(constraint.max_distance == 20LL);
	}

	SUBCASE("Get distance between points") {
		stn.add_time_point("a");
		stn.add_time_point("b");
		stn.add_constraint("a", "b", 10LL, 20LL);

		stn.check_consistency();
		int64_t distance = stn.get_distance("a", "b");
		CHECK(distance == 20LL); // Should be max distance
	}

	SUBCASE("Remove constraint") {
		stn.add_time_point("a");
		stn.add_time_point("b");
		stn.add_constraint("a", "b", 10LL, 20LL);
		CHECK(stn.has_constraint("a", "b"));

		bool removed = stn.remove_constraint("a", "b");
		CHECK(removed);
		CHECK(!stn.has_constraint("a", "b"));
	}
}

TEST_CASE("[Modules][STN] PlannerSTNSolver consistency checking") {
	PlannerSTNSolver stn;

	SUBCASE("Consistent constraints") {
		stn.add_time_point("a");
		stn.add_time_point("b");
		stn.add_time_point("c");

		// a -> b: 10-20
		stn.add_constraint("a", "b", 10LL, 20LL);
		// b -> c: 5-15
		stn.add_constraint("b", "c", 5LL, 15LL);

		stn.check_consistency();
		CHECK(stn.is_consistent());
	}

	SUBCASE("Inconsistent constraints (negative cycle)") {
		stn.add_time_point("a");
		stn.add_time_point("b");

		// Create a temporal paradox: a -> b: 10+, b -> a: 10+
		// This means b must be at least 10 after a, AND a must be at least 10 after b
		// This is impossible (negative cycle)
		stn.add_constraint("a", "b", 10LL, 10000LL);
		stn.add_constraint("b", "a", 10LL, 10000LL);

		stn.check_consistency();
		CHECK(!stn.is_consistent());
	}

	SUBCASE("Constraint intersection (tightening)") {
		stn.add_time_point("a");
		stn.add_time_point("b");

		// First constraint: 10-20
		stn.add_constraint("a", "b", 10LL, 20LL);

		// Second constraint: 15-25 (intersects with 10-20 to give 15-20)
		stn.add_constraint("a", "b", 15LL, 25LL);

		stn.check_consistency();
		CHECK(stn.is_consistent());

		PlannerSTNSolver::Constraint constraint = stn.get_constraint("a", "b");
		CHECK(constraint.min_distance == 15LL);
		CHECK(constraint.max_distance == 20LL);
	}

	SUBCASE("Empty intersection (inconsistent)") {
		stn.add_time_point("a");
		stn.add_time_point("b");

		// First constraint: 10-20
		stn.add_constraint("a", "b", 10LL, 20LL);

		// Second constraint: 25-30 (no intersection with 10-20)
		bool success = stn.add_constraint("a", "b", 25LL, 30LL);
		CHECK(!success); // Should fail due to empty intersection
		CHECK(!stn.is_consistent());
	}
}

TEST_CASE("[Modules][STN] PlannerSTNConstraints interval conversion") {
	PlannerSTNSolver stn;

	SUBCASE("Add interval with duration") {
		int64_t start_time = 1000000LL; // 1 second
		int64_t end_time = 2000000LL; // 2 seconds
		int64_t duration = 1000000LL; // 1 second

		bool success = PlannerSTNConstraints::add_interval(stn, "action1", start_time, end_time, duration);
		CHECK(success);
		CHECK(stn.has_time_point("action1_start"));
		CHECK(stn.has_time_point("action1_end"));
		CHECK(stn.is_consistent());
	}

	SUBCASE("Add durative action") {
		int64_t duration = 500000LL; // 0.5 seconds

		bool success = PlannerSTNConstraints::add_durative_action(stn, "action1", duration);
		CHECK(success);
		CHECK(stn.has_time_point("action1_start"));
		CHECK(stn.has_time_point("action1_end"));

		// Check duration constraint
		PlannerSTNSolver::Constraint constraint = stn.get_constraint("action1_start", "action1_end");
		CHECK(constraint.min_distance == duration);
		CHECK(constraint.max_distance == duration);
	}

	SUBCASE("Anchor to origin") {
		stn.add_time_point("origin");
		int64_t absolute_time = 1735689600000000LL;

		bool success = PlannerSTNConstraints::anchor_to_origin(stn, "point1", absolute_time);
		CHECK(success);
		CHECK(stn.has_time_point("point1"));

		// Check constraint from origin to point1
		PlannerSTNSolver::Constraint constraint = stn.get_constraint("origin", "point1");
		CHECK(constraint.min_distance == absolute_time);
		CHECK(constraint.max_distance == absolute_time);
	}

	SUBCASE("Temporal relation: before") {
		PlannerSTNConstraints::add_durative_action(stn, "action1", 1000000LL);
		PlannerSTNConstraints::add_durative_action(stn, "action2", 1000000LL);

		// action1 happens before action2
		bool success = PlannerSTNConstraints::add_temporal_relation(stn, "action1", "action2", "before");
		CHECK(success);
		CHECK(stn.is_consistent());

		// Check that action1_end <= action2_start
		int64_t distance = stn.get_distance("action1_end", "action2_start");
		CHECK(distance >= 0); // Should be non-negative
	}

	SUBCASE("Temporal relation: during") {
		PlannerSTNConstraints::add_durative_action(stn, "action1", 1000000LL);
		PlannerSTNConstraints::add_durative_action(stn, "action2", 5000000LL);

		// action1 happens during action2
		bool success = PlannerSTNConstraints::add_temporal_relation(stn, "action1", "action2", "during");
		CHECK(success);
		CHECK(stn.is_consistent());
	}
}

TEST_CASE("[Modules][STN] PlannerSTNSolver snapshot and restore") {
	PlannerSTNSolver stn;

	SUBCASE("Create and restore snapshot") {
		// Set up initial STN
		stn.add_time_point("a");
		stn.add_time_point("b");
		stn.add_constraint("a", "b", 10LL, 20LL);

		// Create snapshot
		PlannerSTNSolver::Snapshot snapshot = stn.create_snapshot();

		// Modify STN
		stn.add_time_point("c");
		stn.add_constraint("b", "c", 5LL, 15LL);

		// Restore snapshot
		stn.restore_snapshot(snapshot);

		// Verify STN is back to original state
		CHECK(stn.has_time_point("a"));
		CHECK(stn.has_time_point("b"));
		CHECK(!stn.has_time_point("c")); // Should not exist after restore
		CHECK(stn.has_constraint("a", "b"));
		CHECK(!stn.has_constraint("b", "c")); // Should not exist after restore
	}

	SUBCASE("Snapshot preserves consistency") {
		stn.add_time_point("a");
		stn.add_time_point("b");
		stn.add_constraint("a", "b", 10LL, 20LL);
		stn.check_consistency();

		PlannerSTNSolver::Snapshot snapshot = stn.create_snapshot();

		// Make STN inconsistent
		stn.add_constraint("b", "a", 10LL, 10LL); // Creates negative cycle
		stn.check_consistency();
		CHECK(!stn.is_consistent());

		// Restore snapshot
		stn.restore_snapshot(snapshot);
		stn.check_consistency();
		CHECK(stn.is_consistent());
	}

	SUBCASE("Multiple snapshots") {
		stn.add_time_point("a");
		PlannerSTNSolver::Snapshot snapshot1 = stn.create_snapshot();

		stn.add_time_point("b");
		PlannerSTNSolver::Snapshot snapshot2 = stn.create_snapshot();

		stn.add_time_point("c");

		// Restore to snapshot2
		stn.restore_snapshot(snapshot2);
		CHECK(stn.has_time_point("a"));
		CHECK(stn.has_time_point("b"));
		CHECK(!stn.has_time_point("c"));

		// Restore to snapshot1
		stn.restore_snapshot(snapshot1);
		CHECK(stn.has_time_point("a"));
		CHECK(!stn.has_time_point("b"));
		CHECK(!stn.has_time_point("c"));
	}
}

TEST_CASE("[Modules][STN] Complex temporal scenarios") {
	PlannerSTNSolver stn;

	SUBCASE("Multiple actions with temporal constraints") {
		// Add three actions with different durations
		PlannerSTNConstraints::add_durative_action(stn, "action1", 1000000LL); // 1 second
		PlannerSTNConstraints::add_durative_action(stn, "action2", 2000000LL); // 2 seconds
		PlannerSTNConstraints::add_durative_action(stn, "action3", 500000LL); // 0.5 seconds

		// action1 before action2
		PlannerSTNConstraints::add_temporal_relation(stn, "action1", "action2", "before");

		// action2 before action3
		PlannerSTNConstraints::add_temporal_relation(stn, "action2", "action3", "before");

		stn.check_consistency();
		CHECK(stn.is_consistent());

		// Verify temporal ordering
		int64_t dist1_2 = stn.get_distance("action1_end", "action2_start");
		int64_t dist2_3 = stn.get_distance("action2_end", "action3_start");
		CHECK(dist1_2 >= 0);
		CHECK(dist2_3 >= 0);
	}

	SUBCASE("Action with absolute time anchoring") {
		int64_t origin_time = 1735689600000000LL; // Unix epoch anchor
		stn.add_time_point("origin");
		PlannerSTNConstraints::anchor_to_origin(stn, "origin", origin_time);

		// Add action with absolute times
		int64_t action_start = 1735689601000000LL; // 1 second after origin
		int64_t action_end = 1735689602000000LL; // 2 seconds after origin
		int64_t duration = 1000000LL;

		bool success = PlannerSTNConstraints::add_interval(stn, "action1", action_start, action_end, duration);
		CHECK(success);
		CHECK(stn.is_consistent());

		// Verify absolute timing constraints
		int64_t origin_to_start = stn.get_distance("origin", "action1_start");
		CHECK(origin_to_start == action_start);
	}

	SUBCASE("Temporal conflict detection") {
		PlannerSTNConstraints::add_durative_action(stn, "action1", 1000000LL);
		PlannerSTNConstraints::add_durative_action(stn, "action2", 1000000LL);

		// Create conflicting constraints:
		// action1 ends before action2 starts (action1_end <= action2_start)
		// action2 ends before action1 starts (action2_end <= action1_start)
		// This creates a conflict if both actions have duration

		// Add first constraint
		PlannerSTNConstraints::add_temporal_relation(stn, "action1", "action2", "before");

		// Add conflicting constraint (action2 before action1)
		PlannerSTNConstraints::add_temporal_relation(stn, "action2", "action1", "before");

		stn.check_consistency();
		// This should be inconsistent due to the conflict
		CHECK(!stn.is_consistent());
	}

	SUBCASE("Earliest and latest time queries") {
		stn.add_time_point("origin");
		PlannerSTNConstraints::anchor_to_origin(stn, "origin", 0LL);

		PlannerSTNConstraints::add_durative_action(stn, "action1", 1000000LL);

		// action1 starts at least 500ms after origin
		stn.add_constraint("origin", "action1_start", 500000LL, 500000LL);

		stn.check_consistency();
		CHECK(stn.is_consistent());

		// Query earliest time for action1_start
		int64_t earliest = stn.get_earliest_time("action1_start");
		CHECK(earliest >= 500000LL);

		// Query latest time (should be at least as large as earliest)
		int64_t latest = stn.get_latest_time("action1_start");
		CHECK(latest >= earliest);
	}
}

TEST_CASE("[Modules][STN] Clear and reset") {
	PlannerSTNSolver stn;

	SUBCASE("Clear STN") {
		stn.add_time_point("a");
		stn.add_time_point("b");
		stn.add_constraint("a", "b", 10LL, 20LL);

		CHECK(stn.get_time_points().size() > 0);

		stn.clear();

		CHECK(stn.get_time_points().size() == 0);
		CHECK(stn.is_consistent());
	}
}

} //namespace TestSTNSolver
