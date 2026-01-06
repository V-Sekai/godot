/**************************************************************************/
/*  temporal_travel_test.h                                                */
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

#include "../../src/domain.h"
#include "../../src/plan.h"
#include "../../src/planner_result.h"
#include "../../src/planner_time_range.h"
#include "core/variant/callable.h"
#include "temporal_travel_domain.h"
#include "tests/test_macros.h"

namespace TestTemporalTravel {

// Helper: Create temporal travel domain with actions and task methods
Ref<PlannerDomain> create_temporal_travel_domain() {
	Ref<PlannerDomain> domain = memnew(PlannerDomain);

	// Add actions
	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&TemporalTravelDomainCallable::action_walk));
	actions.push_back(callable_mp_static(&TemporalTravelDomainCallable::action_call_taxi));
	actions.push_back(callable_mp_static(&TemporalTravelDomainCallable::action_ride_taxi));
	actions.push_back(callable_mp_static(&TemporalTravelDomainCallable::action_pay_driver));
	domain->add_actions(actions);

	// Add task methods for travel
	TypedArray<Callable> travel_methods;
	travel_methods.push_back(callable_mp_static(&TemporalTravelDomainCallable::task_travel_do_nothing));
	travel_methods.push_back(callable_mp_static(&TemporalTravelDomainCallable::task_travel_by_foot));
	travel_methods.push_back(callable_mp_static(&TemporalTravelDomainCallable::task_travel_by_taxi));
	domain->add_task_methods("travel", travel_methods);

	return domain;
}

// Helper: Create initial state for temporal travel problem
// Uses absolute time: 2025-01-01 10:00:00 UTC = 1735732800000000 microseconds
Dictionary create_temporal_travel_init_state() {
	Dictionary state;

	// Rigid relations
	Dictionary rigid;

	// Types
	Dictionary types;
	Array persons;
	persons.push_back("alice");
	persons.push_back("bob");
	types["person"] = persons;

	Array locations;
	locations.push_back("home_a");
	locations.push_back("home_b");
	locations.push_back("park");
	locations.push_back("station");
	locations.push_back("downtown");
	types["location"] = locations;

	Array taxis;
	taxis.push_back("taxi1");
	taxis.push_back("taxi2");
	types["taxi"] = taxis;

	rigid["types"] = types;

	// Distances (using string keys for tuple pairs)
	Dictionary dist;
	dist["(home_a, park)"] = 8;
	dist["(home_b, park)"] = 2;
	dist["(station, home_a)"] = 1;
	dist["(station, home_b)"] = 7;
	dist["(downtown, home_a)"] = 3;
	dist["(downtown, home_b)"] = 7;
	dist["(station, downtown)"] = 2;
	rigid["dist"] = dist;

	state["rigid"] = rigid;

	// Locations
	Dictionary loc;
	loc["alice"] = "home_a";
	loc["bob"] = "home_b";
	loc["taxi1"] = "park";
	loc["taxi2"] = "station";
	state["loc"] = loc;

	// Cash
	Dictionary cash;
	cash["alice"] = 20.0;
	cash["bob"] = 15.0;
	state["cash"] = cash;

	// Owe (initially 0)
	Dictionary owe;
	owe["alice"] = 0.0;
	owe["bob"] = 0.0;
	state["owe"] = owe;

	return state;
}

TEST_CASE("[Modules][Planner][TemporalTravel] Alice travels to park") {
	Ref<PlannerDomain> domain = create_temporal_travel_domain();
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	Dictionary state = create_temporal_travel_init_state();

	// Set initial time: 2025-01-01 10:00:00 UTC = 1735732800000000 microseconds
	int64_t initial_time = 1735732800000000LL;
	PlannerTimeRange time_range;
	time_range.set_start_time(initial_time);
	plan->set_time_range(time_range);

	// Task: Alice travels to park
	Array todo_list;
	Array task;
	task.push_back("travel");
	task.push_back("alice");
	task.push_back("park");
	todo_list.push_back(task);

	// Attach temporal durations to actions (using absolute time in microseconds)
	// a_walk: 5 minutes per unit = 300000000 microseconds per unit
	// a_call_taxi: instant = 0 microseconds
	// a_ride_taxi: 10 minutes per unit = 600000000 microseconds per unit
	// a_pay_driver: instant = 0 microseconds

	// For taxi travel from home_a to park (distance 8):
	// - call_taxi: 0 microseconds
	// - ride_taxi: 8 * 600000000 = 4800000000 microseconds (80 minutes)
	// - pay_driver: 0 microseconds

	Dictionary call_taxi_temporal;
	call_taxi_temporal["duration"] = static_cast<int64_t>(0LL);
	Array call_taxi_action;
	call_taxi_action.push_back("action_call_taxi");
	call_taxi_action.push_back("alice");
	call_taxi_action.push_back("home_a");
	plan->attach_metadata(call_taxi_action, call_taxi_temporal);

	Dictionary ride_taxi_temporal;
	ride_taxi_temporal["duration"] = static_cast<int64_t>(4800000000LL); // 8 * 600000000
	Array ride_taxi_action;
	ride_taxi_action.push_back("action_ride_taxi");
	ride_taxi_action.push_back("alice");
	ride_taxi_action.push_back("park");
	plan->attach_metadata(ride_taxi_action, ride_taxi_temporal);

	Dictionary pay_driver_temporal;
	pay_driver_temporal["duration"] = static_cast<int64_t>(0LL);
	Array pay_driver_action;
	pay_driver_action.push_back("action_pay_driver");
	pay_driver_action.push_back("alice");
	pay_driver_action.push_back("park");
	plan->attach_metadata(pay_driver_action, pay_driver_temporal);

	Ref<PlannerResult> result = plan->find_plan(state, todo_list);

	CHECK(result.is_valid());
	CHECK(result->get_success());

	Array plan_result = result->extract_plan();
	CHECK(plan_result.size() == 3); // call_taxi, ride_taxi, pay_driver

	// Verify expected actions
	CHECK(plan_result[0].get_type() == Variant::ARRAY);
	Array action1 = plan_result[0];
	CHECK(action1[0] == "action_call_taxi");

	CHECK(plan_result[1].get_type() == Variant::ARRAY);
	Array action2 = plan_result[1];
	CHECK(action2[0] == "action_ride_taxi");

	CHECK(plan_result[2].get_type() == Variant::ARRAY);
	Array action3 = plan_result[2];
	CHECK(action3[0] == "action_pay_driver");

	// Verify final state
	Dictionary final_state = result->get_final_state();
	Dictionary final_loc = final_state["loc"];
	CHECK(String(final_loc["alice"]) == "park");

	Dictionary final_cash = final_state["cash"];
	double alice_cash = double(final_cash["alice"]);
	// Fare = 1.5 + 0.5 * 8 = 5.5, so alice should have 20 - 5.5 = 14.5
	CHECK(alice_cash == doctest::Approx(14.5));
}

TEST_CASE("[Modules][Planner][TemporalTravel] Alice and Bob both travel to park") {
	Ref<PlannerDomain> domain = create_temporal_travel_domain();
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	Dictionary state = create_temporal_travel_init_state();

	// Set initial time
	int64_t initial_time = 1735732800000000LL;
	PlannerTimeRange time_range;
	time_range.set_start_time(initial_time);
	plan->set_time_range(time_range);

	// Tasks: Alice and Bob both travel to park
	Array todo_list;

	Array task1;
	task1.push_back("travel");
	task1.push_back("alice");
	task1.push_back("park");
	todo_list.push_back(task1);

	Array task2;
	task2.push_back("travel");
	task2.push_back("bob");
	task2.push_back("park");
	todo_list.push_back(task2);

	// Attach temporal durations
	// Alice: taxi (distance 8) - call(0), ride(4800000000), pay(0)
	// Bob: walk (distance 2) - walk(2 * 300000000 = 600000000)

	Dictionary call_taxi_temporal;
	call_taxi_temporal["duration"] = static_cast<int64_t>(0LL);
	Array call_taxi_action;
	call_taxi_action.push_back("action_call_taxi");
	call_taxi_action.push_back("alice");
	call_taxi_action.push_back("home_a");
	plan->attach_metadata(call_taxi_action, call_taxi_temporal);

	Dictionary ride_taxi_temporal;
	ride_taxi_temporal["duration"] = static_cast<int64_t>(4800000000LL);
	Array ride_taxi_action;
	ride_taxi_action.push_back("action_ride_taxi");
	ride_taxi_action.push_back("alice");
	ride_taxi_action.push_back("park");
	plan->attach_metadata(ride_taxi_action, ride_taxi_temporal);

	Dictionary pay_driver_temporal;
	pay_driver_temporal["duration"] = static_cast<int64_t>(0LL);
	Array pay_driver_action;
	pay_driver_action.push_back("action_pay_driver");
	pay_driver_action.push_back("alice");
	pay_driver_action.push_back("park");
	plan->attach_metadata(pay_driver_action, pay_driver_temporal);

	Dictionary walk_temporal;
	walk_temporal["duration"] = static_cast<int64_t>(600000000LL); // 2 * 300000000
	Array walk_action;
	walk_action.push_back("action_walk");
	walk_action.push_back("bob");
	walk_action.push_back("home_b");
	walk_action.push_back("park");
	plan->attach_metadata(walk_action, walk_temporal);

	Ref<PlannerResult> result = plan->find_plan(state, todo_list);

	CHECK(result.is_valid());
	CHECK(result->get_success());

	Array plan_result = result->extract_plan();
	CHECK(plan_result.size() >= 4); // At least 4 actions (Alice's taxi + Bob's travel)

	// Verify Alice's actions (taxi travel)
	bool found_alice_call = false;
	bool found_alice_ride = false;
	bool found_alice_pay = false;
	bool found_bob_travel = false;

	for (int i = 0; i < plan_result.size(); i++) {
		Array action = plan_result[i];
		if (action.size() > 0) {
			String action_name = action[0];
			if (action_name == "action_call_taxi" && action.size() > 1 && String(action[1]) == "alice") {
				found_alice_call = true;
			} else if (action_name == "action_ride_taxi" && action.size() > 1 && String(action[1]) == "alice") {
				found_alice_ride = true;
			} else if (action_name == "action_pay_driver" && action.size() > 1 && String(action[1]) == "alice") {
				found_alice_pay = true;
			} else if ((action_name == "action_walk" || action_name == "action_call_taxi") && action.size() > 1 && String(action[1]) == "bob") {
				found_bob_travel = true;
			}
		}
	}

	CHECK(found_alice_call);
	CHECK(found_alice_ride);
	CHECK(found_alice_pay);
	CHECK(found_bob_travel);

	// Verify final state
	Dictionary final_state = result->get_final_state();
	Dictionary final_loc = final_state["loc"];
	CHECK(String(final_loc["alice"]) == "park");
	CHECK(String(final_loc["bob"]) == "park");
}

} // namespace TestTemporalTravel
