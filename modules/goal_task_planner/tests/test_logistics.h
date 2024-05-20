/**************************************************************************/
/*  test_logistics.h                                                      */
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

#ifndef TEST_LOGISTICS_H
#define TEST_LOGISTICS_H

// SPDX-FileCopyrightText: 2021 University of Maryland
// SPDX-License-Identifier: BSD-3-Clause-Clear
// Author: Dana Nau <nau@umd.edu>, July 7, 2021

#include "core/variant/array.h"
#include "core/variant/callable.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"
#include "tests/test_macros.h"

#include "modules/goal_task_planner/domain.h"
#include "modules/goal_task_planner/multigoal.h"
#include "modules/goal_task_planner/plan.h"

namespace TestLogistics {

// This file is based on the logistics-domain examples included with HGNpyhop:
//	https://github.com/ospur/hgn-pyhop
//
// For a discussion of the adaptations that were needed, see the relevant
// section of Some_GTPyhop_Details.md in the top-level directory.
//
// -- Dana Nau <nau@umd.edu>, July 20, 2021

// Actions

static Dictionary drive_truck(Dictionary p_state, String p_truck, String p_location) {
	Dictionary truck_at = p_state["truck_at"];
	truck_at[p_truck] = p_location;
	p_state["truck_at"] = truck_at;
	return p_state;
}

static Dictionary fly_plane(Dictionary p_state, String p_plane, String p_airport) {
	Dictionary plane_at = p_state["plane_at"];
	plane_at[p_plane] = p_airport;
	p_state["plane_at"] = plane_at;
	return p_state;
}

static Dictionary load_truck(Dictionary p_state, String p_object, String p_truck) {
	Dictionary at = p_state["at"];
	at[p_object] = p_truck;
	p_state["at"] = at;
	return p_state;
}

static Dictionary load_plane(Dictionary p_state, String p_object, String p_plane) {
	Dictionary at = p_state["at"];
	at[p_object] = p_plane;
	p_state["at"] = at;
	return p_state;
}

static Dictionary unload_plane(Dictionary p_state, String p_object, String p_airport) {
	Dictionary at = p_state["at"];
	Variant plane = at[p_object];
	Dictionary plane_at = p_state["plane_at"];
	if (plane_at[plane] == p_airport) {
		at[p_object] = p_airport;
		p_state["at"] = at;
	}
	return p_state;
}

static Dictionary unload_truck(Dictionary p_state, String p_object, String p_location) {
	Dictionary at = p_state["at"];
	Variant truck = at[p_object];
	Dictionary truck_at = p_state["truck_at"];
	if (truck_at[truck] == p_location) {
		at[p_object] = p_location;
		p_state["at"] = at;
	}
	return p_state;
}

// Helper functions for the methods.

// Find a truck in the same city as the package.
static Variant _find_truck(Dictionary p_state, String p_object) {
	Array trucks = p_state["trucks"];

	Dictionary truck_at = p_state["truck_at"];
	Dictionary in_city = p_state["in_city"];
	Dictionary at = p_state["at"];
	for (String truck : trucks) {
		if (in_city[truck_at[truck]] == in_city[at[p_object]]) {
			return truck;
		}
	}
	return false;
}

// Find a plane in the same city as the package; if none available, find a random plane
static Variant _find_plane(Dictionary p_state, String p_object) {
	Array airplanes = p_state["airplanes"];
	Dictionary in_city = p_state["in_city"];
	Dictionary plane_at = p_state["plane_at"];
	Dictionary at = p_state["at"];
	Variant last_plane;
	for (Variant plane : airplanes) {
		if (in_city[plane_at[plane]] == in_city[at[p_object]]) {
			return plane;
		}
		last_plane = plane;
	}
	return last_plane;
}

//  Find an airport in the same city as the location.
static Variant _find_airport(Dictionary p_state, String p_location) {
	Array airports = p_state["airports"];
	Dictionary in_city = p_state["in_city"];
	for (Variant airport : airports) {
		if (in_city[airport] == in_city[p_location]) {
			return airport;
		}
	}
	return false;
}

static Variant method_drive_truck(Dictionary p_state, String p_truck, String p_location) {
	Array trucks = p_state["trucks"];
	Array locations = p_state["locations"];

	Dictionary in_city = p_state["in_city"];
	Dictionary truck_at = p_state["truck_at"];
	if (trucks.has(p_truck) && locations.has(p_location) && in_city[truck_at[p_truck]] == in_city[p_location]) {
		Array plan;
		plan.push_back(varray("drive_truck", p_truck, p_location));
		return plan;
	}
	return false;
}

static Variant method_load_truck(Dictionary p_state, String p_object, String p_truck) {
	Array packages = p_state["packages"];
	Array trucks = p_state["trucks"];
	Dictionary at = p_state["at"];
	Dictionary truck_at = p_state["truck_at"];
	if (packages.has(p_object) && trucks.has(p_truck) && at[p_object] == truck_at[p_truck]) {
		Array plan;
		plan.push_back(varray("load_truck", p_object, p_truck));
		return plan;
	}
	return false;
}

static Variant method_unload_truck(Dictionary p_state, String p_object, String p_location) {
	Array packages = p_state["packages"];
	Array trucks = p_state["trucks"];
	Array locations = p_state["locations"];
	Dictionary at = p_state["at"];
	if (packages.has(p_object) && trucks.has(at[p_object]) && locations.has(p_location)) {
		Array plan;
		plan.push_back(varray("unload_truck", p_object, p_location));
		return plan;
	}
	return false;
}

static Variant method_fly_plane(Dictionary p_state, String p_plane, String p_airport) {
	Array airplanes = p_state["airplanes"];
	Array airports = p_state["airports"];

	if (airplanes.has(p_plane) && airports.has(p_airport)) {
		Array plan;
		plan.push_back(varray("fly_plane", p_plane, p_airport));
		return plan;
	}
	return false;
}

static Variant method_load_plane(Dictionary p_state, String p_object, String p_plane) {
	Array packages = p_state["packages"];
	Array airplanes = p_state["airplanes"];
	Dictionary at = p_state["at"];
	Dictionary plane_at = p_state["plane_at"];
	if (packages.has(p_object) && airplanes.has(p_plane) && at[p_object] == plane_at[p_plane]) {
		Array plan;
		plan.push_back(varray("load_plane", p_object, p_plane));
		return plan;
	}

	return false;
}

static Variant method_unload_plane(Dictionary p_state, String p_object, String p_airport) {
	Array packages = p_state["packages"];
	Array airplanes = p_state["airplanes"];

	Dictionary at = p_state["at"];
	Array airports = p_state["airports"];

	bool package_exists = packages.has(p_object);
	bool airplane_exists = airplanes.has(at[p_object]);
	bool airport_exists = airports.has(p_airport);

	if (package_exists && airplane_exists && airport_exists) {
		Array plan;
		plan.push_back(varray("unload_plane", p_object, p_airport));
		return plan;
	}

	return false;
}

// Other methods

static Variant method_move_within_city(Dictionary p_state, String p_object, String p_location) {
	Array packages = p_state["packages"];
	Array locations = p_state["locations"];
	Dictionary in_city = p_state["in_city"];
	Dictionary at = p_state["at"];

	bool package_exists = packages.has(p_object);
	bool current_location_exists = locations.has(at[p_object]);
	bool same_city = in_city[at[p_object]] == in_city[p_location];

	if (package_exists && current_location_exists && same_city) {
		Array plan;
		Variant truck = _find_truck(p_state, p_object);

		bool truck_available = truck;

		if (truck_available) {
			plan.push_back(varray("truck_at", truck, at[p_object]));
			plan.push_back(varray("at", p_object, truck));
			plan.push_back(varray("truck_at", truck, p_location));
			plan.push_back(varray("at", p_object, p_location));
			return plan;
		}
	}
	return false;
}

Variant method_move_between_airports(Dictionary p_state, String p_object, String p_airport) {
	Array packages = p_state["packages"];
	Array airports = p_state["airports"];
	Dictionary in_city = p_state["in_city"];
	Dictionary at = p_state["at"];

	bool package_exists = packages.has(p_object);
	bool current_location_is_airport = airports.has(at[p_object]);
	bool destination_airport_exists = airports.has(p_airport);
	bool different_cities = in_city[at[p_object]] != in_city[p_airport];

	if (package_exists && current_location_is_airport && destination_airport_exists && different_cities) {
		Variant plane = _find_plane(p_state, p_object);

		bool plane_available = plane;

		if (plane_available) {
			Array plan;
			plan.push_back(varray("plane_at", plane, at[p_object]));
			plan.push_back(varray("at", p_object, plane));
			plan.push_back(varray("plane_at", plane, p_airport));
			plan.push_back(varray("at", p_object, p_airport));
			return plan;
		}
	}
	return false;
}

Variant method_move_between_city(Dictionary p_state, String p_object, String p_location) {
	Array packages = p_state["packages"];
	Array locations = p_state["locations"];
	Dictionary in_city = p_state["in_city"];
	Dictionary at = p_state["at"];

	bool does_package_exist = packages.has(p_object);
	bool does_location_exist = locations.has(at[p_object]);
	bool is_different_city = in_city[at[p_object]] != in_city[p_location];

	if (does_package_exist && does_location_exist && is_different_city) {
		Variant airport_1 = _find_airport(p_state, at[p_object]);
		Variant airport_2 = _find_airport(p_state, p_location);

		bool are_airports_valid = airport_1 != Variant(false) && airport_2 != Variant(false);

		if (are_airports_valid) {
			Array plan;
			plan.push_back(varray("at", p_object, airport_1));
			plan.push_back(varray("at", p_object, airport_2));
			plan.push_back(varray("at", p_object, p_location));
			return plan;
		}
	}

	return false;
}

void before_each(Dictionary &p_state, Ref<Plan> p_planner, Ref<Domain> p_the_domain) {
	ERR_FAIL_COND(p_planner.is_null());
	ERR_FAIL_COND(p_the_domain.is_null());
	p_planner->set_verbose(0);
	p_state.clear();
	TypedArray<Domain> domains;
	domains.push_back(p_the_domain);
	p_planner->set_domains(domains);

	p_planner->set_current_domain(p_the_domain);
	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&drive_truck));
	actions.push_back(callable_mp_static(&load_truck));
	actions.push_back(callable_mp_static(&unload_truck));
	actions.push_back(callable_mp_static(&fly_plane));
	actions.push_back(callable_mp_static(&load_plane));
	actions.push_back(callable_mp_static(&unload_plane));
	p_the_domain->add_actions(actions);

	TypedArray<Callable> truck_at_methods;
	truck_at_methods.push_back(callable_mp_static(&method_drive_truck));
	p_the_domain->add_unigoal_methods("truck_at", truck_at_methods);

	TypedArray<Callable> plane_at_methods;
	plane_at_methods.push_back(callable_mp_static(&method_fly_plane));
	p_the_domain->add_unigoal_methods("plane_at", plane_at_methods);

	TypedArray<Callable> at_methods;
	at_methods.push_back(callable_mp_static(&method_load_truck));
	at_methods.push_back(callable_mp_static(&method_unload_truck));
	at_methods.push_back(callable_mp_static(&method_load_plane));
	at_methods.push_back(callable_mp_static(&method_unload_plane));
	p_the_domain->add_unigoal_methods("at", at_methods);

	TypedArray<Callable> move_methods;
	at_methods.push_back(callable_mp_static(&method_move_within_city));
	at_methods.push_back(callable_mp_static(&method_move_between_airports));
	at_methods.push_back(callable_mp_static(&method_move_between_city));
	p_the_domain->add_unigoal_methods("at", move_methods);

	Array packages;
	packages.push_back("package1");
	packages.push_back("package2");
	p_state["packages"] = packages;
	Array trucks;
	trucks.push_back("truck1");
	trucks.push_back("truck6");
	p_state["trucks"] = trucks;
	Array airplanes;
	airplanes.push_back("plane2");
	p_state["airplanes"] = airplanes;
	Array locations;
	locations.push_back("airport1");
	locations.push_back("location1");
	locations.push_back("location2");
	locations.push_back("location3");
	locations.push_back("airport2");
	locations.push_back("location10");
	p_state["locations"] = locations;
	Array airports;
	airports.push_back("airport1");
	airports.push_back("airport2");
	p_state["airports"] = airports;
	Array cities;
	cities.push_back("city1");
	cities.push_back("city2");
	p_state["cities"] = cities;

	Dictionary at;
	at["package1"] = "location1";
	at["package2"] = "location2";
	p_state["at"] = at;

	Dictionary truck_at;
	truck_at["truck1"] = "location3";
	truck_at["truck6"] = "location10";
	p_state["truck_at"] = truck_at;

	Dictionary plane_at;
	plane_at["plane2"] = "airport2";
	p_state["plane_at"] = plane_at;

	Dictionary in_city;
	in_city["airport1"] = "city1";
	in_city["location1"] = "city1";
	in_city["location2"] = "city1";
	in_city["location3"] = "city1";
	in_city["airport2"] = "city2";
	in_city["location10"] = "city2";
	p_state["in_city"] = in_city;
}

TEST_CASE("[Modules][GoalTaskPlanner] m_drive_truck") {
	Ref<Plan> planner;
	planner.instantiate();
	Ref<Domain> the_domain;
	the_domain.instantiate();
	Dictionary state1;
	before_each(state1, planner, the_domain);
	TypedArray<Array> task;
	task.push_back(varray("truck_at", "truck1", "location2"));
	Variant plan = planner->find_plan(state1, task);
	Array answer;
	answer.push_back(varray("drive_truck", "truck1", "location2"));
	CHECK_EQ(plan, answer);
}

TEST_CASE("[Modules][GoalTaskPlanner] Fly plane") {
	Ref<Plan> planner;
	planner.instantiate();
	Ref<Domain> the_domain;
	the_domain.instantiate();
	Dictionary state1;
	before_each(state1, planner, the_domain);
	TypedArray<Array> task;
	task.push_back(varray("plane_at", "plane2", "airport1"));
	Variant plan = planner->find_plan(state1, task);
	TypedArray<Array> answer;
	answer.push_back(varray("fly_plane", "plane2", "airport1"));
	CHECK_EQ(plan, answer);
}

TEST_CASE("[Modules][GoalTaskPlanner] Load truck") {
	Ref<Plan> planner;
	planner.instantiate();
	Ref<Domain> the_domain;
	the_domain.instantiate();
	Dictionary state1;
	before_each(state1, planner, the_domain);
	TypedArray<Array> task;
	task.push_back(varray("at", "package1", "truck6"));
	Variant plan = planner->find_plan(state1, task);
	TypedArray<Array> answer;
	CHECK_NE(plan, answer);
}

TEST_CASE("[Modules][GoalTaskPlanner] Move Goal 1") {
	Ref<Plan> planner;
	planner.instantiate();
	Ref<Domain> the_domain;
	the_domain.instantiate();
	Dictionary state1;
	before_each(state1, planner, the_domain);
	TypedArray<Array> task;
	task.push_back(varray("at", "package1", "location2"));
	task.push_back(varray("at", "package2", "location3"));
	Variant plan = planner->find_plan(
			state1,
			task);

	TypedArray<Array> answer;
	answer.push_back(varray("drive_truck", "truck1", "location1"));
	answer.push_back(varray("load_truck", "package1", "truck1"));
	answer.push_back(varray("drive_truck", "truck1", "location2"));
	answer.push_back(varray("unload_truck", "package1", "location2"));
	answer.push_back(varray("load_truck", "package2", "truck1"));
	answer.push_back(varray("drive_truck", "truck1", "location3"));
	answer.push_back(varray("unload_truck", "package2", "location3"));
	CHECK_EQ(plan, answer);
}

TEST_CASE("[Modules][GoalTaskPlanner] Move Goal 2") {
	Ref<Plan> planner;
	planner.instantiate();
	Ref<Domain> the_domain;
	the_domain.instantiate();
	Dictionary state1;
	before_each(state1, planner, the_domain);

	Array task;
	task.push_back(varray("at", "package1", "location10"));

	Variant plan = planner->find_plan(state1, task);

	TypedArray<Array> answer;
	answer.push_back(varray("drive_truck", "truck1", "location1"));
	answer.push_back(varray("load_truck", "package1", "truck1"));
	answer.push_back(varray("drive_truck", "truck1", "airport1"));
	answer.push_back(varray("unload_truck", "package1", "airport1"));
	answer.push_back(varray("fly_plane", "plane2", "airport1"));
	answer.push_back(varray("load_plane", "package1", "plane2"));
	answer.push_back(varray("fly_plane", "plane2", "airport2"));
	answer.push_back(varray("unload_plane", "package1", "airport2"));
	answer.push_back(varray("drive_truck", "truck6", "airport2"));
	answer.push_back(varray("load_truck", "package1", "truck6"));
	answer.push_back(varray("drive_truck", "truck6", "location10"));
	answer.push_back(varray("unload_truck", "package1", "location10"));
	CHECK_EQ(plan, answer);
}

TEST_CASE("[Modules][GoalTaskPlanner] Move Goal 3") {
	Ref<Plan> planner;
	planner.instantiate();
	Ref<Domain> the_domain;
	the_domain.instantiate();
	Dictionary state1;
	before_each(state1, planner, the_domain);

	Array task;
	task.push_back(varray("at", "package1", "location1"));

	Variant plan = planner->find_plan(state1, task);
	Array answer;
	CHECK_EQ(plan, answer);
}

TEST_CASE("[Modules][GoalTaskPlanner] Move Goal 4") {
	Ref<Plan> planner;
	planner.instantiate();
	Ref<Domain> the_domain;
	the_domain.instantiate();
	Dictionary state1;
	before_each(state1, planner, the_domain);
	Array task;
	task.push_back(varray("at", "package1", "location2"));
	Variant plan = planner->find_plan(state1, task);
	Array answer;
	answer.push_back(varray("drive_truck", "truck1", "location1"));
	answer.push_back(varray("load_truck", "package1", "truck1"));
	answer.push_back(varray("drive_truck", "truck1", "location2"));
	answer.push_back(varray("unload_truck", "package1", "location2"));
	CHECK_EQ(plan, answer);
}

TEST_CASE("[Modules][GoalTaskPlanner] run_lazy_lookahead") {
	Ref<Plan> planner;
	planner.instantiate();
	Ref<Domain> the_domain;
	the_domain.instantiate();
	Dictionary state;
	before_each(state, planner, the_domain);
	Array task;
	task.push_back(varray("at", "package1", "location2"));
	Dictionary final_state = planner->run_lazy_lookahead(state, task);

	Dictionary answer;
	answer["packages"] = varray("package1", "package2");
	answer["trucks"] = varray("truck1", "truck6");
	answer["airplanes"] = varray("plane2");
	answer["locations"] = varray("airport1", "location1", "location2", "location3", "airport2", "location10");
	answer["airports"] = varray("airport1", "airport2");
	answer["cities"] = varray("city1", "city2");

	Dictionary at;
	at["package1"] = "location2";
	at["package2"] = "location2";
	answer["at"] = at;

	Dictionary truck_at;
	truck_at["truck1"] = "location2";
	truck_at["truck6"] = "location10";
	answer["truck_at"] = truck_at;

	Dictionary plane_at;
	plane_at["plane2"] = "airport2";
	answer["plane_at"] = plane_at;

	Dictionary in_city;
	in_city["airport1"] = "city1";
	in_city["location1"] = "city1";
	in_city["location2"] = "city1";
	in_city["location3"] = "city1";
	in_city["airport2"] = "city2";
	in_city["location10"] = "city2";
	answer["in_city"] = in_city;

	CHECK_EQ(final_state, answer);
}

TEST_CASE("[Modules][GoalTaskPlanner] Multigoal") {
	Ref<Plan> planner;
	planner.instantiate();
	Ref<Domain> the_domain;
	the_domain.instantiate();
	Dictionary state;
	before_each(state, planner, the_domain);
	Ref<Multigoal> multi_goal;
	Dictionary goal_state = state.duplicate(true);
	Dictionary at = goal_state["at"];
	at["package1"] = "location2";
	Dictionary truck_at = goal_state["truck_at"];
	truck_at["truck1"] = "location1";
	goal_state["truck_at"] = truck_at;
	goal_state["at"] = at;
	multi_goal.instantiate("Multigoal", goal_state);
	Array task;
	task.push_back(multi_goal);
	Array plan = planner->find_plan(state, task);
	Array answer;
	answer.push_back(varray("drive_truck", "truck1", "location1"));
	answer.push_back(varray("load_truck", "package1", "truck1"));
	answer.push_back(varray("drive_truck", "truck1", "location2"));
	answer.push_back(varray("unload_truck", "package1", "location2"));
	CHECK_EQ(plan, answer);
}

} // namespace TestLogistics

#endif // TEST_LOGISTICS_H
