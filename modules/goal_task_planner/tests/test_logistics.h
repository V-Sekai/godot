#ifndef TEST_LOGISTICS_H
#define TEST_LOGISTICS_H

// Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
// K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
// SPDX-License-Identifier: MIT

// SPDX-FileCopyrightText: 2021 University of Maryland
// SPDX-License-Identifier: BSD-3-Clause-Clear
// Author: Dana Nau <nau@umd.edu>, July 7, 2021

#include "core/error/error_macros.h"
#include "core/variant/array.h"
#include "tests/test_macros.h"

#include "core/variant/callable.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"
#include "core/variant/variant.h"

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

static Dictionary load_truck(Dictionary p_state, String p_object, String p_truck) {
	Dictionary at = p_state["at"];
	at[p_object] = p_truck;
	p_state["at"] = at;
	return p_state;
}

static Variant unload_truck(Dictionary p_state, String p_object, String p_location) {
	Dictionary at = p_state["at"];
	Variant truck = at[p_object];
	Dictionary truck_at = p_state["truck_at"];
	if (truck_at[truck] == p_location) {
		at[p_object] = p_location;
		p_state["at"] = at;
		return p_state;
	}
	return false;
}

static Dictionary fly_plane(Dictionary p_state, String p_plane, String p_airport) {
	Dictionary plane_at = p_state["plane_at"];
	plane_at[p_plane] = p_airport;
	return p_state;
}

static Dictionary load_plane(Dictionary p_state, String p_object, String p_plane) {
	Dictionary at = p_state["at"];
	at[p_object] = p_plane;
	p_state["at"] = at;
	return p_state;
}

static Variant unload_plane(Dictionary p_state, String p_object, String p_airport) {
	Dictionary at = p_state["at"];
	String plane = at[p_object];
	Dictionary plane_at = p_state["plane_at"];
	if (plane_at[plane] == p_airport) {
		at[p_object] = p_airport;
		p_state["at"] = at;
		return p_state;
	}
	return false;
}

// Helper functions for the methods.

// Find a truck in the same city as the package.

static Variant find_truck(Dictionary p_state, String p_object) {
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

//  Find a plane in the same city as the package; if none available, find a random plane.

static Variant find_plane(Dictionary p_state, String p_object) {
	Variant random_plane;
	Array airplanes = p_state["airplanes"];

	Dictionary in_city = p_state["in_city"];
	Dictionary plane_at = p_state["plane_at"];
	Dictionary at = p_state["at"];
	for (int64_t i = 0; i < airplanes.size(); ++i) {
		String plane = airplanes[i];
		if (in_city[plane_at[plane]] == in_city[at[p_object]]) {
			return plane;
		}
		random_plane = plane;
	}
	return random_plane;
}

//  Find an airport in the same city as the location.

static Variant find_airport(Dictionary p_state, String p_location) {
	Array airports = p_state["airports"];

	Dictionary in_city = p_state["in_city"];
	for (int i = 0; i < airports.size(); ++i) {
		String airport = airports[i];
		if (in_city[airport] == in_city[p_location]) {
			return airport;
		}
	}
	return false;
}

// Methods to call the actions

static Variant m_drive_truck(Dictionary p_state, String p_truck, String p_location) {
	Array trucks = p_state["trucks"];
	Array locations = p_state["locations"];

	Dictionary in_city = p_state["in_city"];
	Dictionary truck_at = p_state["truck_at"];
	if (trucks.has(p_truck) && locations.has(p_location) && in_city[truck_at[p_truck]] == in_city[p_location]) {
		TypedArray<Array> task;
		task.push_back(varray("drive_truck", p_truck, p_location));
		return task;
	}
	return false;
}

static Variant m_load_truck(Dictionary p_state, String p_object, String p_truck) {
	Array packages = p_state["packages"];
	Array trucks = p_state["trucks"];

	Dictionary at = p_state["at"];
	Dictionary truck_at = p_state["truck_at"];

	bool is_package_existing = packages.has(p_object);
	bool is_truck_existing = trucks.has(p_truck);
	bool is_object_at_truck = (at[p_object] == truck_at[p_truck]);

	if (is_package_existing && is_truck_existing && is_object_at_truck) {
		TypedArray<Array> todo;
		todo.push_back(varray("load_truck", p_object, p_truck));
		return todo;
	}
	return false;
}

static Variant m_unload_truck(Dictionary p_state, String p_object, String p_location) {
	Array packages = p_state["packages"];
	Array trucks = p_state["trucks"];
	Array locations = p_state["locations"];

	Dictionary at = p_state["at"];
	if (packages.has(p_object) && trucks.has(at[p_object]) && locations.has(p_location)) {
		TypedArray<Array> todo;
		todo.push_back(varray("unload_truck", p_object, p_location));
		return todo;
	}
	return false;
}

static Variant m_fly_plane(Dictionary p_state, String p_plane, String p_airport) {
	Array airplanes = p_state["airplanes"];
	Array airports = p_state["airports"];
	if (airplanes.has(p_plane) && airports.has(p_airport)) {
		TypedArray<Array> todo;
		todo.push_back(varray("fly_plane", p_plane, p_airport));
		return todo;
	}
	return false;
}

static Variant m_load_plane(Dictionary p_state, String p_object, String p_plane) {
	Array packages = p_state["packages"];
	Array airplanes = p_state["airplanes"];

	Dictionary at = p_state["at"];
	Dictionary plane_at = p_state["plane_at"];
	if (packages.has(p_object) && airplanes.has(p_plane) && at[p_object] == plane_at[p_plane]) {
		TypedArray<Array> todo;
		todo.push_back(varray("load_plane", p_object, p_plane));
		return todo;
	}
	return false;
}

static Variant m_unload_plane(Dictionary p_state, String p_object, String p_airport) {
	Dictionary packages = p_state["packages"];
	Dictionary at = p_state["at"];
	Dictionary airplanes = p_state["airplanes"];
	Dictionary airports = p_state["airports"];
	if (packages.has(p_object) && airplanes.has(at[p_object]) && airports.has(p_airport)) {
		TypedArray<Array> todo;
		todo.push_back(varray("unload_plane", p_object, p_airport));
		return todo;
	}
	return false;
}

// Other methods

static Variant move_within_city(Dictionary p_state, String p_object, String p_location) {
	Array packages = p_state["packages"];
	Array locations = p_state["locations"];

	Dictionary in_city = p_state["in_city"];
	Dictionary at = p_state["at"];
	String truck = find_truck(p_state, p_object);
	if (packages.has(p_object) && locations.has(at[p_object]) && in_city[at[p_object]] == in_city[p_location] && !truck.is_empty()) {
		TypedArray<Array> todo;
		todo.push_back(varray("truck_at", truck, at[p_object]));
		todo.push_back(varray("at", p_object, truck));
		todo.push_back(varray("truck_at", truck, p_location));
		todo.push_back(varray("at", p_object, p_location));
		return todo;
	}
	return false;
}

static Variant move_between_airports(Dictionary state, String o, String a) {
	Array packages = state["packages"];
	Array airports = state["airports"];

	Dictionary in_city = state["in_city"];
	Dictionary at = state["at"];
	String plane = find_plane(state, o);
	if (packages.has(o) && airports.has(at[o]) && airports.has(a) && in_city[at[o]] != in_city[a] && !plane.is_empty()) {
		Array result;
		result.push_back(varray("plane_at", plane, at[o]));
		result.push_back(varray("at", o, plane));
		result.push_back(varray("plane_at", plane, a));
		result.push_back(varray("at", o, a));
		return result;
	}
	return false;
}

Variant move_between_city(Dictionary p_state, String p_object, String p_location) {
	Array packages = p_state["packages"];
	Array locations = p_state["locations"];

	Dictionary in_city = p_state["in_city"];
	Dictionary at = p_state["at"];
	String airport_1 = find_airport(p_state, at[p_object]);
	String airport_2 = find_airport(p_state, p_location);
	if (packages.has(p_object) && locations.has(at[p_object]) && in_city[at[p_object]] != in_city[p_location] && !airport_1.is_empty() && !airport_2.is_empty()) {
		TypedArray<Array> todo;
		todo.push_back(varray("at", p_object, airport_1));
		todo.push_back(varray("at", p_object, airport_2));
		todo.push_back(varray("at", p_object, p_location));
		return todo;
	}
	return false;
}

void before_each(Dictionary &p_state, Ref<Plan> p_planner, Ref<Domain> p_the_domain) {
	ERR_FAIL_COND(p_planner.is_null());
	ERR_FAIL_COND(p_the_domain.is_null());
	p_planner->set_verbose(3);
	p_state.clear();
	TypedArray<Domain> domains;
	domains.push_back(p_the_domain);
	p_planner->set_domains(domains);

	// If we've changed to some other domain, this will change us back.
	p_planner->set_current_domain(p_the_domain);
	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&drive_truck));
	actions.push_back(callable_mp_static(&load_truck));
	actions.push_back(callable_mp_static(&unload_truck));
	actions.push_back(callable_mp_static(&fly_plane));
	actions.push_back(callable_mp_static(&load_plane));
	actions.push_back(callable_mp_static(&unload_plane));
	p_planner->declare_actions(actions);

	TypedArray<Callable> truck_at_methods;
	truck_at_methods.push_back(callable_mp_static(&m_drive_truck));
	p_planner->declare_unigoal_methods("truck_at", truck_at_methods);

	TypedArray<Callable> plane_at_methods;
	plane_at_methods.push_back(callable_mp_static(&m_fly_plane));
	p_planner->declare_unigoal_methods("plane_at", plane_at_methods);

	TypedArray<Callable> at_methods;
	at_methods.push_back(callable_mp_static(&m_load_truck));
	at_methods.push_back(callable_mp_static(&m_unload_truck));
	at_methods.push_back(callable_mp_static(&m_load_plane));
	at_methods.push_back(callable_mp_static(&m_unload_plane));
	p_planner->declare_unigoal_methods("at", at_methods);

	TypedArray<Callable> move_methods;
	at_methods.push_back(callable_mp_static(&move_within_city));
	at_methods.push_back(callable_mp_static(&move_between_airports));
	at_methods.push_back(callable_mp_static(&move_between_city));
	p_planner->declare_unigoal_methods("at", move_methods);

	p_planner->get_current_domain()->print_domain();
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
	locations.push_back("location1");
	locations.push_back("location2");
	locations.push_back("location3");
	locations.push_back("airport1");
	locations.push_back("location10");
	locations.push_back("airport2");
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
	in_city["location1"] = "city1";
	in_city["location2"] = "city1";
	in_city["location3"] = "city1";
	in_city["airport1"] = "city1";
	in_city["location10"] = "city2";
	in_city["airport2"] = "city2";
	p_state["in_city"] = in_city;
}

TEST_CASE("[Modules][GoalTaskPlanner] Move Goal 1") {
	Ref<Plan> planner;
	planner.instantiate();
	Ref<Domain> the_domain;
	the_domain.instantiate("Move Goal 1");
	Dictionary state1;
	before_each(state1, planner, the_domain);
	TypedArray<Array> task;
	task.push_back(varray("at", "package1", "location2"));
	task.push_back(varray("at", "package2", "location3"));
	Variant plan = planner->find_plan(
			state1.duplicate(true),
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
	the_domain.instantiate("Move Goal 2");
	Dictionary state1;
	before_each(state1, planner, the_domain);

	Array task;
	task.push_back(varray("at", "package1", "location10"));

	Variant plan = planner->find_plan(state1.duplicate(true), task);

	CHECK_EQ(plan, Array());
	// CHECK_ARRAY_EQ(
	// 		plan,
	// 		varray(
	// 				Dictionary::make("drive_truck", "truck1", "location1"),
	// 				Dictionary::make("load_truck", "package1", "truck1"),
	// 				Dictionary::make("drive_truck", "truck1", "airport1"),
	// 				Dictionary::make("unload_truck", "package1", "airport1"),
	// 				Dictionary::make("fly_plane", "plane2", "airport1"),
	// 				Dictionary::make("load_plane", "package1", "plane2"),
	// 				Dictionary::make("fly_plane", "plane2", "airport2"),
	// 				Dictionary::make("unload_plane", "package1", "airport2"),
	// 				Dictionary::make("drive_truck", "truck6", "airport2"),
	// 				Dictionary::make("load_truck", "package1", "truck6"),
	// 				Dictionary::make("drive_truck", "truck6", "location10"),
	// 				Dictionary::make("unload_truck", "package1", "location10")));
}

TEST_CASE("[Modules][GoalTaskPlanner] Move Goal 3") {
	Ref<Plan> planner;
	planner.instantiate();
	Ref<Domain> the_domain;
	the_domain.instantiate("Move Goal 3");
	Dictionary state1;
	before_each(state1, planner, the_domain);

	Array task;
	task.push_back(varray("at", "package1", "location1"));

	Variant plan = planner->find_plan(state1.duplicate(true), task);

	CHECK_EQ(plan, Array());
}

TEST_CASE("[Modules][GoalTaskPlanner] Move Goal 4") {
	Ref<Plan> planner;
	planner.instantiate();
	Ref<Domain> the_domain;
	the_domain.instantiate("Move Goal 4");
	Dictionary state1;
	before_each(state1, planner, the_domain);
	Array task;
	task.push_back(varray("at", "package1", "location2"));

	Variant plan = planner->find_plan(state1.duplicate(true), task);
	CHECK_EQ(plan, Array());
	// CHECK_ARRAY_EQ(
	// 		plan,
	// 		varray(
	// 				Dictionary::make("drive_truck", "truck1", "location1"),
	// 				Dictionary::make("load_truck", "package1", "truck1"),
	// 				Dictionary::make("drive_truck", "truck1", "location2"),
	// 				Dictionary::make("unload_truck", "package1", "location2")));
}

} // namespace TestLogistics

#endif // TEST_LOGISTICS_H
