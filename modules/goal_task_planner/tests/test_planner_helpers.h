/**************************************************************************/
/*  test_planner_helpers.h                                               */
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

// Helper functions and domain definitions shared by planner tests.

#pragma once

#include "../planner_state.h"
#include "core/variant/callable.h"
#include "tests/test_macros.h"

namespace TestComprehensivePlanner {

// Wrapper class used to construct Callables for free functions in RestaurantDomain.
// Defined before RestaurantDomain namespace so it can be referenced from within it.
class RestaurantDomainCallable {
public:
	static Dictionary action_cook(Dictionary p_state, String p_dish, String p_chef);
	static Dictionary action_serve(Dictionary p_state, String p_dish, String p_customer, String p_waiter);
	static Dictionary action_clean(Dictionary p_state, String p_table);
	static Array task_prepare_meal(Dictionary p_state, String p_dish, String p_chef);
	static Array task_serve_customer(Dictionary p_state, String p_dish, String p_customer, String p_waiter);
	static Array unigoal_cook_dish(Dictionary p_state, String p_dish, String p_chef);
	static Array unigoal_clean_table(Dictionary p_state, String p_table);
	static Array multigoal_serve_customers(Dictionary p_state, Dictionary p_multigoal);
};

// Helper functions for restaurant domain
namespace RestaurantDomain {

// Actions
Dictionary action_cook(Dictionary state, String dish, String chef) {
	Dictionary new_state = state.duplicate();
	Dictionary chef_state;
	if (new_state.has(chef)) {
		chef_state = new_state[chef];
	} else {
		chef_state = Dictionary();
	}
	chef_state["cooking"] = dish;
	new_state[chef] = chef_state;

	Dictionary dish_state;
	if (new_state.has(dish)) {
		dish_state = new_state[dish];
	} else {
		dish_state = Dictionary();
	}
	dish_state["status"] = "cooked";
	new_state[dish] = dish_state;
	return new_state;
}

Dictionary action_serve(Dictionary state, String dish, String customer, String waiter) {
	Dictionary new_state = state.duplicate();
	Dictionary customer_state;
	if (new_state.has(customer)) {
		customer_state = new_state[customer];
	} else {
		customer_state = Dictionary();
	}
	customer_state["served"] = dish;
	new_state[customer] = customer_state;

	Dictionary dish_state;
	if (new_state.has(dish)) {
		dish_state = new_state[dish];
	} else {
		dish_state = Dictionary();
	}
	dish_state["status"] = "served";
	new_state[dish] = dish_state;
	return new_state;
}

Dictionary action_clean(Dictionary state, String table) {
	Dictionary new_state = state.duplicate();
	Dictionary table_state;
	if (new_state.has(table)) {
		table_state = new_state[table];
	} else {
		table_state = Dictionary();
	}
	table_state["clean"] = true;
	new_state[table] = table_state;
	return new_state;
}

// Task methods
Array task_prepare_meal(Dictionary state, String dish, String chef) {
	Array subtasks;
	subtasks.push_back(callable_mp_static(&RestaurantDomainCallable::action_cook).bind(dish, chef));
	return subtasks;
}

Array task_serve_customer(Dictionary state, String dish, String customer, String waiter) {
	Array subtasks;
	subtasks.push_back(callable_mp_static(&RestaurantDomainCallable::action_serve).bind(dish, customer, waiter));
	return subtasks;
}

// Unigoal methods
Array unigoal_cook_dish(Dictionary state, String dish, String chef) {
	Array subtasks;
	Dictionary chef_state = state.get(chef, Dictionary());
	if (!chef_state.has("cooking") || chef_state["cooking"] != dish) {
		subtasks.push_back(callable_mp_static(&RestaurantDomainCallable::action_cook).bind(dish, chef));
	}
	return subtasks;
}

Array unigoal_clean_table(Dictionary state, String table) {
	Array subtasks;
	Dictionary table_state = state.get(table, Dictionary());
	if (!table_state.has("clean") || !table_state["clean"]) {
		subtasks.push_back(callable_mp_static(&RestaurantDomainCallable::action_clean).bind(table));
	}
	return subtasks;
}

// Multigoal method
Array multigoal_serve_customers(Dictionary state, Dictionary multigoal) {
	Array result;
	Array customers = multigoal.keys();
	for (int i = 0; i < customers.size(); i++) {
		String customer = customers[i];
		Dictionary customer_goal = multigoal[customer];
		String dish = customer_goal.get("dish", "");
		String waiter = customer_goal.get("waiter", "");
		if (!dish.is_empty() && !waiter.is_empty()) {
			result.push_back(callable_mp_static(&RestaurantDomainCallable::task_serve_customer).bind(dish, customer, waiter));
		}
	}
	return result;
}

} // namespace RestaurantDomain

// Implementations of RestaurantDomainCallable static methods
inline Dictionary RestaurantDomainCallable::action_cook(Dictionary p_state, String p_dish, String p_chef) {
	return RestaurantDomain::action_cook(p_state, p_dish, p_chef);
}

inline Dictionary RestaurantDomainCallable::action_serve(Dictionary p_state, String p_dish, String p_customer, String p_waiter) {
	return RestaurantDomain::action_serve(p_state, p_dish, p_customer, p_waiter);
}

inline Dictionary RestaurantDomainCallable::action_clean(Dictionary p_state, String p_table) {
	return RestaurantDomain::action_clean(p_state, p_table);
}

inline Array RestaurantDomainCallable::task_prepare_meal(Dictionary p_state, String p_dish, String p_chef) {
	return RestaurantDomain::task_prepare_meal(p_state, p_dish, p_chef);
}

inline Array RestaurantDomainCallable::task_serve_customer(Dictionary p_state, String p_dish, String p_customer, String p_waiter) {
	return RestaurantDomain::task_serve_customer(p_state, p_dish, p_customer, p_waiter);
}

inline Array RestaurantDomainCallable::unigoal_cook_dish(Dictionary p_state, String p_dish, String p_chef) {
	return RestaurantDomain::unigoal_cook_dish(p_state, p_dish, p_chef);
}

inline Array RestaurantDomainCallable::unigoal_clean_table(Dictionary p_state, String p_table) {
	return RestaurantDomain::unigoal_clean_table(p_state, p_table);
}

inline Array RestaurantDomainCallable::multigoal_serve_customers(Dictionary p_state, Dictionary p_multigoal) {
	return RestaurantDomain::multigoal_serve_customers(p_state, p_multigoal);
}

} // namespace TestComprehensivePlanner
