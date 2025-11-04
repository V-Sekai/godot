// C++ unit tests for goal solver and entity matching functionality

#pragma once

#include "tests/test_macros.h"
#include "../plan.h"
#include "../domain.h"
#include "../planner_state.h"
#include "../entity_requirement.h"

namespace TestGoalSolver {

// Note: Entity matching is tested through PlannerMetadata integration in the planning loop
// The _match_entities helper is private and used internally when PlannerMetadata has entity requirements
// Entity matching tests should be done through actual planning scenarios with PlannerMetadata

TEST_CASE("[Modules][GoalSolver] Temporal constraints methods") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = memnew(PlannerDomain);
	plan->set_current_domain(domain);
	
	// Test attach_temporal_constraints
	Dictionary temporal_constraints;
	temporal_constraints["duration"] = "PT2H";
	temporal_constraints["start_time"] = "2024-01-01T10:00:00Z";
	
	Array test_item;
	test_item.push_back("test");
	test_item.push_back("goal");
	
	Variant result = plan->_attach_temporal_constraints(test_item, temporal_constraints);
	CHECK(result.get_type() == Variant::DICTIONARY);
	
	Dictionary result_dict = result;
	CHECK(result_dict.has("item"));
	CHECK(result_dict.has("temporal_constraints"));
	
	// Test has_temporal_constraints
	bool has_constraints = plan->_has_temporal_constraints(result);
	CHECK(has_constraints == true);
	
	// Test get_temporal_constraints
	Dictionary retrieved = plan->_get_temporal_constraints(result);
	CHECK(retrieved.has("duration"));
	CHECK(retrieved["duration"] == "PT2H");
}

TEST_CASE("[Modules][GoalSolver] Unigoal ordering optimization") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = memnew(PlannerDomain);
	plan->set_current_domain(domain);
	
	// Create unigoal method dictionary
	Dictionary unigoal_method_dict;
	
	// Goal 1: has 2 methods (less constraining)
	TypedArray<Callable> methods1;
	unigoal_method_dict["goal1"] = methods1; // Empty for now
	
	// Goal 2: has 1 method (more constraining)
	TypedArray<Callable> methods2;
	unigoal_method_dict["goal2"] = methods2; // Empty for now
	
	// Create unigoals array
	Array unigoals;
	Array goal1;
	goal1.push_back("goal1");
	goal1.push_back("arg1");
	goal1.push_back("value1");
	unigoals.push_back(goal1);
	
	Array goal2;
	goal2.push_back("goal2");
	goal2.push_back("arg2");
	goal2.push_back("value2");
	unigoals.push_back(goal2);
	
	Dictionary state;
	
	// Optimize order
	Array optimized = plan->_optimize_unigoal_order(unigoals, state, unigoal_method_dict);
	
	// Should return same number of goals
	CHECK(optimized.size() == unigoals.size());
}

}
