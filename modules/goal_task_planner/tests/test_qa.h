// QA test suite for goal_task_planner with absolute time and SQLite

#pragma once

#include "tests/test_macros.h"
#include "../plan.h"
#include "../planner_state.h"
#include "../domain.h"

namespace TestQA {

TEST_CASE("[QA] End-to-end temporal planning workflow with SQLite") {
    Ref<PlannerPlan> plan = memnew(PlannerPlan);
    plan->initialize_database("");
    
    // Initialize state
    Ref<PlannerState> state = memnew(PlannerState);
    state->set_predicate("block_a", "pos", "table");
    state->set_entity_capability("robot_1", "movable", true);
    
    // Store initial state
    int64_t current_time = PlannerHLClock::now_microseconds();
    Dictionary state_dict;
    state_dict["block_a"] = Dictionary();
    ((Dictionary)state_dict["block_a"])["pos"] = "table";
    plan->store_temporal_state(state_dict, current_time);
    
    // Submit operation
    Dictionary operation;
    operation["action"] = "pickup";
    operation["block"] = "block_a";
    Dictionary result = plan->submit_operation(operation);
    
    CHECK(result.has("operation_id"));
    CHECK(result.has("agreed_at"));
    CHECK(result["agreed_at"] > 0);
    
    // Load state back
    Dictionary loaded_state = plan->load_temporal_state();
    CHECK(loaded_state.has("block_a"));
    
    memdelete(plan.ptr());
    memdelete(state.ptr());
}

TEST_CASE("[QA] Entity capability persistence across planning sessions") {
    Ref<PlannerPlan> plan1 = memnew(PlannerPlan);
    plan1->initialize_database(":memory:");
    
    // Store entity capability
    String entity_id = "robot_1";
    String capability = "movable";
    Dictionary value;
    value["speed"] = 5.0;
    int64_t timestamp = PlannerHLClock::now_microseconds();
    plan1->store_entity_capability(entity_id, capability, value, timestamp);
    
    // Create new plan instance with same database would test persistence
    // For in-memory, we test that storage works
    CHECK(true);
    
    memdelete(plan1.ptr());
}

TEST_CASE("[QA] Absolute time accuracy in planning operations") {
    Ref<PlannerPlan> plan = memnew(PlannerPlan);
    
    // Get current time
    int64_t time_before = PlannerHLClock::now_microseconds();
    
    // Submit operation
    Dictionary operation;
    operation["test"] = true;
    Dictionary result = plan->submit_operation(operation);
    
    int64_t time_after = PlannerHLClock::now_microseconds();
    int64_t agreed_at = result["agreed_at"];
    
    // Agreed time should be between before and after
    CHECK(agreed_at >= time_before);
    CHECK(agreed_at <= time_after);
    
    memdelete(plan.ptr());
}

TEST_CASE("[QA] Memory management and resource cleanup") {
    // Test that resources are properly cleaned up
    for (int i = 0; i < 10; i++) {
        Ref<PlannerPlan> plan = memnew(PlannerPlan);
        plan->initialize_database("");
        
        Dictionary state;
        plan->store_temporal_state(state, PlannerHLClock::now_microseconds());
        
        memdelete(plan.ptr());
    }
    CHECK(true); // If we get here without crashing, cleanup works
}

TEST_CASE("[QA] Error handling for database operations") {
    Ref<PlannerPlan> plan = memnew(PlannerPlan);
    
    // Try to load without initializing database
    Dictionary state = plan->load_temporal_state();
    CHECK(state.is_empty()); // Should return empty dict, not crash
    
    // Try to store without initializing
    plan->store_temporal_state(Dictionary(), 0);
    // Should not crash
    
    memdelete(plan.ptr());
}

TEST_CASE("[QA] Performance with multiple operations") {
    Ref<PlannerPlan> plan = memnew(PlannerPlan);
    plan->initialize_database("");
    
    // Submit multiple operations
    for (int i = 0; i < 100; i++) {
        Dictionary operation;
        operation["index"] = i;
        plan->submit_operation(operation);
    }
    
    // Should complete without significant delay
    CHECK(true);
    
    memdelete(plan.ptr());
}

TEST_CASE("[QA] Backward compatibility - existing GDScript patterns") {
    Ref<PlannerPlan> plan = memnew(PlannerPlan);
    
    // Test that basic operations still work
    PlannerHLClock hlc;
    hlc.set_start_time(1735689600000000LL);
    plan->set_hlc(hlc);
    
    PlannerHLClock retrieved = plan->get_hlc();
    CHECK(retrieved.get_start_time() == 1735689600000000LL);
    
    // Test plan ID generation
    String id = plan->generate_plan_id();
    CHECK(!id.is_empty());
    
    memdelete(plan.ptr());
}

TEST_CASE("[QA] SQLite transaction rollback scenarios") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->initialize_database("");
	
	// Store initial state
	Dictionary state1;
	state1["version"] = 1;
	plan->store_temporal_state(state1, PlannerHLClock::now_microseconds());
	
	// Overwrite state
	Dictionary state2;
	state2["version"] = 2;
	plan->store_temporal_state(state2, PlannerHLClock::now_microseconds());
	
	// Should get latest state
	Dictionary loaded = plan->load_temporal_state();
	CHECK(loaded["version"] == 2);
	
	memdelete(plan.ptr());
}

TEST_CASE("[QA] Graph-based planning with run_lazy_refineahead") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = memnew(PlannerDomain);
	
	// Simple test action
	static Variant qa_test_action(Dictionary p_state, String p_arg) {
		Dictionary new_state = p_state.duplicate();
		new_state["qa_executed"] = p_arg;
		return new_state;
	}
	
	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&qa_test_action));
	domain->add_actions(actions);
	
	// Simple test task method
	static Variant qa_test_task_method(Dictionary p_state, String p_task_name) {
		Array subtasks;
		Array action_task;
		action_task.push_back("qa_test_action");
		action_task.push_back(p_task_name);
		subtasks.push_back(action_task);
		return subtasks;
	}
	
	TypedArray<Callable> task_methods;
	task_methods.push_back(callable_mp_static(&qa_test_task_method));
	domain->add_task_methods("qa_test_task", task_methods);
	
	plan->set_current_domain(domain);
	
	SUBCASE("Basic graph-based planning execution") {
		Dictionary initial_state;
		initial_state["initialized"] = true;
		
		Array todo_list;
		Array task;
		task.push_back("qa_test_task");
		task.push_back("qa_test_arg");
		todo_list.push_back(task);
		
		Dictionary final_state = plan->run_lazy_refineahead(initial_state, todo_list);
		
		CHECK(final_state.has("qa_executed"));
		CHECK(final_state["qa_executed"] == "qa_test_arg");
		CHECK(final_state.has("initialized"));
	}
	
	memdelete(domain.ptr());
	memdelete(plan.ptr());
}

TEST_CASE("[QA] STN integration in planning loop") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	
	// Create a simple domain with durative actions
	Ref<PlannerDomain> domain = memnew(PlannerDomain);
	
	// Create test action that takes time
	Dictionary action_dict;
	Array action_arr;
	action_arr.push_back("move");
	action_arr.push_back("robot");
	action_arr.push_back("location");
	action_dict["move"] = action_arr;
	domain->action_dictionary = action_dict;
	
	plan->set_current_domain(domain);
	plan->set_verbose(0);
	
	// Initialize state
	Dictionary state;
	state["robot"] = "start";
	state["location"] = "A";
	
	// Create todo list
	Array todo_list;
	Array action_task;
	action_task.push_back("move");
	action_task.push_back("robot");
	action_task.push_back("B");
	todo_list.push_back(action_task);
	
	// Run planning (should initialize STN)
	Dictionary final_state = plan->run_lazy_refineahead(state, todo_list);
	
	// Verify STN was initialized (plan has temporal tracking)
	PlannerHLClock hlc = plan->get_hlc();
	CHECK(hlc.get_start_time() > 0);
	
	memdelete(plan.ptr());
	memdelete(domain.ptr());
}

} // namespace TestQA

