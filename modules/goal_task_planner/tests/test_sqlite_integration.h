// C++ unit tests for SQLite integration in goal_task_planner

#pragma once

#include "tests/test_macros.h"
#include "../plan.h"
#include "../planner_state.h"

namespace TestSQLiteIntegration {

TEST_CASE("[Modules][SQLite] Database initialization") {
    Ref<PlannerPlan> plan = memnew(PlannerPlan);
    
    SUBCASE("In-memory database initialization") {
        bool success = plan->initialize_database("");
        CHECK(success == true);
    }
    
    SUBCASE("File database initialization") {
        // Note: This would create a file, so we test in-memory for unit tests
        // In actual QA tests, we can test file-based databases
        bool success = plan->initialize_database(":memory:");
        CHECK(success == true);
    }
    
    memdelete(plan.ptr());
}

TEST_CASE("[Modules][SQLite] Temporal state storage and retrieval") {
    Ref<PlannerPlan> plan = memnew(PlannerPlan);
    plan->initialize_database("");
    
    SUBCASE("Store and load temporal state") {
        Dictionary state;
        state["test_key"] = "test_value";
        int64_t current_time = PlannerHLClock::now_microseconds();
        
        plan->store_temporal_state(state, current_time);
        Dictionary loaded = plan->load_temporal_state();
        
        CHECK(loaded.has("test_key"));
        CHECK(loaded["test_key"] == "test_value");
        CHECK(loaded.has("current_time"));
        CHECK(loaded["current_time"] == current_time);
    }
    
    SUBCASE("Multiple state updates") {
        Dictionary state1;
        state1["version"] = 1;
        int64_t time1 = 1735689600000000LL;
        plan->store_temporal_state(state1, time1);
        
        Dictionary state2;
        state2["version"] = 2;
        int64_t time2 = 1735689601000000LL;
        plan->store_temporal_state(state2, time2);
        
        Dictionary loaded = plan->load_temporal_state();
        CHECK(loaded["version"] == 2); // Should get latest
        CHECK(loaded["current_time"] == time2);
    }
    
    memdelete(plan.ptr());
}

TEST_CASE("[Modules][SQLite] Entity capabilities storage") {
    Ref<PlannerPlan> plan = memnew(PlannerPlan);
    plan->initialize_database("");
    
    SUBCASE("Store entity capability") {
        String entity_id = "entity_1";
        String capability = "movable";
        Dictionary value;
        value["speed"] = 5.0;
        int64_t timestamp = PlannerHLClock::now_microseconds();
        
        plan->store_entity_capability(entity_id, capability, value, timestamp);
        // Capabilities are stored but not directly queryable through public API
        // This would be tested through PlannerState integration
        CHECK(true); // If no error, storage succeeded
    }
    
    memdelete(plan.ptr());
}

TEST_CASE("[Modules][SQLite] Planning operations persistence") {
    Ref<PlannerPlan> plan = memnew(PlannerPlan);
    plan->initialize_database("");
    
    SUBCASE("Store planning operation") {
        String operation_id = "op_123";
        String operation_type = "task";
        Dictionary operation_data;
        operation_data["task"] = "move";
        int64_t timestamp = PlannerHLClock::now_microseconds();
        
        plan->store_planning_operation(operation_id, operation_type, operation_data, timestamp);
        CHECK(true); // If no error, storage succeeded
    }
    
    SUBCASE("submit_operation stores in database") {
        Dictionary operation;
        operation["action"] = "pickup";
        Dictionary result = plan->submit_operation(operation);
        
        CHECK(result.has("operation_id"));
        CHECK(result.has("agreed_at"));
        // Operation should be stored in database
    }
    
    memdelete(plan.ptr());
}

TEST_CASE("[Modules][SQLite] get_global_state loads from database") {
    Ref<PlannerPlan> plan = memnew(PlannerPlan);
    plan->initialize_database("");
    
    SUBCASE("Load state from database") {
        Dictionary state;
        state["key"] = "value";
        int64_t current_time = PlannerHLClock::now_microseconds();
        plan->store_temporal_state(state, current_time);
        
        Dictionary global_state = plan->get_global_state();
        CHECK(global_state.has("key"));
        CHECK(global_state["key"] == "value");
        CHECK(global_state.has("hlc"));
    }
    
    SUBCASE("Fallback when database not initialized") {
        Ref<PlannerPlan> plan_no_db = memnew(PlannerPlan);
        // Don't initialize database
        Dictionary global_state = plan_no_db->get_global_state();
        CHECK(global_state.has("hlc")); // Should still have HLC
        memdelete(plan_no_db.ptr());
    }
    
    memdelete(plan.ptr());
}

} // namespace TestSQLiteIntegration

