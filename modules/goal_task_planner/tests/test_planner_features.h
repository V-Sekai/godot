// C++ unit tests for goal_task_planner temporal features, following Godot conventions

#pragma once

#include "tests/test_macros.h"
#include "../planner_hl_clock.h"
#include "../domain.h"
#include "../plan.h"
#include "../planner_state.h"

namespace TestPlannerFeatures {

TEST_CASE("[Modules][PlannerHLClock] Basic functionality") {
    PlannerHLClock hlc;

    SUBCASE("Initial state") {
        CHECK(hlc.get_start_time() == 0);
        CHECK(hlc.get_end_time() == 0);
        CHECK(hlc.get_duration() == 0);
    }

    SUBCASE("Set times with absolute microseconds") {
        int64_t start = 1735689600000000LL;
        int64_t end = 1735689601000000LL;
        int64_t duration = 1000000LL;
        hlc.set_start_time(start);
        hlc.set_end_time(end);
        hlc.set_duration(duration);
        CHECK(hlc.get_start_time() == start);
        CHECK(hlc.get_end_time() == end);
        CHECK(hlc.get_duration() == duration);
    }
    
    SUBCASE("now_microseconds returns reasonable value") {
        int64_t now = PlannerHLClock::now_microseconds();
        // Should be a reasonable absolute time (after year 2001)
        CHECK(now > 1000000000000000LL);
    }
}

TEST_CASE("[Modules][PlannerTaskMetadata] Basic functionality") {
    Ref<PlannerTaskMetadata> metadata = memnew(PlannerTaskMetadata);

    SUBCASE("ID generation") {
        String id = metadata->get_task_id();
        CHECK(!id.is_empty());
        // Check for UUID format (contains dashes)
        CHECK(id.contains("-"));
        CHECK(id.length() == 36);  // Standard UUID length
    }

    SUBCASE("HLC integration with absolute microseconds") {
        int64_t absolute_time = 1735689600000000LL;
        metadata->update_metadata(absolute_time);
        PlannerHLClock hlc = metadata->get_hlc();
        CHECK(hlc.get_start_time() == absolute_time);
    }

    memdelete(metadata.ptr());
}

TEST_CASE("[Modules][PlannerPlan] ID generation and HLC") {
    PlannerPlan plan;

    SUBCASE("Generate plan ID") {
        String id = plan.generate_plan_id();
        CHECK(!id.is_empty());
        // Check for Base32 characters
        String valid_chars = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";
        for (int i = 0; i < id.length(); i++) {
            char c = id[i];
            CHECK(valid_chars.contains(String::chr(c)));
        }
    }

    SUBCASE("HLC in plan") {
        PlannerHLClock hlc;
        plan.set_hlc(hlc);
        PlannerHLClock retrieved = plan.get_hlc();
        CHECK(retrieved.start_time == hlc.start_time);
    }
}

TEST_CASE("[Modules][PlannerTask] With metadata") {
    PlannerTask task;
    Ref<PlannerTaskMetadata> metadata = memnew(PlannerTaskMetadata);
    task.set_metadata(metadata);

    SUBCASE("Metadata attachment") {
        CHECK(task.get_metadata() == metadata);
    }

    memdelete(metadata.ptr());
}

TEST_CASE("[Modules][PlannerState] Entity capabilities") {
    Ref<PlannerState> state = memnew(PlannerState);
    
    SUBCASE("Entity capability get/set operations") {
        String entity_id = "robot_1";
        String capability = "movable";
        Dictionary value;
        value["speed"] = 5.0;
        
        state->set_entity_capability(entity_id, capability, value);
        Variant retrieved = state->get_entity_capability(entity_id, capability);
        CHECK(retrieved.get_type() == Variant::DICTIONARY);
        Dictionary retrieved_dict = retrieved;
        CHECK(retrieved_dict.has("speed"));
    }
    
    SUBCASE("has_entity check") {
        String entity_id = "robot_1";
        CHECK(state->has_entity(entity_id) == false);
        
        state->set_entity_capability(entity_id, "movable", true);
        CHECK(state->has_entity(entity_id) == true);
    }
    
    SUBCASE("get_all_entities") {
        state->set_entity_capability("entity_1", "cap1", 1);
        state->set_entity_capability("entity_2", "cap2", 2);
        
        Array entities = state->get_all_entities();
        CHECK(entities.size() >= 2);
    }
    
    memdelete(state.ptr());
}

TEST_CASE("[Modules][PlannerPlan] SQLite integration") {
    Ref<PlannerPlan> plan = memnew(PlannerPlan);
    
    SUBCASE("Database initialization") {
        bool success = plan->initialize_database("");
        CHECK(success == true);
    }
    
    SUBCASE("Temporal state persistence") {
        plan->initialize_database("");
        Dictionary state;
        state["key"] = "value";
        int64_t current_time = PlannerHLClock::now_microseconds();
        plan->store_temporal_state(state, current_time);
        
        Dictionary loaded = plan->load_temporal_state();
        CHECK(loaded.has("key"));
    }
    
    memdelete(plan.ptr());
}

} // namespace TestPlannerFeatures