// C++ unit tests for goal_task_planner temporal features, following Godot conventions

#pragma once

#include "tests/test_macros.h"
#include "../planner_hl_clock.h"
#include "../domain.h"
#include "../plan.h"
#include "../planner_parallel_commits.h"

namespace TestPlannerFeatures {

TEST_CASE("[Modules][PlannerHLClock] Basic functionality") {
    PlannerHLClock hlc;

    SUBCASE("Initial state") {
        CHECK(hlc.get_start_time() == 0);
        CHECK(hlc.get_end_time() == 0);
        CHECK(hlc.get_duration() == 0);
    }

    SUBCASE("Set times") {
        hlc.set_start_time(1000);
        hlc.set_end_time(2000);
        hlc.set_duration(1000);
        CHECK(hlc.get_start_time() == 1000);
        CHECK(hlc.get_end_time() == 2000);
        CHECK(hlc.get_duration() == 1000);
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

    SUBCASE("HLC integration") {
        PlannerHLClock hlc = metadata->get_hlc();
        metadata->update_metadata(2000);
        CHECK(hlc.get_start_time() == 2000);
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

} // namespace TestPlannerFeatures