// C++ unit tests for temporal goal-task planner features

#pragma once

#include "tests/test_macros.h"
#include "../planner_hl_clock.h"
#include "../domain.h"
#include "../plan.h"

namespace TestTemporal {

TEST_CASE("[Modules][Temporal] PlannerHLClock time calculations") {
    PlannerHLClock hlc;

    SUBCASE("Duration calculation") {
        hlc.set_start_time(1000);
        hlc.set_end_time(2000);
        hlc.set_duration(1000);
        CHECK(hlc.get_duration() == 1000);
        CHECK(hlc.get_end_time() - hlc.get_start_time() == hlc.get_duration());
    }

    SUBCASE("Time progression") {
        hlc.set_start_time(0);
        hlc.set_duration(500);
        hlc.set_end_time(500);
        CHECK(hlc.get_end_time() == hlc.get_start_time() + hlc.get_duration());
    }
}

TEST_CASE("[Modules][Temporal] PlannerTaskMetadata temporal updates") {
    Ref<PlannerTaskMetadata> metadata = memnew(PlannerTaskMetadata);

    SUBCASE("Metadata time tracking") {
        metadata->update_metadata(1500);
        PlannerHLClock hlc = metadata->get_hlc();
        CHECK(hlc.get_start_time() == 1500);
    }

    SUBCASE("Multiple updates") {
        metadata->update_metadata(1000);
        metadata->update_metadata(2000);
        PlannerHLClock hlc = metadata->get_hlc();
        CHECK(hlc.get_start_time() == 2000);  // Last update wins
    }

    memdelete(metadata.ptr());
}

TEST_CASE("[Modules][Temporal] PlannerPlan temporal integration") {
    PlannerPlan plan;

    SUBCASE("Plan HLC management") {
        PlannerHLClock hlc;
        hlc.set_start_time(1000);
        hlc.set_duration(500);
        hlc.set_end_time(1500);

        plan.set_hlc(hlc);
        PlannerHLClock retrieved = plan.get_hlc();

        CHECK(retrieved.get_start_time() == 1000);
        CHECK(retrieved.get_duration() == 500);
        CHECK(retrieved.get_end_time() == 1500);
    }

    SUBCASE("Plan temporal state") {
        // Test that plan maintains temporal state
        PlannerHLClock hlc;
        hlc.set_start_time(0);
        plan.set_hlc(hlc);

        // Simulate some operation
        PlannerHLClock updated_hlc = plan.get_hlc();
        updated_hlc.set_end_time(1000);
        plan.set_hlc(updated_hlc);

        PlannerHLClock final_hlc = plan.get_hlc();
        CHECK(final_hlc.get_start_time() == 0);
        CHECK(final_hlc.get_end_time() == 1000);
    }
}

TEST_CASE("[Modules][Temporal] Temporal constraints validation") {
    PlannerHLClock hlc;

    SUBCASE("Valid time ranges") {
        hlc.set_start_time(1000);
        hlc.set_end_time(2000);
        hlc.set_duration(1000);
        CHECK(hlc.get_start_time() < hlc.get_end_time());
        CHECK(hlc.get_duration() == hlc.get_end_time() - hlc.get_start_time());
    }

    SUBCASE("Edge case: zero duration") {
        hlc.set_start_time(1000);
        hlc.set_end_time(1000);
        hlc.set_duration(0);
        CHECK(hlc.get_start_time() == hlc.get_end_time());
        CHECK(hlc.get_duration() == 0);
    }
}

} // namespace TestTemporal