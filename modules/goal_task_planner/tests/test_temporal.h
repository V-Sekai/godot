// C++ unit tests for temporal goal-task planner features

#pragma once

#include "tests/test_macros.h"
#include "../planner_hl_clock.h"
#include "../domain.h"
#include "../plan.h"

namespace TestTemporal {

TEST_CASE("[Modules][Temporal] PlannerHLClock time calculations") {
    PlannerHLClock hlc;

    SUBCASE("Duration calculation with absolute microseconds") {
        // Use realistic absolute microseconds (e.g., 2025-01-01 00:00:00 UTC = ~1735689600000000 microseconds)
        int64_t start = 1735689600000000LL; // Absolute microseconds
        int64_t end = 1735689601000000LL;    // 1 second later
        int64_t duration = 1000000LL;        // 1 second in microseconds
        
        hlc.set_start_time(start);
        hlc.set_end_time(end);
        hlc.set_duration(duration);
        CHECK(hlc.get_duration() == duration);
        CHECK(hlc.get_end_time() - hlc.get_start_time() == hlc.get_duration());
    }

    SUBCASE("Time progression with absolute microseconds") {
        int64_t start = 1735689600000000LL;
        int64_t duration = 500000LL; // 0.5 seconds in microseconds
        hlc.set_start_time(start);
        hlc.set_duration(duration);
        hlc.calculate_end_from_duration();
        CHECK(hlc.get_end_time() == hlc.get_start_time() + hlc.get_duration());
    }
    
    SUBCASE("Unix time to microseconds conversion") {
        double unix_time = 1735689600.0; // 2025-01-01 00:00:00 UTC in seconds
        int64_t microseconds = PlannerHLClock::unix_time_to_microseconds(unix_time);
        CHECK(microseconds == 1735689600000000LL);
    }
    
    SUBCASE("Calculate duration from start and end") {
        int64_t start = 1735689600000000LL;
        int64_t end = 1735689601000000LL;
        hlc.set_start_time(start);
        hlc.set_end_time(end);
        hlc.calculate_duration();
        CHECK(hlc.get_duration() == 1000000LL); // 1 second
    }
}

TEST_CASE("[Modules][Temporal] PlannerTaskMetadata temporal updates") {
    Ref<PlannerTaskMetadata> metadata = memnew(PlannerTaskMetadata);

    SUBCASE("Metadata time tracking with absolute microseconds") {
        int64_t absolute_time = 1735689600000000LL;
        metadata->update_metadata(absolute_time);
        PlannerHLClock hlc = metadata->get_hlc();
        CHECK(hlc.get_start_time() == absolute_time);
    }

    SUBCASE("Multiple updates with absolute microseconds") {
        int64_t time1 = 1735689600000000LL;
        int64_t time2 = 1735689601000000LL;
        metadata->update_metadata(time1);
        metadata->update_metadata(time2);
        PlannerHLClock hlc = metadata->get_hlc();
        CHECK(hlc.get_start_time() == time2);  // Last update wins
    }

    memdelete(metadata.ptr());
}

TEST_CASE("[Modules][Temporal] PlannerPlan temporal integration") {
    PlannerPlan plan;

    SUBCASE("Plan HLC management with absolute microseconds") {
        PlannerHLClock hlc;
        int64_t start = 1735689600000000LL;
        int64_t duration = 500000LL; // 0.5 seconds
        hlc.set_start_time(start);
        hlc.set_duration(duration);
        hlc.calculate_end_from_duration();

        plan.set_hlc(hlc);
        PlannerHLClock retrieved = plan.get_hlc();

        CHECK(retrieved.get_start_time() == start);
        CHECK(retrieved.get_duration() == duration);
        CHECK(retrieved.get_end_time() == start + duration);
    }

    SUBCASE("Plan temporal state with absolute microseconds") {
        // Test that plan maintains temporal state
        PlannerHLClock hlc;
        int64_t start = 1735689600000000LL;
        hlc.set_start_time(start);
        plan.set_hlc(hlc);

        // Simulate some operation
        PlannerHLClock updated_hlc = plan.get_hlc();
        int64_t end = start + 1000000LL; // 1 second later
        updated_hlc.set_end_time(end);
        plan.set_hlc(updated_hlc);

        PlannerHLClock final_hlc = plan.get_hlc();
        CHECK(final_hlc.get_start_time() == start);
        CHECK(final_hlc.get_end_time() == end);
    }
    
    SUBCASE("submit_operation uses absolute microseconds") {
        Dictionary operation;
        operation["type"] = "test";
        Dictionary result = plan.submit_operation(operation);
        
        CHECK(result.has("operation_id"));
        CHECK(result.has("agreed_at"));
        int64_t agreed_at = result["agreed_at"];
        // Should be a reasonable absolute time (large number representing microseconds since epoch)
        CHECK(agreed_at > 1000000000000000LL); // Should be after year 2001
    }
}

TEST_CASE("[Modules][Temporal] Temporal constraints validation") {
    PlannerHLClock hlc;

    SUBCASE("Valid time ranges with absolute microseconds") {
        int64_t start = 1735689600000000LL;
        int64_t end = 1735689601000000LL; // 1 second later
        int64_t duration = 1000000LL;
        hlc.set_start_time(start);
        hlc.set_end_time(end);
        hlc.set_duration(duration);
        CHECK(hlc.get_start_time() < hlc.get_end_time());
        CHECK(hlc.get_duration() == hlc.get_end_time() - hlc.get_start_time());
    }

    SUBCASE("Edge case: zero duration") {
        int64_t time = 1735689600000000LL;
        hlc.set_start_time(time);
        hlc.set_end_time(time);
        hlc.set_duration(0);
        CHECK(hlc.get_start_time() == hlc.get_end_time());
        CHECK(hlc.get_duration() == 0);
    }
    
    SUBCASE("Time arithmetic with absolute microseconds") {
        int64_t base_time = 1735689600000000LL;
        int64_t duration = 5000000LL; // 5 seconds
        int64_t end_time = base_time + duration;
        
        hlc.set_start_time(base_time);
        hlc.set_duration(duration);
        hlc.calculate_end_from_duration();
        CHECK(hlc.get_end_time() == end_time);
    }
    
    SUBCASE("Timestamp comparisons") {
        int64_t time1 = 1735689600000000LL;
        int64_t time2 = 1735689601000000LL;
        CHECK(time1 < time2);
        CHECK(time2 > time1);
        CHECK(time1 == time1);
    }
}

} // namespace TestTemporal