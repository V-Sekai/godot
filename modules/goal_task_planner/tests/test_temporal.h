/**************************************************************************/
/*  test_temporal.h                                                       */
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

// C++ unit tests for temporal goal-task planner features

#pragma once

#include "../domain.h"
#include "../plan.h"
#include "../planner_time_range.h"
#include "tests/test_macros.h"

namespace TestTemporal {

TEST_CASE("[Modules][Temporal] PlannerTimeRange time calculations") {
	PlannerTimeRange time_range;

	SUBCASE("Duration calculation with absolute microseconds") {
		// Use realistic absolute microseconds (e.g., 2025-01-01 00:00:00 UTC = ~1735689600000000 microseconds)
		int64_t start = 1735689600000000LL; // Absolute microseconds
		int64_t end = 1735689601000000LL; // 1 second later
		int64_t duration = 1000000LL; // 1 second in microseconds

		time_range.set_start_time(start);
		time_range.set_end_time(end);
		time_range.set_duration(duration);
		CHECK(time_range.get_duration() == duration);
		CHECK(time_range.get_end_time() - time_range.get_start_time() == time_range.get_duration());
	}

	SUBCASE("Time progression with absolute microseconds") {
		int64_t start = 1735689600000000LL;
		int64_t duration = 500000LL; // 0.5 seconds in microseconds
		time_range.set_start_time(start);
		time_range.set_duration(duration);
		time_range.calculate_end_from_duration();
		CHECK(time_range.get_end_time() == time_range.get_start_time() + time_range.get_duration());
	}

	SUBCASE("Unix time to microseconds conversion") {
		double unix_time = 1735689600.0; // 2025-01-01 00:00:00 UTC in seconds
		int64_t microseconds = PlannerTimeRange::unix_time_to_microseconds(unix_time);
		CHECK(microseconds == 1735689600000000LL);
	}

	SUBCASE("Calculate duration from start and end") {
		int64_t start = 1735689600000000LL;
		int64_t end = 1735689601000000LL;
		time_range.set_start_time(start);
		time_range.set_end_time(end);
		time_range.calculate_duration();
		CHECK(time_range.get_duration() == 1000000LL); // 1 second
	}
}

TEST_CASE("[Modules][Temporal] PlannerTaskMetadata temporal updates") {
	Ref<PlannerTaskMetadata> metadata = memnew(PlannerTaskMetadata);

	SUBCASE("Metadata time tracking with absolute microseconds") {
		int64_t absolute_time = 1735689600000000LL;
		metadata->update_metadata(absolute_time);
		PlannerTimeRange time_range = metadata->get_time_range();
		CHECK(time_range.get_start_time() == absolute_time);
	}

	SUBCASE("Multiple updates with absolute microseconds") {
		int64_t time1 = 1735689600000000LL;
		int64_t time2 = 1735689601000000LL;
		metadata->update_metadata(time1);
		metadata->update_metadata(time2);
		PlannerTimeRange time_range = metadata->get_time_range();
		CHECK(time_range.get_start_time() == time2); // Last update wins
	}

	memdelete(metadata.ptr());
}

TEST_CASE("[Modules][Temporal] PlannerPlan temporal integration") {
	PlannerPlan plan;

	SUBCASE("Plan time range management with absolute microseconds") {
		PlannerTimeRange time_range;
		int64_t start = 1735689600000000LL;
		int64_t duration = 500000LL; // 0.5 seconds
		time_range.set_start_time(start);
		time_range.set_duration(duration);
		time_range.calculate_end_from_duration();

		plan.set_time_range(time_range);
		PlannerTimeRange retrieved = plan.get_time_range();

		CHECK(retrieved.get_start_time() == start);
		CHECK(retrieved.get_duration() == duration);
		CHECK(retrieved.get_end_time() == start + duration);
	}

	SUBCASE("Plan temporal state with absolute microseconds") {
		// Test that plan maintains temporal state
		PlannerTimeRange time_range;
		int64_t start = 1735689600000000LL;
		time_range.set_start_time(start);
		plan.set_time_range(time_range);

		// Simulate some operation
		PlannerTimeRange updated_time_range = plan.get_time_range();
		int64_t end = start + 1000000LL; // 1 second later
		updated_time_range.set_end_time(end);
		plan.set_time_range(updated_time_range);

		PlannerTimeRange final_time_range = plan.get_time_range();
		CHECK(final_time_range.get_start_time() == start);
		CHECK(final_time_range.get_end_time() == end);
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
	PlannerTimeRange time_range;

	SUBCASE("Valid time ranges with absolute microseconds") {
		int64_t start = 1735689600000000LL;
		int64_t end = 1735689601000000LL; // 1 second later
		int64_t duration = 1000000LL;
		time_range.set_start_time(start);
		time_range.set_end_time(end);
		time_range.set_duration(duration);
		CHECK(time_range.get_start_time() < time_range.get_end_time());
		CHECK(time_range.get_duration() == time_range.get_end_time() - time_range.get_start_time());
	}

	SUBCASE("Edge case: zero duration") {
		int64_t time = 1735689600000000LL;
		time_range.set_start_time(time);
		time_range.set_end_time(time);
		time_range.set_duration(0);
		CHECK(time_range.get_start_time() == time_range.get_end_time());
		CHECK(time_range.get_duration() == 0);
	}

	SUBCASE("Time arithmetic with absolute microseconds") {
		int64_t base_time = 1735689600000000LL;
		int64_t duration = 5000000LL; // 5 seconds
		int64_t end_time = base_time + duration;

		time_range.set_start_time(base_time);
		time_range.set_duration(duration);
		time_range.calculate_end_from_duration();
		CHECK(time_range.get_end_time() == end_time);
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
