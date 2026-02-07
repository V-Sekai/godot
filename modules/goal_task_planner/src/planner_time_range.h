/**************************************************************************/
/*  planner_time_range.h                                                  */
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

#pragma once

#include "core/os/os.h"
#include "core/typedefs.h"

struct PlannerTimeRange {
	// All times in absolute microseconds since Unix epoch (1970-01-01 00:00:00 UTC)
	static constexpr double MICROSECONDS_PER_SECOND = 1000000.0; // Conversion factor from seconds to microseconds

	int64_t start_time = 0; // Absolute microseconds
	int64_t end_time = 0; // Absolute microseconds
	int64_t duration = 0; // Duration in microseconds

	PlannerTimeRange() {}

	void set_start_time(int64_t p_time) { start_time = p_time; }
	int64_t get_start_time() const { return start_time; }

	void set_end_time(int64_t p_time) { end_time = p_time; }
	int64_t get_end_time() const { return end_time; }

	void set_duration(int64_t p_duration) { duration = p_duration; }
	int64_t get_duration() const { return duration; }

	// Helper methods for absolute time conversion
	// Convert from Godot's unix time (seconds) to microseconds
	static int64_t unix_time_to_microseconds(double p_unix_time) {
		return static_cast<int64_t>(p_unix_time * MICROSECONDS_PER_SECOND);
	}

	// Get current absolute time in microseconds
	static int64_t now_microseconds() {
		return unix_time_to_microseconds(OS::get_singleton()->get_unix_time());
	}

	// Set start time to current absolute time
	void set_start_now() {
		start_time = now_microseconds();
	}

	// Set end time to current absolute time
	void set_end_now() {
		end_time = now_microseconds();
	}

	// Calculate duration from start_time to end_time
	void calculate_duration() {
		if (end_time > start_time) {
			duration = end_time - start_time;
		} else {
			duration = 0;
		}
	}

	// Calculate end_time from start_time and duration
	void calculate_end_from_duration() {
		end_time = start_time + duration;
	}
};
