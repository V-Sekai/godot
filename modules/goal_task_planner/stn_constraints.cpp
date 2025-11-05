/**************************************************************************/
/*  stn_constraints.cpp                                                   */
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

#include "stn_constraints.h"
#include "core/string/print_string.h"

void PlannerSTNConstraints::ensure_origin(PlannerSTNSolver &p_stn) {
	if (!p_stn.has_time_point("origin")) {
		p_stn.add_time_point("origin");
	}
}

bool PlannerSTNConstraints::add_interval(PlannerSTNSolver &p_stn, const String &p_id, int64_t p_start_time, int64_t p_end_time, int64_t p_duration) {
	String start_point = p_id + "_start";
	String end_point = p_id + "_end";

	// Add time points
	p_stn.add_time_point(start_point);
	p_stn.add_time_point(end_point);

	// Add duration constraint: start -> end: {duration, duration}
	// Duration is in microseconds, ensure it's positive
	if (p_duration < 0) {
		return false;
	}

	bool success = p_stn.add_constraint(start_point, end_point, p_duration, p_duration);
	if (!success) {
		return false;
	}

	// If absolute times provided, anchor to origin
	if (p_start_time > 0) {
		ensure_origin(p_stn);
		// origin -> start: {start_time, start_time}
		success = p_stn.add_constraint("origin", start_point, p_start_time, p_start_time);
		if (!success) {
			return false;
		}
	}

	if (p_end_time > 0) {
		ensure_origin(p_stn);
		// origin -> end: {end_time, end_time}
		success = p_stn.add_constraint("origin", end_point, p_end_time, p_end_time);
		if (!success) {
			return false;
		}
	}

	return true;
}

bool PlannerSTNConstraints::add_durative_action(PlannerSTNSolver &p_stn, const String &p_action_id, int64_t p_duration) {
	String start_point = p_action_id + "_start";
	String end_point = p_action_id + "_end";

	// Add time points
	p_stn.add_time_point(start_point);
	p_stn.add_time_point(end_point);

	// Add duration constraint: start -> end: {duration, duration}
	if (p_duration < 0) {
		return false;
	}

	return p_stn.add_constraint(start_point, end_point, p_duration, p_duration);
}

bool PlannerSTNConstraints::add_temporal_relation(PlannerSTNSolver &p_stn, const String &p_from, const String &p_to, const String &p_relation) {
	String from_point = p_from;
	String to_point = p_to;

	// Ensure points exist (they should be full names like "action1_start")
	if (!p_from.ends_with("_start") && !p_from.ends_with("_end")) {
		from_point = p_from + "_start";
	}
	if (!p_to.ends_with("_start") && !p_to.ends_with("_end")) {
		to_point = p_to + "_start";
	}

	if (p_relation == "before") {
		// from ends before to starts: from_end -> to_start: {0, infinity}
		// But we want to ensure from_end <= to_start, so: from_end -> to_start: {0, infinity}
		// Actually, "before" means from must complete before to starts, so: from_end -> to_start: {0, infinity}
		String from_end = p_from;
		String to_start = p_to;
		if (!from_end.ends_with("_end")) {
			from_end = p_from + "_end";
		}
		if (!to_start.ends_with("_start")) {
			to_start = p_to + "_start";
		}

		// Constraint: to_start - from_end >= 0 (from_end <= to_start)
		// Which means: from_end -> to_start: {0, infinity}
		return p_stn.add_constraint(from_end, to_start, 0, INT64_MAX);
	} else if (p_relation == "after") {
		// to starts after from ends: to_start -> from_end: {0, infinity}
		String from_end = p_from;
		String to_start = p_to;
		if (!from_end.ends_with("_end")) {
			from_end = p_from + "_end";
		}
		if (!to_start.ends_with("_start")) {
			to_start = p_to + "_start";
		}

		// Constraint: from_end - to_start <= 0 (from_end <= to_start)
		// Which means: to_start -> from_end: {0, infinity} (same as "before" reversed)
		return p_stn.add_constraint(to_start, from_end, 0, INT64_MAX);
	} else if (p_relation == "during") {
		// from happens during to: to_start <= from_start and from_end <= to_end
		String from_start = p_from;
		String from_end = p_from;
		String to_start = p_to;
		String to_end = p_to;

		if (!from_start.ends_with("_start")) {
			from_start = p_from + "_start";
		}
		if (!from_end.ends_with("_end")) {
			from_end = p_from + "_end";
		}
		if (!to_start.ends_with("_start")) {
			to_start = p_to + "_start";
		}
		if (!to_end.ends_with("_end")) {
			to_end = p_to + "_end";
		}

		// to_start <= from_start: from_start -> to_start: {0, infinity}
		// from_end <= to_end: from_end -> to_end: {0, infinity}
		bool success1 = p_stn.add_constraint(from_start, to_start, 0, INT64_MAX);
		bool success2 = p_stn.add_constraint(from_end, to_end, 0, INT64_MAX);
		return success1 && success2;
	}

	return false; // Unknown relation
}

bool PlannerSTNConstraints::anchor_to_origin(PlannerSTNSolver &p_stn, const String &p_point, int64_t p_absolute_time) {
	ensure_origin(p_stn);

	// Add constraint: origin -> point: {absolute_time, absolute_time}
	return p_stn.add_constraint("origin", p_point, p_absolute_time, p_absolute_time);
}
