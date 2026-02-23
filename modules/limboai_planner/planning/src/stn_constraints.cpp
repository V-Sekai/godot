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

bool PlannerSTNConstraints::anchor_to_origin(PlannerSTNSolver &p_stn, const String &p_point, int64_t p_absolute_time) {
	ensure_origin(p_stn);

	// Don't anchor origin to itself (origin is always at time 0)
	if (p_point == "origin") {
		// If trying to set origin to non-zero time, this is invalid
		if (p_absolute_time != 0) {
			return false;
		}
		return true; // Origin at time 0 is already the default
	}

	// Add constraint: origin -> point: {absolute_time, absolute_time}
	return p_stn.add_constraint("origin", p_point, p_absolute_time, p_absolute_time);
}
