/**************************************************************************/
/*  stn_constraints.h                                                     */
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

#include "core/typedefs.h"
#include "core/variant/variant.h"
#include "stn_solver.h"

// Helper class for converting intervals (start/end/duration) to STN constraints
// Provides convenient methods for adding temporal intervals to STN solver

class PlannerSTNConstraints {
public:
	// Add an interval with start time, end time, and duration
	// Creates time points: {p_id}_start and {p_id}_end
	// Adds constraint: start -> end: {duration, duration}
	// If absolute times provided, anchors to origin time point
	static bool add_interval(PlannerSTNSolver &p_stn, const String &p_id, int64_t p_start_time, int64_t p_end_time, int64_t p_duration);

	// Anchor a time point to absolute time (relative to origin)
	// If origin doesn't exist, creates it
	static bool anchor_to_origin(PlannerSTNSolver &p_stn, const String &p_point, int64_t p_absolute_time);

private:
	// Helper to ensure origin time point exists
	static void ensure_origin(PlannerSTNSolver &p_stn);
};
