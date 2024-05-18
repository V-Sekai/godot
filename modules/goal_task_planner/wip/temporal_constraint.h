/**************************************************************************/
/*  temporal_constraint.h                                                 */
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

#ifndef TEMPORAL_CONSTRAINT_H
#define TEMPORAL_CONSTRAINT_H
// Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
// K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
// temporal_constraint.h
// SPDX-License-Identifier: MIT

#include "core/io/resource.h"
#include "core/math/vector2i.h"

class TemporalConstraint : public Resource {
	GDCLASS(TemporalConstraint, Resource);

public:
	enum TemporalQualifier {
		AT_START,
		AT_END,
		OVERALL,
	};

private:
	Vector2i time_interval;
	int duration;
	TemporalQualifier temporal_qualifier;
	String resource_name;

public:
	void _init(int start, int end, int duration, int qualifier, String resource);
	bool overlaps_with(Ref<TemporalConstraint> other);
	String _to_string();

	Vector2i get_time_interval() const { return time_interval; }
	int get_duration() const { return duration; }
	int get_temporal_qualifier() const { return temporal_qualifier; }

	void set_time_interval(Vector2i interval) { time_interval = interval; }
	void set_duration(int dur) { duration = dur; }
	void set_temporal_qualifier(int qualifier) { temporal_qualifier = TimeQualifier(qualifier); }

protected:
	static void _bind_methods();
};

#endif // TEMPORAL_CONSTRAINT_H
