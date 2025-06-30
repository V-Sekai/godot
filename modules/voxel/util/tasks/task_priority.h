/**************************************************************************/
/*  task_priority.h                                                       */
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

#include <cstdint>

namespace zylann {

// Represents the priorirty of a task, which can be compared quickly to another.
struct TaskPriority {
	static const uint8_t BAND_MAX = 255;

	union {
		struct {
			uint8_t band0; // Higher means higher priority (to state the obvious)
			uint8_t band1; // Takes precedence over band0
			uint8_t band2; // Takes precedence over band1
			uint8_t band3; // Takes precedence over band2
		};
		uint32_t whole;
	};

	TaskPriority() : whole(0) {}

	TaskPriority(uint8_t p_band0, uint8_t p_band1, uint8_t p_band2, uint8_t p_band3) :
			band0(p_band0), band1(p_band1), band2(p_band2), band3(p_band3) {}

	// Returns `true` if the left-hand priority is lower than the right-hand one.
	// Means the right-hand task should run first.
	inline bool operator<(const TaskPriority &other) const {
		return whole < other.whole;
	}

	// Returns `true` if the left-hand priority is higher than the right-hand one.
	// Means the left-hand task should run first.
	inline bool operator>(const TaskPriority &other) const {
		return whole > other.whole;
	}

	inline bool operator==(const TaskPriority &other) const {
		return whole == other.whole;
	}

	static inline TaskPriority min() {
		TaskPriority p;
		p.whole = 0;
		return p;
	}

	static inline TaskPriority max() {
		TaskPriority p;
		p.whole = 0xffffffff;
		return p;
	}
};

} // namespace zylann
