/**************************************************************************/
/*  thread.h                                                              */
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

struct ThreadImpl;

class Thread {
public:
	enum Priority { //
		PRIORITY_LOW,
		PRIORITY_NORMAL,
		PRIORITY_HIGH
	};

	typedef uint64_t ID;

	typedef void (*Callback)(void *p_userdata);

	Thread();
	~Thread();

	void start(Callback p_callback, void *p_userdata, Priority priority = PRIORITY_NORMAL);
	bool is_started() const;
	void wait_to_finish();

	// Gets a hint of the number of concurrent threads natively supported
	static unsigned int get_hardware_concurrency();

	// Targets the current thread
	static void set_name(const char *name);
	static void sleep_usec(uint32_t microseconds);

	// Get ID of the current thread
	static ID get_caller_id();

private:
	ThreadImpl *_impl = nullptr;
};

} // namespace zylann
