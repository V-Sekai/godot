/**************************************************************************/
/*  godot_thread_helper.h                                                 */
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

#ifndef ZN_GODOT_EXTENSION
#error "This class is exclusive to Godot Extension"
#endif

#include "../errors.h"
#include "../thread/thread.h"
#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/core/class_db.hpp>

namespace zylann {

// Proxy-object to run a C-style callback using GDExtension threads.
// This class isn't intented to be exposed.
//
// It is recommended to use Godot threads instead of vanilla std::thread,
// because Godot sets up additional stuff in `Thread` (like script debugging and platform-specific stuff to set
// priority). To use Godot threads in GDExtension, you are FORCED to send an object method as callback. And to do that,
// the object must be registered.
class ZN_GodotThreadHelper : public ::godot::Object {
	GDCLASS(ZN_GodotThreadHelper, ::godot::Object)
public:
	ZN_GodotThreadHelper() {}

	void set_callback(Thread::Callback callback, void *data) {
		_callback = callback;
		_callback_data = data;
	}

private:
	void run();

	static void _bind_methods();

	Thread::Callback _callback = nullptr;
	void *_callback_data = nullptr;
};

} // namespace zylann
