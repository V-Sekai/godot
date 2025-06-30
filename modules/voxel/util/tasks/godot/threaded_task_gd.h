/**************************************************************************/
/*  threaded_task_gd.h                                                    */
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

#include "../../godot/classes/ref_counted.h"
#include "../../godot/core/gdvirtual.h"
#include "../threaded_task.h"

namespace zylann {

class ZN_ThreadedTaskInternal;

class ZN_ThreadedTask : public RefCounted {
	GDCLASS(ZN_ThreadedTask, RefCounted)
public:
	void run(int thread_index);
	int get_priority();
	bool is_cancelled();

	// Internal
	bool is_scheduled() const;
	void mark_completed();
	IThreadedTask *create_task();

private:
	GDVIRTUAL1(_run, int);
	GDVIRTUAL0R(int, _get_priority);
	GDVIRTUAL0R(bool, _is_cancelled);

	static void _bind_methods();

	// Created upon scheduling, owned by the task runner
	ZN_ThreadedTaskInternal *_scheduled_task = nullptr;
	bool _completed = false;
};

} // namespace zylann
