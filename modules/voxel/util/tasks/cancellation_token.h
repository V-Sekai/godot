/**************************************************************************/
/*  cancellation_token.h                                                  */
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

#include "../errors.h"
#include "../memory/memory.h"
#include <atomic>

namespace zylann {

// Simple object shared between a task and the requester of the task. Allows the requester to cancel the task before it
// runs or finishes.
class TaskCancellationToken {
public:
	// TODO Could be optimized
	// - Pointer to an atomic refcount?
	// - Index into a [paged] pool of atomic ints?

	static TaskCancellationToken create() {
		TaskCancellationToken token;
		token._cancelled = make_shared_instance<std::atomic_bool>(false);
		return token;
	}

	inline bool is_valid() const {
		return _cancelled != nullptr;
	}

	inline void cancel() {
#ifdef TOOLS_ENABLED
		ZN_ASSERT(_cancelled != nullptr);
#endif
		*_cancelled = true;
	}

	inline bool is_cancelled() const {
#ifdef TOOLS_ENABLED
		ZN_ASSERT(_cancelled != nullptr);
#endif
		return *_cancelled;
	}

private:
	std::shared_ptr<std::atomic_bool> _cancelled;
};

} // namespace zylann
