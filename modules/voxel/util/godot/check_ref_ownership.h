/**************************************************************************/
/*  check_ref_ownership.h                                                 */
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

#ifdef TOOLS_ENABLED
#include "../errors.h"
#include "../macros.h"
#include "../string/format.h"
#include "classes/ref_counted.h"
#include "core/string.h"

namespace zylann::godot {

// Checks that nothing takes extra ownership of a RefCounted object between the beginning and the end of a scope.
// This can be used when calling GDVIRTUAL methods that are passed an object that must not be held by the callee after
// the end of the call.
class CheckRefCountDoesNotChange {
public:
	// Ideally this shouldn't need to be turned off in project settings, but some languages like C# like to keep
	// references on object due to garbage-collection memory model. In those cases, the strategy of this sanity check
	// falls apart because we can't tell what happened...
	static void set_enabled(bool enabled);
	static bool is_enabled();

	// Note: not taking a `const Ref<RefCounted>&` for convenience, because it may involve casting, which means C++ will
	// not pass Ref<T> by reference but by value instead, which would increase the refcount.
	inline CheckRefCountDoesNotChange(const char *method_name, RefCounted *rc) :
			_method_name(method_name), _rc(rc), _initial_count(rc->get_reference_count()) {}

	inline ~CheckRefCountDoesNotChange() {
		if (!is_enabled()) {
			return;
		}
		const int after_count = _rc->get_reference_count();
		if (after_count != _initial_count && !was_reported()) {
			mark_reported();
			ZN_PRINT_ERROR(
					format("Holding a reference to the passed {} outside {} is not allowed (count before: {}, "
						   "count after: {}). If you are using a garbage-collected language (like C#), "
						   "you may want to turn off this check in ProjectSettings.",
						   _rc->get_class(),
						   _method_name,
						   _initial_count,
						   after_count)
			);
		}
	}

private:
	static bool was_reported();
	static void mark_reported();

	const char *_method_name;
	const RefCounted *_rc;
	const int _initial_count;
};

} // namespace zylann::godot

#define ZN_GODOT_CHECK_REF_COUNT_DOES_NOT_CHANGE(m_ref)                                                                \
	ZN_ASSERT(m_ref.is_valid());                                                                                       \
	zylann::godot::CheckRefCountDoesNotChange ZN_CONCAT(ref_count_checker_, __LINE__)(__FUNCTION__, m_ref.ptr())

#else // TOOLS_ENABLED

#define ZN_GODOT_CHECK_REF_COUNT_DOES_NOT_CHANGE(m_ref)

#endif // TOOLS_ENABLED
