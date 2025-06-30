/**************************************************************************/
/*  resource.h                                                            */
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

#if defined(ZN_GODOT)
#include <core/io/resource.h>
#elif defined(ZN_GODOT_EXTENSION)
#include <godot_cpp/classes/resource.hpp>
using namespace godot;
#endif

namespace zylann::godot {

// Godot doesn't have configuration warnings yet on resources.
// But when we add them and they are nested, it can be difficult to put them into context when the warning appears in
// the scene tree. This helper prepends context so we can nest configuration warnings in resources.
// Context is a callable template returning something convertible to a String, so it only gets evaluated when there
// actually are warnings.
template <typename TResource, typename FContext>
inline void get_resource_configuration_warnings(
		const TResource &resource,
		PackedStringArray &warnings,
		FContext get_context_string_func
) {
	const int prev_size = warnings.size();

	// This method is by us, not Godot.
	resource.get_configuration_warnings(warnings);

	const int current_size = warnings.size();
	if (current_size != prev_size) {
		// New warnings were added
		String context = get_context_string_func();
		for (int i = prev_size; i < current_size; ++i) {
			const String w = context + warnings[i];
#if defined(ZN_GODOT)
			warnings.write[i] = w;
#elif defined(ZN_GODOT_EXTENSION)
			warnings[i] = w;
#endif
		}
	}
}

} // namespace zylann::godot
