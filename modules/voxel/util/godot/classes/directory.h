/**************************************************************************/
/*  directory.h                                                           */
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
#include <core/io/dir_access.h>
#elif defined(ZN_GODOT_EXTENSION)
#include <godot_cpp/classes/dir_access.hpp>
using namespace godot;
#endif

namespace zylann::godot {

inline Ref<DirAccess> open_directory(const String &directory_path, Error *out_err) {
	Ref<DirAccess> dir = DirAccess::open(directory_path);
	if (out_err != nullptr) {
		*out_err = DirAccess::get_open_error();
	}
	return dir;
}

inline bool directory_exists(const String &directory_path) {
	return DirAccess::dir_exists_absolute(directory_path);
}

inline bool directory_exists(DirAccess &dir, const String &relative_directory_path) {
	// Why this function is not `const`, I wonder
	return dir.dir_exists(relative_directory_path);
}

inline Error rename_directory(const String &from, const String &to) {
	return DirAccess::rename_absolute(from, to);
}

} // namespace zylann::godot
