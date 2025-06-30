/**************************************************************************/
/*  file_access.h                                                         */
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
#include <core/io/file_access.h>

#elif defined(ZN_GODOT_EXTENSION)
#include "../core/packed_arrays.h"
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/global_constants.hpp> // For `Error`
using namespace godot;
#endif

#include "../../containers/span.h"

namespace zylann::godot {

inline Ref<FileAccess> open_file(const String path, FileAccess::ModeFlags mode_flags, Error &out_error) {
#if defined(ZN_GODOT)
	return FileAccess::open(path, mode_flags, &out_error);
#elif defined(ZN_GODOT_EXTENSION)
	Ref<FileAccess> file = FileAccess::open(path, mode_flags);
	out_error = FileAccess::get_open_error();
	if (out_error != ::godot::OK) {
		return Ref<FileAccess>();
	} else {
		return file;
	}
#endif
}

inline uint64_t get_buffer(FileAccess &f, Span<uint8_t> dst) {
#if defined(ZN_GODOT)
	return f.get_buffer(dst.data(), dst.size());
#elif defined(ZN_GODOT_EXTENSION)
	PackedByteArray bytes = f.get_buffer(dst.size());
	copy_to(dst, bytes);
	return bytes.size();
#endif
}

inline void store_buffer(FileAccess &f, Span<const uint8_t> src) {
#if defined(ZN_GODOT)
	f.store_buffer(src.data(), src.size());
#elif defined(ZN_GODOT_EXTENSION)
	PackedByteArray bytes;
	copy_to(bytes, src);
	f.store_buffer(bytes);
#endif
}

inline String get_as_text(FileAccess &f) {
#if defined(ZN_GODOT)
	return f.get_as_utf8_string();
#elif defined(ZN_GODOT_EXTENSION)
	return f.get_as_text();
#endif
}

} // namespace zylann::godot
