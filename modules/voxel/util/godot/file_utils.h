/**************************************************************************/
/*  file_utils.h                                                          */
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

#include "../math/vector3i.h"
#include "classes/file_access.h"
#include "core/string.h"

namespace zylann::godot {

inline Vector3i get_vec3u8(FileAccess &f) {
	Vector3i v;
	v.x = f.get_8();
	v.y = f.get_8();
	v.z = f.get_8();
	return v;
}

inline void store_vec3u8(FileAccess &f, const Vector3i v) {
	f.store_8(v.x);
	f.store_8(v.y);
	f.store_8(v.z);
}

inline Vector3i get_vec3u32(FileAccess &f) {
	Vector3i v;
	v.x = f.get_32();
	v.y = f.get_32();
	v.z = f.get_32();
	return v;
}

inline void store_vec3u32(FileAccess &f, const Vector3i v) {
	f.store_32(v.x);
	f.store_32(v.y);
	f.store_32(v.z);
}

enum FileResult { //
	FILE_OK = 0,
	FILE_CANT_OPEN,
	FILE_DOES_NOT_EXIST,
	FILE_UNEXPECTED_EOF,
	FILE_INVALID_DATA
};

const char *to_string(FileResult res);

Error check_directory_created(const String &p_directory_path);

void insert_bytes(FileAccess &f, size_t count, size_t temp_chunk_size = 512);

} // namespace zylann::godot
