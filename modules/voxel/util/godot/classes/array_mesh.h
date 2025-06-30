/**************************************************************************/
/*  array_mesh.h                                                          */
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
#include <scene/resources/mesh.h>
#elif defined(ZN_GODOT_EXTENSION)
#include <godot_cpp/classes/array_mesh.hpp>
using namespace godot;
#endif

namespace zylann::godot {

// TODO The following functions should be able to work on `Mesh`,
// but the script/extension API exposes some methods only on `ArrayMesh`, even though they exist on `Mesh` internally...

// TODO I need a cheap way to check this at `Mesh` level, but it seems it would require getting the surface arrays,
// which might not be cheap...
inline bool is_mesh_empty(const ArrayMesh &mesh) {
	if (mesh.get_surface_count() == 0) {
		return true;
	}
	if (mesh.surface_get_array_len(0) == 0) {
		return true;
	}
	return false;
}

#ifdef TOOLS_ENABLED

// Generates a wireframe-mesh that highlights edges of a triangle-mesh where vertices are not shared.
// Used for debugging.
Array generate_debug_seams_wireframe_surface(const ArrayMesh &src_mesh, int surface_index);

#endif

} // namespace zylann::godot
