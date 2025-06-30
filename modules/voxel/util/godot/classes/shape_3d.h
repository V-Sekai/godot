/**************************************************************************/
/*  shape_3d.h                                                            */
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

#include "../core/version.h"
#include "../macros.h"

#if defined(ZN_GODOT)

#if GODOT_VERSION_MAJOR == 4 && GODOT_VERSION_MINOR <= 2
#include <scene/resources/shape_3d.h>
#else
#include <scene/resources/3d/shape_3d.h>
#endif

#elif defined(ZN_GODOT_EXTENSION)
#include <godot_cpp/classes/shape3d.hpp>
using namespace godot;
#endif

ZN_GODOT_FORWARD_DECLARE(class SceneTree);

namespace zylann::godot {

#ifdef DEBUG_ENABLED

inline void set_shape_3d_debug_color(Shape3D &shape, const Color color) {
	// TODO GDX: `set_debug_color` is not exposed to GDExtensions
	// Which means there is no way for us to show debug shapes of the terrain when the option is enabled, because they
	// default to transparent black, which is invisible
#if defined(ZN_GODOT)
#if GODOT_VERSION_MAJOR == 4 && GODOT_VERSION_MINOR >= 4
	shape.set_debug_color(color);
#endif
#endif
}

// This function is primarily to implement a workaround...
// See https://github.com/godotengine/godot/pull/100328
Color get_shape_3d_default_color(const SceneTree &scene_tree);

#endif

} // namespace zylann::godot
