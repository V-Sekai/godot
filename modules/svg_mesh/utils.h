/**************************************************************************/
/*  utils.h                                                               */
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

#ifndef UTILS_H
#define UTILS_H

#include "core/io/file_access.h"
#include "scene/2d/mesh_instance_2d.h"
#include <stdint.h>

#include "modules/svg_mesh/thirdparty/tove2d/src/cpp/graphics.h"
#include "modules/svg_mesh/thirdparty/tove2d/src/cpp/paint.h"
#include "modules/svg_mesh/thirdparty/tove2d/src/cpp/path.h"
#include "modules/svg_mesh/thirdparty/tove2d/src/cpp/subpath.h"

inline Rect2 tove_bounds_to_rect2(const float *bounds) {
	return Rect2(bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]);
}

tove::PathRef new_transformed_path(const tove::PathRef &p_tove_path, const Transform2D &p_transform);

Ref<ShaderMaterial> copy_mesh(
		Ref<ArrayMesh> &p_mesh,
		tove::MeshRef &p_tove_mesh,
		const tove::GraphicsRef &p_graphics,
		Ref<Texture> &r_texture,
		bool p_spatial = false);

#endif // UTILS_H
