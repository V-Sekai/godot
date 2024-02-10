/**************************************************************************/
/*  cube_tables.h                                                         */
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

#ifndef CUBE_TABLES_H
#define CUBE_TABLES_H

#include "../util/math/voxel_vector3i.h"
#include <core/math/vector3.h>

namespace Cube {
// Index convention used in some lookup tables
enum Side {
	SIDE_LEFT = 0,
	SIDE_RIGHT,
	SIDE_BOTTOM,
	SIDE_TOP,
	// TODO I would like to swap the Z ones because it's inconsistent with
	// coordinates...
	SIDE_BACK,
	SIDE_FRONT,

	SIDE_COUNT
};
// Alias to the above for clarity, fixing some interpretation problems regarding
// the side_normals table...
enum SideAxis {
	SIDE_POSITIVE_X = 0,
	SIDE_NEGATIVE_X,
	SIDE_NEGATIVE_Y,
	SIDE_POSITIVE_Y,
	SIDE_NEGATIVE_Z,
	SIDE_POSITIVE_Z
};

// Index convention used in some lookup tables
enum Edge {
	EDGE_BOTTOM_BACK = 0,
	EDGE_BOTTOM_RIGHT,
	EDGE_BOTTOM_FRONT,
	EDGE_BOTTOM_LEFT,
	EDGE_BACK_LEFT,
	EDGE_BACK_RIGHT,
	EDGE_FRONT_RIGHT,
	EDGE_FRONT_LEFT,
	EDGE_TOP_BACK,
	EDGE_TOP_RIGHT,
	EDGE_TOP_FRONT,
	EDGE_TOP_LEFT,

	EDGE_COUNT
};

// Index convention used in some lookup tables
enum Corner {
	CORNER_BOTTOM_BACK_LEFT = 0,
	CORNER_BOTTOM_BACK_RIGHT,
	CORNER_BOTTOM_FRONT_RIGHT,
	CORNER_BOTTOM_FRONT_LEFT,
	CORNER_TOP_BACK_LEFT,
	CORNER_TOP_BACK_RIGHT,
	CORNER_TOP_FRONT_RIGHT,
	CORNER_TOP_FRONT_LEFT,

	CORNER_COUNT
};

extern const Vector3 g_corner_position[CORNER_COUNT];

extern const int g_side_quad_triangles[SIDE_COUNT][6];

// extern const unsigned int g_side_coord[SIDE_COUNT];
// extern const unsigned int g_side_sign[SIDE_COUNT];

extern const VoxelVector3i g_side_normals[SIDE_COUNT];
extern const float g_side_tangents[SIDE_COUNT][4];

extern const unsigned int g_side_corners[SIDE_COUNT][4];
extern const unsigned int g_side_edges[SIDE_COUNT][4];

extern const VoxelVector3i g_corner_inormals[CORNER_COUNT];

extern const VoxelVector3i g_edge_inormals[EDGE_COUNT];

extern const unsigned int g_edge_corners[EDGE_COUNT][2];

const unsigned int MOORE_NEIGHBORING_3D_COUNT = 26;
extern const VoxelVector3i g_moore_neighboring_3d[MOORE_NEIGHBORING_3D_COUNT];

const unsigned int MOORE_AREA_3D_COUNT = 27;
const unsigned int MOORE_AREA_3D_CENTRAL_INDEX = 13;
extern const VoxelVector3i g_ordered_moore_area_3d[MOORE_AREA_3D_COUNT];
} // namespace Cube

#endif // CUBE_TABLES_H
