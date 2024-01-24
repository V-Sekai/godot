/**************************************************************************/
/*  cube_tables.cpp                                                       */
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

#include "cube_tables.h"

namespace Cube {
// The following tables respect the following conventions
//
//    7-------6
//   /|      /|
//  / |     / |  Corners
// 4-------5  |
// |  3----|--2
// | /     | /     y z
// |/      |/      |/   OpenGL axis convention
// 0-------1    x--o
//
//
//     o---10----o
//    /|        /|
//  11 7       9 6   Edges
//  /  |      /  |
// o----8----o   |
// |   o---2-|---o
// 4  /      5  /
// | 3       | 1
// |/        |/
// o----0----o
//
// Sides are ordered according to the Voxel::Side enum.
// Edges are ordered according to the Voxel::Edge enum (only g_edge_inormals!).
//

// Ordered as per the cube corners diagram
const Vector3 g_corner_position[CORNER_COUNT] = {
	Vector3(1, 0, 0), Vector3(0, 0, 0), Vector3(0, 0, 1), Vector3(1, 0, 1),
	Vector3(1, 1, 0), Vector3(0, 1, 0), Vector3(0, 1, 1), Vector3(1, 1, 1)
};

const int g_side_quad_triangles[SIDE_COUNT][6] = {
	{ 0, 2, 1, 0, 3, 2 }, // LEFT (+x)
	{ 0, 2, 1, 0, 3, 2 }, // RIGHT (-x)
	{ 0, 2, 1, 0, 3, 2 }, // BOTTOM (-y)
	{ 0, 2, 1, 0, 3, 2 }, // TOP (+y)
	{ 0, 2, 1, 0, 3, 2 }, // BACK (-z)
	{ 0, 2, 1, 0, 3, 2 }, // FRONT (+z)
};

// const unsigned int g_side_coord[SIDE_COUNT] = { 0, 0, 1, 1, 2, 2 };
// const unsigned int g_side_sign[SIDE_COUNT] = { 0, 1, 0, 1, 0, 1 };

const VoxelVector3i g_side_normals[SIDE_COUNT] = {
	// TODO Wrong! Left should be -X
	VoxelVector3i(1, 0, 0), // LEFT
	VoxelVector3i(-1, 0, 0), // RIGHT

	VoxelVector3i(0, -1, 0), // BOTTOM
	VoxelVector3i(0, 1, 0), // TOP

	// TODO Wrong! Front should be -Z
	VoxelVector3i(0, 0, -1), // BACK
	VoxelVector3i(0, 0, 1), // FRONT
};

const float g_side_tangents[SIDE_COUNT][4] = {
	{ 0.f, 0.f, -1.f, 1.f }, { 0.f, 0.f, 1.f, 1.f },

	{ 1.f, 0.f, 0.f, 1.f }, { -1.f, 0.f, 0.f, 1.f },

	{ -1.f, 0.f, 0.f, 1.f }, { 1.f, 0.f, 0.f, 1.f }
};

// Corners have same winding, relative to the face's normal
const unsigned int g_side_corners[SIDE_COUNT][4] = { { 3, 0, 4, 7 }, { 1, 2, 6, 5 },
	{ 1, 0, 3, 2 }, { 4, 5, 6, 7 },
	{ 0, 1, 5, 4 }, { 2, 3, 7, 6 } };

const unsigned int g_side_edges[SIDE_COUNT][4] = { { 3, 7, 11, 4 }, { 1, 6, 9, 5 },
	{ 0, 1, 2, 3 }, { 8, 9, 10, 11 },
	{ 0, 5, 8, 4 }, { 2, 6, 10, 7 } };

// 3---2
// | / | {0,1,2,0,2,3}
// 0---1
// static const unsigned int g_vertex_to_corner[Voxel::SIDE_COUNT][6] = {
//    { 0, 3, 7, 0, 7, 4 },
//    { 2, 1, 5, 2, 5, 6 },
//    { 0, 1, 2, 0, 2, 3 },
//    { 7, 6, 5, 7, 5, 4 },
//    { 1, 0, 4 ,1, 4, 5 },
//    { 3, 2, 6, 3, 6, 7 }
//};

const VoxelVector3i g_corner_inormals[CORNER_COUNT] = {
	VoxelVector3i(1, -1, -1), VoxelVector3i(-1, -1, -1),
	VoxelVector3i(-1, -1, 1), VoxelVector3i(1, -1, 1),

	VoxelVector3i(1, 1, -1), VoxelVector3i(-1, 1, -1),
	VoxelVector3i(-1, 1, 1), VoxelVector3i(1, 1, 1)
};

const VoxelVector3i g_edge_inormals[EDGE_COUNT] = {
	VoxelVector3i(0, -1, -1), VoxelVector3i(-1, -1, 0),
	VoxelVector3i(0, -1, 1), VoxelVector3i(1, -1, 0),

	VoxelVector3i(1, 0, -1), VoxelVector3i(-1, 0, -1),
	VoxelVector3i(-1, 0, 1), VoxelVector3i(1, 0, 1),

	VoxelVector3i(0, 1, -1), VoxelVector3i(-1, 1, 0),
	VoxelVector3i(0, 1, 1), VoxelVector3i(1, 1, 0)
};

const unsigned int g_edge_corners[EDGE_COUNT][2] = {
	{ 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 }, { 0, 4 }, { 1, 5 },
	{ 2, 6 }, { 3, 7 }, { 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 }
};

// Order is irrelevant
const VoxelVector3i g_moore_neighboring_3d[MOORE_NEIGHBORING_3D_COUNT] = {
	VoxelVector3i(-1, -1, -1),
	VoxelVector3i(0, -1, -1),
	VoxelVector3i(1, -1, -1),
	VoxelVector3i(-1, -1, 0),
	VoxelVector3i(0, -1, 0),
	VoxelVector3i(1, -1, 0),
	VoxelVector3i(-1, -1, 1),
	VoxelVector3i(0, -1, 1),
	VoxelVector3i(1, -1, 1),

	VoxelVector3i(-1, 0, -1),
	VoxelVector3i(0, 0, -1),
	VoxelVector3i(1, 0, -1),
	VoxelVector3i(-1, 0, 0),
	// VoxelVector3i(0,0,0),
	VoxelVector3i(1, 0, 0),
	VoxelVector3i(-1, 0, 1),
	VoxelVector3i(0, 0, 1),
	VoxelVector3i(1, 0, 1),

	VoxelVector3i(-1, 1, -1),
	VoxelVector3i(0, 1, -1),
	VoxelVector3i(1, 1, -1),
	VoxelVector3i(-1, 1, 0),
	VoxelVector3i(0, 1, 0),
	VoxelVector3i(1, 1, 0),
	VoxelVector3i(-1, 1, 1),
	VoxelVector3i(0, 1, 1),
	VoxelVector3i(1, 1, 1),
};

// Order is IMPORTANT:
// This is used in multithread context, in which we may iterate blocks in XYZ
// order, to avoid deadlocks.
const VoxelVector3i g_ordered_moore_area_3d[MOORE_AREA_3D_COUNT] = {
	VoxelVector3i(-1, -1, -1), VoxelVector3i(0, -1, -1),
	VoxelVector3i(1, -1, -1), VoxelVector3i(-1, 0, -1),
	VoxelVector3i(0, 0, -1), VoxelVector3i(1, 0, -1),
	VoxelVector3i(-1, 1, -1), VoxelVector3i(0, 1, -1),
	VoxelVector3i(1, 1, -1),

	VoxelVector3i(-1, -1, 0), VoxelVector3i(0, -1, 0),
	VoxelVector3i(1, -1, 0), VoxelVector3i(-1, 0, 0),
	VoxelVector3i(0, 0, 0), VoxelVector3i(1, 0, 0),
	VoxelVector3i(-1, 1, 0), VoxelVector3i(0, 1, 0),
	VoxelVector3i(1, 1, 0),

	VoxelVector3i(-1, -1, 1), VoxelVector3i(0, -1, 1),
	VoxelVector3i(1, -1, 1), VoxelVector3i(-1, 0, 1),
	VoxelVector3i(0, 0, 1), VoxelVector3i(1, 0, 1),
	VoxelVector3i(-1, 1, 1), VoxelVector3i(0, 1, 1),
	VoxelVector3i(1, 1, 1)
};
} // namespace Cube
