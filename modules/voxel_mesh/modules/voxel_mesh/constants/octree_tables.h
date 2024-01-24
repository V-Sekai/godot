/**************************************************************************/
/*  octree_tables.h                                                       */
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

#ifndef OCTREE_TABLES_H
#define OCTREE_TABLES_H

namespace OctreeTables {
//  Corners:                                    Octants:
//
//         6---------------18--------------7       o---o---o
//        /               /               /|       | 6 | 7 |
//       /               /               / |       o---o---o  Upper
//      17--------------25--------------19 |       | 5 | 4 |
//     /               /               /   |       o---o---o
//    /               /               /    |
//   5---------------16--------------4     |       o---o---o
//   |     14--------|-----23--------|-----15      | 2 | 3 |
//   |    /          |    /          |    /|       o---o---o  Lower        Z
//   |   /           |   /           |   / |       | 1 | 0 |               |
//   |  22-----------|--26-----------|--24 |       o---o---o           X---o
//   | /             | /             | /   |
//   |/              |/              |/    |
//   13--------------21--------------12    |
//   |     2---------|-----10--------|-----3
//   |    /          |    /          |    /
//   |   /           |   /           |   /
//   |  9------------|--20-----------|--11           Y
//   | /             | /             | /             | Z
//   |/              |/              |/              |/
//   1---------------8---------------0          X----o

// The order is important for some algorithms (DMC)
const int g_octant_position[8][3]{ { 0, 0, 0 }, { 1, 0, 0 }, { 1, 0, 1 }, { 0, 0, 1 },

	{ 0, 1, 0 }, { 1, 1, 0 }, { 1, 1, 1 }, { 0, 1, 1 } };
} // namespace OctreeTables

#endif // OCTREE_TABLES_H
