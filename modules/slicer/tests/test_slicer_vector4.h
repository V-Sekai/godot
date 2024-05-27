/**************************************************************************/
/*  test_slicer_vector4.h                                                 */
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

#if 0

#include "../utils/slicer_vector4.h"

TEST_CASE("[SlicerVector4]") {
	SECTION("x,y,z,w maps to coord") {
		SlicerVector4 vec = SlicerVector4(0.5, 1.5, 2.5, 3.5);
		REQUIRE(vec.x == 0.5);
		REQUIRE(vec.y == 1.5);
		REQUIRE(vec.z == 2.5);
		REQUIRE(vec.w == 3.5);

		REQUIRE(vec.coord[0] == 0.5);
		REQUIRE(vec.coord[1] == 1.5);
		REQUIRE(vec.coord[2] == 2.5);
		REQUIRE(vec.coord[3] == 3.5);

		vec.x = 5.0;

		REQUIRE(vec.x == 5.0);
		REQUIRE(vec.coord[0] == 5.0);
	}

	SECTION("operator*") {
		SlicerVector4 vec = SlicerVector4(1, 2, 3, 4) * 2;

		REQUIRE(vec.x == 2);
		REQUIRE(vec.y == 4);
		REQUIRE(vec.z == 6);
		REQUIRE(vec.w == 8);
	}

	SECTION("operator*+") {
		SlicerVector4 vec = SlicerVector4(1, 2, 3, 4) + SlicerVector4(2, 3, 4, 5);

		REQUIRE(vec.x == 3);
		REQUIRE(vec.y == 5);
		REQUIRE(vec.z == 7);
		REQUIRE(vec.w == 9);
	}

	SECTION("operator==") {
		REQUIRE(SlicerVector4(1, 1, 1, 1) == SlicerVector4(1, 1, 1, 1));
		REQUIRE_FALSE(SlicerVector4(1, 1, 1, 1) == SlicerVector4(2, 1, 1, 1));
		REQUIRE_FALSE(SlicerVector4(1, 1, 1, 1) == SlicerVector4(1, 2, 1, 1));
		REQUIRE_FALSE(SlicerVector4(1, 1, 1, 1) == SlicerVector4(1, 1, 2, 1));
		REQUIRE_FALSE(SlicerVector4(1, 1, 1, 1) == SlicerVector4(1, 1, 1, 2));
	}
}

#endif
