/**************************************************************************/
/*  test_triangulator.h                                                   */
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

#include "../utils/triangulator.h"

TEST_CASE("[triangulator]") {
	SECTION("monotone_chain") {
		PoolVector<Vector3> interception_points;
		interception_points.push_back(Vector3(0, 0, 0));
		interception_points.push_back(Vector3(1, 0, 0));
		interception_points.push_back(Vector3(1, 0, 1));
		interception_points.push_back(Vector3(0, 0, 1));
		interception_points.push_back(Vector3(0.5, 0, 0.5));

		PoolVector<SlicerFace> faces = Triangulator::monotone_chain(interception_points, Vector3(0, 1, 0));
		REQUIRE(faces.size() == 2);
		REQUIRE(faces[0] == SlicerFace(Vector3(1, 0, 1), Vector3(0, 0, 1), Vector3(0, 0, 0)));
		REQUIRE(faces[1] == SlicerFace(Vector3(1, 0, 1), Vector3(0, 0, 0), Vector3(1, 0, 0)));

		REQUIRE((faces[0].has_normals && faces[0].has_uvs && faces[0].has_tangents));
		REQUIRE((faces[1].has_normals && faces[1].has_uvs && faces[1].has_tangents));

		REQUIRE((faces[0].normal[0] == Vector3(0, 1, 0) && faces[0].normal[1] == Vector3(0, 1, 0) && faces[0].normal[2] == Vector3(0, 1, 0)));
		REQUIRE((faces[1].normal[0] == Vector3(0, 1, 0) && faces[1].normal[1] == Vector3(0, 1, 0) && faces[1].normal[2] == Vector3(0, 1, 0)));

		REQUIRE((faces[0].uv[0] == Vector2(0, 0) && faces[0].uv[1] == Vector2(1, 0) && faces[0].uv[2] == Vector2(1, 1)));
		REQUIRE((faces[1].uv[0] == Vector2(0, 0) && faces[1].uv[1] == Vector2(1, 1) && faces[1].uv[2] == Vector2(0, 1)));

		REQUIRE(faces[0].tangent[0] == Vector4(-1, 0, 0, -1));
		REQUIRE(faces[0].tangent[1] == Vector4(-1, 0, 0, -1));
		REQUIRE(faces[0].tangent[2] == Vector4(-1, 0, 0, -1));

		REQUIRE(faces[1].tangent[0] == Vector4(-1, 0, 0, -1));
		REQUIRE(faces[1].tangent[1] == Vector4(-1, 0, 0, -1));
		REQUIRE(faces[1].tangent[2] == Vector4(-1, 0, 0, -1));
	}
}

#endif
