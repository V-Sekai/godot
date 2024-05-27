/**************************************************************************/
/*  test_intersector.h                                                    */
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

#ifndef TEST_INTERSECTOR_H
#define TEST_INTERSECTOR_H

#include "tests/test_macros.h"

namespace TestIntersector {

#include "../utils/intersector.h"
#include "scene/resources/3d/primitive_meshes.h"

TEST_SUITE("[get_side_of]") {
	// A plane with a normal pointing directly up, 5 units off of the origin
	Plane plane(Vector3(0, 1, 0), 5);
	TEST_CASE("Finds points under plane") {
		Vector3 point(0, 0, 0);
		REQUIRE(Intersector::get_side_of(plane, point) == Intersector::SideOfPlane::UNDER);
	}
	TEST_CASE("Finds points over plane") {
		Vector3 point(0, 6, 0);
		REQUIRE(Intersector::get_side_of(plane, point) == Intersector::SideOfPlane::OVER);
	}
	TEST_CASE("Finds points on plane") {
		Vector3 point(1, 5, 1);
		REQUIRE(Intersector::get_side_of(plane, point) == Intersector::SideOfPlane::ON);
	}
}

TEST_SUITE("[split_face_by_plane]") {
	Plane plane(Vector3(0, 1, 0), 0);

	TEST_CASE("Smoke test") {
		SphereMesh sphere_mesh;
		Vector<SlicerFace> faces = SlicerFace::faces_from_surface(sphere_mesh, 0);
		REQUIRE(faces.size() == 4224);
		Intersector::SplitResult result;
		for (int i = 0; i < faces.size(); i++) {
			Intersector::split_face_by_plane(plane, faces[i], result);
		}
		REQUIRE(result.lower_faces.size() == 2240);
		REQUIRE(result.upper_faces.size() == 2240);
		REQUIRE(result.intersection_points.size() == 256);
	}

	TEST_CASE("points_all_on_same_side") {
		Intersector::SplitResult result;
		Intersector::split_face_by_plane(plane, SlicerFace(Vector3(0, 1, 0), Vector3(1, 2, 0), Vector3(2, 1, 0)), result);
		REQUIRE(result.upper_faces.size() == 1);
		REQUIRE(result.lower_faces.size() == 0);
		REQUIRE(result.intersection_points.size() == 0);
		result.reset();

		Intersector::split_face_by_plane(plane, SlicerFace(Vector3(0, -1, 0), Vector3(1, -2, 0), Vector3(2, -1, 0)), result);
		REQUIRE(result.upper_faces.size() == 0);
		REQUIRE(result.lower_faces.size() == 1);
		REQUIRE(result.intersection_points.size() == 0);
	}

	TEST_CASE("one_side_is_parallel") {
		Intersector::SplitResult result;
		Intersector::split_face_by_plane(plane, SlicerFace(Vector3(0, 0, 0), Vector3(1, 1, 0), Vector3(2, 0, 0)), result);
		REQUIRE(result.upper_faces.size() == 1);
		REQUIRE(result.lower_faces.size() == 0);
		REQUIRE(result.intersection_points.size() == 0);
		result.reset();

		Intersector::split_face_by_plane(plane, SlicerFace(Vector3(0, 0, 0), Vector3(1, -2, 0), Vector3(2, 0, 0)), result);
		REQUIRE(result.upper_faces.size() == 0);
		REQUIRE(result.lower_faces.size() == 1);
		REQUIRE(result.intersection_points.size() == 0);
	}

	TEST_CASE("pointed_away") {
		Intersector::SplitResult result;
		Intersector::split_face_by_plane(plane, SlicerFace(Vector3(0, 1, 0), Vector3(1, 0, 0), Vector3(2, 1, 0)), result);
		REQUIRE(result.upper_faces.size() == 1);
		REQUIRE(result.lower_faces.size() == 0);
		REQUIRE(result.intersection_points.size() == 0);
		result.reset();

		Intersector::split_face_by_plane(plane, SlicerFace(Vector3(0, -1, 0), Vector3(1, 0, 0), Vector3(2, -1, 0)), result);
		REQUIRE(result.upper_faces.size() == 0);
		REQUIRE(result.lower_faces.size() == 1);
		REQUIRE(result.intersection_points.size() == 0);
	}

	TEST_SUITE("face_split_in_half") {
		TEST_CASE("point a is on plane") {
			Intersector::SplitResult result;
			Intersector::split_face_by_plane(plane, SlicerFace(Vector3(0, 0, 0), Vector3(1, 1, 0), Vector3(1, -1, 0)), result);
			REQUIRE(result.upper_faces.size() == 1);
			REQUIRE(result.lower_faces.size() == 1);
			REQUIRE(result.intersection_points.size() == 2);
			REQUIRE(result.intersection_points[0] == Vector3(0, 0, 0));
			REQUIRE(result.intersection_points[1] == Vector3(1, 0, 0));
			REQUIRE(result.upper_faces[0] == SlicerFace(Vector3(0, 0, 0), Vector3(1, 1, 0), Vector3(1, 0, 0)));
			REQUIRE(result.lower_faces[0] == SlicerFace(Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(1, -1, 0)));
		}

		TEST_CASE("point b is on plane") {
			Intersector::SplitResult result;
			Intersector::split_face_by_plane(plane, SlicerFace(Vector3(0, -1, 0), Vector3(1, 0, 0), Vector3(0, 1, 0)), result);
			REQUIRE(result.upper_faces.size() == 1);
			REQUIRE(result.lower_faces.size() == 1);
			REQUIRE(result.intersection_points.size() == 2);
			REQUIRE(result.intersection_points[0] == Vector3(1, 0, 0));
			REQUIRE(result.intersection_points[1] == Vector3(0, 0, 0));
			REQUIRE(result.upper_faces[0] == SlicerFace(Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 0)));
			REQUIRE(result.lower_faces[0] == SlicerFace(Vector3(1, 0, 0), Vector3(0, 0, 0), Vector3(0, -1, 0)));
		}

		TEST_CASE("point c is on plane") {
			Intersector::SplitResult result;
			Intersector::split_face_by_plane(plane, SlicerFace(Vector3(0, 1, 0), Vector3(0, -1, 0), Vector3(1, 0, 0)), result);
			REQUIRE(result.upper_faces.size() == 1);
			REQUIRE(result.lower_faces.size() == 1);
			REQUIRE(result.intersection_points.size() == 2);
			REQUIRE(result.intersection_points[0] == Vector3(1, 0, 0));
			REQUIRE(result.intersection_points[1] == Vector3(0, 0, 0));
			REQUIRE(result.upper_faces[0] == SlicerFace(Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 0)));
			REQUIRE(result.lower_faces[0] == SlicerFace(Vector3(1, 0, 0), Vector3(0, 0, 0), Vector3(0, -1, 0)));
		}
	}

	TEST_SUITE("full_split") {
		TEST_CASE("point a is lone") {
			Intersector::SplitResult result;
			Intersector::split_face_by_plane(plane, SlicerFace(Vector3(1, 1, 0), Vector3(2, -1, 0), Vector3(0, -1, 0)), result);
			REQUIRE(result.upper_faces.size() == 1);
			REQUIRE(result.lower_faces.size() == 2);
			REQUIRE(result.intersection_points.size() == 2);
			REQUIRE(result.intersection_points[0] == Vector3(1.5, 0, 0));
			REQUIRE(result.intersection_points[1] == Vector3(0.5, 0, 0));
			REQUIRE(result.upper_faces[0] == SlicerFace(Vector3(1, 1, 0), Vector3(1.5, 0, 0), Vector3(0.5, 0, 0)));
			REQUIRE(result.lower_faces[0] == SlicerFace(Vector3(2, -1, 0), Vector3(0.5, 0, 0), Vector3(1.5, 0, 0)));
			REQUIRE(result.lower_faces[1] == SlicerFace(Vector3(0, -1, 0), Vector3(0.5, 0, 0), Vector3(2, -1, 0)));
		}
		TEST_CASE("point b is lone") {
			Intersector::SplitResult result;
			Intersector::split_face_by_plane(plane, SlicerFace(Vector3(0, -1, 0), Vector3(1, 1, 0), Vector3(2, -1, 0)), result);
			REQUIRE(result.upper_faces.size() == 1);
			REQUIRE(result.lower_faces.size() == 2);
			REQUIRE(result.intersection_points.size() == 2);
			REQUIRE(result.intersection_points[0] == Vector3(0.5, 0, 0));
			REQUIRE(result.intersection_points[1] == Vector3(1.5, 0, 0));
			REQUIRE(result.upper_faces[0] == SlicerFace(Vector3(1, 1, 0), Vector3(1.5, 0, 0), Vector3(0.5, 0, 0)));
			REQUIRE(result.lower_faces[0] == SlicerFace(Vector3(2, -1, 0), Vector3(0.5, 0, 0), Vector3(1.5, 0, 0)));
			REQUIRE(result.lower_faces[1] == SlicerFace(Vector3(0, -1, 0), Vector3(0.5, 0, 0), Vector3(2, -1, 0)));
		}
		TEST_CASE("point c is lone") {
			Intersector::SplitResult result;
			Intersector::split_face_by_plane(plane, SlicerFace(Vector3(2, -1, 0), Vector3(0, -1, 0), Vector3(1, 1, 0)), result);
			REQUIRE(result.upper_faces.size() == 1);
			REQUIRE(result.lower_faces.size() == 2);
			REQUIRE(result.intersection_points.size() == 2);
			REQUIRE(result.intersection_points[0] == Vector3(1.5, 0, 0));
			REQUIRE(result.intersection_points[1] == Vector3(0.5, 0, 0));
			REQUIRE(result.upper_faces[0] == SlicerFace(Vector3(1, 1, 0), Vector3(1.5, 0, 0), Vector3(0.5, 0, 0)));
			REQUIRE(result.lower_faces[0] == SlicerFace(Vector3(2, -1, 0), Vector3(0.5, 0, 0), Vector3(1.5, 0, 0)));
			REQUIRE(result.lower_faces[1] == SlicerFace(Vector3(0, -1, 0), Vector3(0.5, 0, 0), Vector3(2, -1, 0)));
		}
	}
}
} //namespace TestIntersector

#endif // TEST_INTERSECTOR_H
