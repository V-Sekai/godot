/**************************************************************************/
/*  test_slicer_face.h                                                    */
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

#ifndef TEST_SLICER_FACE_H
#define TEST_SLICER_FACE_H

#include "scene/resources/mesh.h"
#include "tests/test_macros.h"

#include "../utils/slicer_face.h"
#include "scene/resources/3d/primitive_meshes.h"

namespace TestSlicerFace {

float rand(int max) {
	return Math::random(0, max);
}

int rounded() {
	return Math::round(rand(1));
}

Array make_test_array(int faces) {
	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);

	Vector<Vector3> points;
	Vector<Vector3> normals;
	Vector<Color> colors;
	Vector<real_t> tangents;
	Vector<Vector2> uvs;

	for (int i = 0; i < faces * 3; i++) {
		points.push_back(Vector3(rand(10), rand(10), rand(10)));
		normals.push_back(Vector3(rounded(), rounded(), rounded()));
		colors.push_back(Color(rounded(), rounded(), rounded(), rounded()));

		tangents.push_back(rounded());
		tangents.push_back(rounded());
		tangents.push_back(rounded());
		tangents.push_back(rounded());

		uvs.push_back(Vector2(rounded(), rounded()));
	}

	arrays[Mesh::ARRAY_VERTEX] = points;
	arrays[Mesh::ARRAY_NORMAL] = normals;
	arrays[Mesh::ARRAY_COLOR] = colors;
	arrays[Mesh::ARRAY_TANGENT] = tangents;
	arrays[Mesh::ARRAY_TEX_UV] = uvs;

	return arrays;
}

TEST_SUITE("[SlicerFace]") {
	TEST_SUITE("faces_from_surface") {
		TEST_CASE("[Modules][Slicer][SceneTree] Parses faces similar to built in method") {
			Ref<SphereMesh> sphere_mesh = memnew(SphereMesh);
			auto control_faces = sphere_mesh->get_faces();
			Vector<SlicerFace> faces = SlicerFace::faces_from_surface(sphere_mesh, 0);
			REQUIRE(faces.size() == control_faces.size());
			for (int i = 0; i < faces.size(); i++) {
				REQUIRE(faces[i] == control_faces[i]);
			}
		}

		// TEST_CASE("[Modules][Slicer][SceneTree] With non indexed arrays") {
		// 	Ref<ArrayMesh> array_mesh = memnew(ArrayMesh);
		// 	Array arrays = make_test_array(3);
		// 	array_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
		// 	Vector<SlicerFace> faces = SlicerFace::faces_from_surface(array_mesh, 0);
		// 	REQUIRE(faces.size() == 3);

		// 	Vector<Vector3> points = arrays[Mesh::ARRAY_VERTEX];
		// 	Vector<Vector3> normals = arrays[Mesh::ARRAY_NORMAL];
		// 	Vector<Color> colors = arrays[Mesh::ARRAY_COLOR];
		// 	Vector<real_t> tangents = arrays[Mesh::ARRAY_TANGENT];
		// 	Vector<Vector2> uvs = arrays[Mesh::ARRAY_TEX_UV];

		// 	for (int i = 0; i < 3; i++) {
		// 		REQUIRE(faces[i].has_normals == true);
		// 		REQUIRE(faces[i].has_tangents == true);
		// 		REQUIRE(faces[i].has_colors == true);
		// 		REQUIRE(faces[i].has_bones == false);
		// 		REQUIRE(faces[i].has_weights == false);
		// 		REQUIRE(faces[i].has_uvs == true);
		// 		REQUIRE(faces[i].has_uv2s == false);

		// 		for (int j = 0; j < 3; j++) {
		// 			REQUIRE(faces[i].vertex[j] == points[i * 3 + j].snapped(Vector3(0.0001, 0.0001, 0.0001)));
		// 			REQUIRE(faces[i].normal[j] == normals[i * 3 + j].snapped(Vector3(0.0001, 0.0001, 0.0001)));
		// 			REQUIRE(faces[i].color[j] == colors[i * 3 + j]);

		// 			real_t tangent0 = tangents[(i * 3 * 4) + (j * 4) + 0];
		// 			real_t tangent1 = tangents[(i * 3 * 4) + (j * 4) + 1];
		// 			real_t tangent2 = tangents[(i * 3 * 4) + (j * 4) + 2];
		// 			real_t tangent3 = tangents[(i * 3 * 4) + (j * 4) + 3];

		// 			REQUIRE(Math::is_equal_approx(faces[i].tangent[j][0], Math::snapped(tangent0, 0.0001)));
		// 			REQUIRE(Math::is_equal_approx(faces[i].tangent[j][1], Math::snapped(tangent1, 0.0001)));
		// 			REQUIRE(Math::is_equal_approx(faces[i].tangent[j][2], Math::snapped(tangent2, 0.0001)));
		// 			REQUIRE(Math::is_equal_approx(faces[i].tangent[j][3], Math::snapped(tangent3, 0.0001)));

		// 			REQUIRE(faces[i].uv[j] == uvs[i * 3 + j]);
		// 		}
		// 	}
		// }

		// 	TEST_CASE("[Modules][Slicer][SceneTree] With indexed arrays") {
		// 		Ref<ArrayMesh> array_mesh = memnew(ArrayMesh);
		// 		Array arrays = make_test_array(24);
		// 		int idxs[9] = { 1, 4, 7, 10, 13, 14, 17, 19, 21 };
		// 		Vector<int> indices;
		// 		for (int i = 0; i < 9; i++) {
		// 			indices.push_back(idxs[i]);
		// 		}

		// 		arrays[Mesh::ARRAY_INDEX] = indices;
		// 		array_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
		// 		Vector<SlicerFace> faces = SlicerFace::faces_from_surface(array_mesh, 0);
		// 		REQUIRE(faces.size() == 3);

		// 		Vector<Vector3> points = arrays[Mesh::ARRAY_VERTEX];
		// 		Vector<Vector3> normals = arrays[Mesh::ARRAY_NORMAL];
		// 		Vector<Color> colors = arrays[Mesh::ARRAY_COLOR];
		// 		Vector<real_t> tangents = arrays[Mesh::ARRAY_TANGENT];
		// 		Vector<Vector2> uvs = arrays[Mesh::ARRAY_TEX_UV];

		// 		for (int i = 0; i < 9; i++) {
		// 			int face = i / 3;
		// 			REQUIRE(faces[face].has_normals == true);
		// 			REQUIRE(faces[face].has_tangents == true);
		// 			REQUIRE(faces[face].has_colors == true);
		// 			REQUIRE(faces[face].has_bones == false);
		// 			REQUIRE(faces[face].has_weights == false);
		// 			REQUIRE(faces[face].has_uvs == true);
		// 			REQUIRE(faces[face].has_uv2s == false);

		// 			REQUIRE(faces[face].vertex[i % 3] == points[idxs[i]].snapped(Vector3(0.0001, 0.0001, 0.0001)));
		// 			REQUIRE(faces[face].normal[i % 3] == normals[idxs[i]]);
		// 			REQUIRE(faces[face].color[i % 3] == colors[idxs[i]]);

		// 			REQUIRE(faces[face].tangent[i % 3][0] == tangents[(idxs[i] * 4)]);
		// 			REQUIRE(faces[face].tangent[i % 3][1] == tangents[(idxs[i] * 4) + 1]);
		// 			REQUIRE(faces[face].tangent[i % 3][2] == tangents[(idxs[i] * 4) + 2]);
		// 			REQUIRE(faces[face].tangent[i % 3][3] == tangents[(idxs[i] * 4) + 3]);

		// 			REQUIRE(faces[face].uv[i % 3] == uvs[idxs[i]]);
		// 		}
		// 	}
	}
	TEST_CASE("[Modules][Slicer] barycentric_weights") {
		SlicerFace face(Vector3(0, 0, 0), Vector3(2, 0, 0), Vector3(2, 2, 0));
		Vector3 weights = face.barycentric_weights(Vector3(1, 1, 0));
		REQUIRE(weights == Vector3(0.5, 0, 0.5));
	}

	// TEST_CASE("[Modules][Slicer] compute_tangents") {
	// 	SlicerFace face(Vector3(0, 0, 0), Vector3(2, 0, 0), Vector3(2, 2, 0));

	// 	// Doesn't work without uvs and normals
	// 	face.compute_tangents();
	// 	REQUIRE_FALSE(face.has_tangents);

	// 	face.set_normals(Vector3(0, 0, 0), Vector3(0, 1, 0), Vector3(1, 1, 1));
	// 	face.set_uvs(Vector2(0, 0), Vector2(1, 0), Vector2(0, 1));
	// 	face.compute_tangents();

	// 	REQUIRE(face.has_tangents);
	// 	REQUIRE(face.tangent[0] == Vector4(1, 0, 0, 1));
	// 	REQUIRE(face.tangent[1] == Vector4(1, 0, 0, 1));
	// 	REQUIRE(face.tangent[2] == Vector4(0.816496551, -0.408248246, -0.408248246, 1));
	// }

	TEST_CASE("[Modules][Slicer] sub_face") {
		SlicerFace face(Vector3(0, 0, 0), Vector3(0, 1, 0), Vector3(0, 1, 1));
		face.set_uvs(Vector2(0, 0), Vector2(1, 0), Vector2(1, 1));
		SlicerFace sub_face = face.sub_face(Vector3(0, 0, 0), Vector3(0, 0.5, 0), Vector3(0, 0.5, 0.5));
		REQUIRE(sub_face.vertex[0] == Vector3(0, 0, 0));
		REQUIRE(sub_face.vertex[1] == Vector3(0, 0.5, 0));
		REQUIRE(sub_face.vertex[2] == Vector3(0, 0.5, 0.5));

		REQUIRE(sub_face.has_uvs);
		REQUIRE(sub_face.uv[0] == Vector2(0, 0));
		REQUIRE(sub_face.uv[1] == Vector2(0.5, 0));
		REQUIRE(sub_face.uv[2] == Vector2(0.5, 0.5));

		REQUIRE_FALSE(sub_face.has_normals);
		REQUIRE_FALSE(sub_face.has_tangents);
		REQUIRE_FALSE(sub_face.has_colors);
		REQUIRE_FALSE(sub_face.has_uv2s);
		REQUIRE_FALSE(sub_face.has_bones);
		REQUIRE_FALSE(sub_face.has_weights);
	}

	TEST_CASE("[Modules][Slicer] set_uvs") {
		SlicerFace face;
		face.set_uvs(Vector2(0, 0), Vector2(0.5, 0.5), Vector2(1, 1));
		REQUIRE(face.has_uvs);
		REQUIRE(face.uv[0] == Vector2(0, 0));
		REQUIRE(face.uv[1] == Vector2(0.5, 0.5));
		REQUIRE(face.uv[2] == Vector2(1, 1));
	}

	TEST_CASE("[Modules][Slicer] set_normals") {
		SlicerFace face;
		face.set_normals(Vector3(0, 0, 0), Vector3(0.5, 0.5, 0.5), Vector3(1, 1, 1));
		REQUIRE(face.has_normals);
		REQUIRE(face.normal[0] == Vector3(0, 0, 0));
		REQUIRE(face.normal[1] == Vector3(0.5, 0.5, 0.5));
		REQUIRE(face.normal[2] == Vector3(1, 1, 1));
	}

	TEST_CASE("[Modules][Slicer] set_tangents") {
		SlicerFace face;
		face.set_tangents(Vector4(0, 0, 0, 0), Vector4(0.5, 0.5, 0.5, 0.5), Vector4(1, 1, 1, 1));
		REQUIRE(face.has_tangents);
		REQUIRE(face.tangent[0] == Vector4(0, 0, 0, 0));
		REQUIRE(face.tangent[1] == Vector4(0.5, 0.5, 0.5, 0.5));
		REQUIRE(face.tangent[2] == Vector4(1, 1, 1, 1));
	}

	TEST_CASE("[Modules][Slicer] set_colors") {
		SlicerFace face;
		face.set_colors(Color(0, 0, 0, 0), Color(0.5, 0.5, 0.5, 0.5), Color(1, 1, 1, 1));
		REQUIRE(face.has_colors);
		REQUIRE(face.color[0] == Color(0, 0, 0, 0));
		REQUIRE(face.color[1] == Color(0.5, 0.5, 0.5, 0.5));
		REQUIRE(face.color[2] == Color(1, 1, 1, 1));
	}

	TEST_CASE("[Modules][Slicer] set_bones") {
		SlicerFace face;
		face.set_bones(Vector4(0, 0, 0, 0), Vector4(0.5, 0.5, 0.5, 0.5), Vector4(1, 1, 1, 1));
		REQUIRE(face.has_bones);
		REQUIRE(face.bones[0] == Vector4(0, 0, 0, 0));
		REQUIRE(face.bones[1] == Vector4(0.5, 0.5, 0.5, 0.5));
		REQUIRE(face.bones[2] == Vector4(1, 1, 1, 1));
	}

	TEST_CASE("[Modules][Slicer] set_weights") {
		SlicerFace face;
		face.set_weights(Vector4(0, 0, 0, 0), Vector4(0.5, 0.5, 0.5, 0.5), Vector4(1, 1, 1, 1));
		REQUIRE(face.has_weights);
		REQUIRE(face.weights[0] == Vector4(0, 0, 0, 0));
		REQUIRE(face.weights[1] == Vector4(0.5, 0.5, 0.5, 0.5));
		REQUIRE(face.weights[2] == Vector4(1, 1, 1, 1));
	}

	TEST_CASE("[Modules][Slicer] set_uv2s") {
		SlicerFace face;
		face.set_uv2s(Vector2(0, 0), Vector2(0.5, 0.5), Vector2(1, 1));
		REQUIRE(face.has_uv2s);
		REQUIRE(face.uv2[0] == Vector2(0, 0));
		REQUIRE(face.uv2[1] == Vector2(0.5, 0.5));
		REQUIRE(face.uv2[2] == Vector2(1, 1));
	}

	TEST_CASE("[Modules][Slicer] operator==") {
		REQUIRE(SlicerFace(Vector3(0, 0, 0), Vector3(1, 1, 1), Vector3(2, 2, 2)) == SlicerFace(Vector3(0, 0, 0), Vector3(1, 1, 1), Vector3(2, 2, 2)));
		REQUIRE_FALSE(SlicerFace(Vector3(0, 0, 0), Vector3(1, 1, 1), Vector3(2, 2, 2)) == SlicerFace(Vector3(1, 0, 0), Vector3(1, 1, 1), Vector3(2, 2, 2)));
		REQUIRE_FALSE(SlicerFace(Vector3(0, 0, 0), Vector3(1, 1, 1), Vector3(2, 2, 2)) == SlicerFace(Vector3(0, 0, 0), Vector3(0, 1, 1), Vector3(2, 2, 2)));
		REQUIRE_FALSE(SlicerFace(Vector3(0, 0, 0), Vector3(1, 1, 1), Vector3(2, 2, 2)) == SlicerFace(Vector3(0, 0, 0), Vector3(1, 1, 1), Vector3(0, 2, 2)));
	}
}
} //namespace TestSlicerFace

#endif // TEST_SLICER_FACE_H
