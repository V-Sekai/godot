/**************************************************************************/
/*  test_surface_filler.h                                                 */
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

#include "../utils/surface_filler.h"

TEST_CASE("[surface_filler]") {
	SECTION("adds a new surface to an array") {
		SlicerFace face_1(Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(1, 0, 1));
		SlicerFace face_2(Vector3(1, 0, 1), Vector3(0, 0, 1), Vector3(0, 0, 0));

		face_1.set_uvs(Vector2(0, 0), Vector2(1, 0), Vector2(1, 1));
		face_2.set_uvs(Vector2(1, 1), Vector2(0, 1), Vector2(0, 0));

		face_1.set_tangents(SlicerVector4(1, 0, 0, 1), SlicerVector4(1, 0, 0, 1), SlicerVector4(1, 0, 0, 1));
		face_2.set_tangents(SlicerVector4(1, 0, 0, 1), SlicerVector4(1, 0, 0, 1), SlicerVector4(1, 0, 0, 1));

		PoolVector<SlicerFace> faces;
		faces.push_back(face_1);
		faces.push_back(face_2);

		SurfaceFiller filler(faces);
		for (int i = 0; i < 6; i++) {
			filler.fill(i, i);
		}

		ArrayMesh mesh;
		Ref<SpatialMaterial> material;
		material.instance();

		filler.add_to_mesh(mesh, material);
		REQUIRE(mesh.get_surface_count() == 1);
		REQUIRE(mesh.surface_get_material(0) == material);
		REQUIRE(mesh.surface_get_primitive_type(0) == Mesh::PRIMITIVE_TRIANGLES);

		Array arrays = mesh.surface_get_arrays(0);

		PoolVector<Vector3> vertices = arrays[Mesh::ARRAY_VERTEX];
		PoolVector<Vector2> uvs = arrays[Mesh::ARRAY_TEX_UV];
		PoolVector<real_t> tangents = arrays[Mesh::ARRAY_TANGENT];
		PoolVector<Vector3> normals = arrays[Mesh::ARRAY_NORMAL];
		PoolVector<Color> colors = arrays[Mesh::ARRAY_COLOR];
		PoolVector<real_t> uv2s = arrays[Mesh::ARRAY_TEX_UV2];
		PoolVector<real_t> weights = arrays[Mesh::ARRAY_WEIGHTS];
		PoolVector<real_t> bones = arrays[Mesh::ARRAY_BONES];
		PoolVector<int> indices = arrays[Mesh::ARRAY_INDEX];

		REQUIRE(vertices.size() == 6);
		REQUIRE(uvs.size() == 6);
		REQUIRE(tangents.size() == 6 * 4);
		REQUIRE(normals.size() == 0);
		REQUIRE(colors.size() == 0);
		REQUIRE(uv2s.size() == 0);
		REQUIRE(weights.size() == 0);
		REQUIRE(bones.size() == 0);
		REQUIRE(indices.size() == 0);

		REQUIRE(vertices[0] == Vector3(0, 0, 0));
		REQUIRE(vertices[1] == Vector3(1, 0, 0));
		REQUIRE(vertices[2] == Vector3(1, 0, 1));
		REQUIRE(vertices[3] == Vector3(1, 0, 1));
		REQUIRE(vertices[4] == Vector3(0, 0, 1));
		REQUIRE(vertices[5] == Vector3(0, 0, 0));

		REQUIRE(uvs[0] == Vector2(0, 0));
		REQUIRE(uvs[1] == Vector2(1, 0));
		REQUIRE(uvs[2] == Vector2(1, 1));
		REQUIRE(uvs[3] == Vector2(1, 1));
		REQUIRE(uvs[4] == Vector2(0, 1));
		REQUIRE(uvs[5] == Vector2(0, 0));

		for (int i = 0; i < 6 * 4; i += 4) {
			REQUIRE(tangents[i + 0] == 1);
			REQUIRE(tangents[i + 1] == 0);
			REQUIRE(tangents[i + 2] == 0);
			REQUIRE(tangents[i + 3] == 1);
		}
	}
}

#endif
