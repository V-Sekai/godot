/**************************************************************************/
/*  mesh_merge_triangle.h                                                 */
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

#ifndef MESH_MERGE_TRIANGLE_H
#define MESH_MERGE_TRIANGLE_H

#include "core/math/vector2.h"
#include "core/math/vector3.h"

/// A callback to sample the environment. Return false to terminate rasterization.
typedef bool (*MeshMergeSamplingCallback)(void *param, int x, int y, const Vector3 &bar, const Vector3 &dx, const Vector3 &dy, float coverage);

struct MeshMergeTriangle {
	MeshMergeTriangle(const Vector2 &v0, const Vector2 &v1, const Vector2 &v2, const Vector3 &t0, const Vector3 &t1, const Vector3 &t2);
	/// Compute texture space deltas.
	/// This method takes two edge vectors that form a basis, determines the
	/// coordinates of the canonic vectors in that basis, and computes the
	/// texture gradient that corresponds to those vectors.
	bool computeDeltas();
	void flipBackface();
	// compute unit inward normals for each edge.
	void computeUnitInwardNormals();
	bool drawAA(MeshMergeSamplingCallback cb, void *param);
	Vector2 v1, v2, v3;
	Vector2 n1, n2, n3; // unit inward normals
	Vector3 t1, t2, t3;
	Vector3 dx, dy;
};

class MeshMergeClippedTriangle {
public:
	MeshMergeClippedTriangle(const Vector2 &a, const Vector2 &b, const Vector2 &c);
	void clipHorizontalPlane(float offset, float clipdirection);
	void clipVerticalPlane(float offset, float clipdirection);
	void computeAreaCentroid();
	void clipAABox(float x0, float y0, float x1, float y1);
	Vector2 centroid();
	float area();

private:
	Vector2 m_verticesA[7 + 1];
	Vector2 m_verticesB[7 + 1];
	Vector2 *m_vertexBuffers[2];
	uint32_t m_numVertices;
	uint32_t m_activeVertexBuffer;
	float m_area;
	Vector2 m_centroid;
};

#endif // MESH_MERGE_TRIANGLE_H
