#pragma once

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
