/**************************************************************************/
/*  mesh_merge_triangle.cpp                                               */
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

#include "mesh_merge_triangle.h"

MeshMergeClippedTriangle::MeshMergeClippedTriangle(const Vector2 &a, const Vector2 &b, const Vector2 &c) {
	m_area = 0;
	m_numVertices = 3;
	m_activeVertexBuffer = 0;
	m_verticesA[0] = a;
	m_verticesA[1] = b;
	m_verticesA[2] = c;
	m_vertexBuffers[0] = m_verticesA;
	m_vertexBuffers[1] = m_verticesB;
}

void MeshMergeClippedTriangle::clipHorizontalPlane(float offset, float clipdirection) {
	Vector2 *v = m_vertexBuffers[m_activeVertexBuffer];
	m_activeVertexBuffer ^= 1;
	Vector2 *v2 = m_vertexBuffers[m_activeVertexBuffer];
	v[m_numVertices] = v[0];
	float dy2, dy1 = offset - v[0].y;
	int dy2in, dy1in = clipdirection * dy1 >= 0;
	uint32_t p = 0;
	for (uint32_t k = 0; k < m_numVertices; k++) {
		dy2 = offset - v[k + 1].y;
		dy2in = clipdirection * dy2 >= 0;
		if (dy1in) {
			v2[p++] = v[k];
		}
		if (dy1in + dy2in == 1) { // not both in/out
			float dx = v[k + 1].x - v[k].x;
			float dy = v[k + 1].y - v[k].y;
			v2[p++] = Vector2(v[k].x + dy1 * (dx / dy), offset);
		}
		dy1 = dy2;
		dy1in = dy2in;
	}
	m_numVertices = p;
}

void MeshMergeClippedTriangle::clipVerticalPlane(float offset, float clipdirection) {
	Vector2 *v = m_vertexBuffers[m_activeVertexBuffer];
	m_activeVertexBuffer ^= 1;
	Vector2 *v2 = m_vertexBuffers[m_activeVertexBuffer];
	v[m_numVertices] = v[0];
	float dx2, dx1 = offset - v[0].x;
	int dx2in, dx1in = clipdirection * dx1 >= 0;
	uint32_t p = 0;
	for (uint32_t k = 0; k < m_numVertices; k++) {
		dx2 = offset - v[k + 1].x;
		dx2in = clipdirection * dx2 >= 0;
		if (dx1in) {
			v2[p++] = v[k];
		}
		if (dx1in + dx2in == 1) { // not both in/out
			float dx = v[k + 1].x - v[k].x;
			float dy = v[k + 1].y - v[k].y;
			v2[p++] = Vector2(offset, v[k].y + dx1 * (dy / dx));
		}
		dx1 = dx2;
		dx1in = dx2in;
	}
	m_numVertices = p;
}

void MeshMergeClippedTriangle::computeAreaCentroid() {
	Vector2 *v = m_vertexBuffers[m_activeVertexBuffer];
	v[m_numVertices] = v[0];
	m_area = 0;
	float centroidx = 0, centroidy = 0;
	for (uint32_t k = 0; k < m_numVertices; k++) {
		// http://local.wasp.uwa.edu.au/~pbourke/geometry/polyarea/
		float f = v[k].x * v[k + 1].y - v[k + 1].x * v[k].y;
		m_area += f;
		centroidx += f * (v[k].x + v[k + 1].x);
		centroidy += f * (v[k].y + v[k + 1].y);
	}
	m_area = 0.5f * fabsf(m_area);
	if (m_area == 0) {
		m_centroid = Vector2(0.0f, 0.0f);
	} else {
		m_centroid = Vector2(centroidx / (6 * m_area), centroidy / (6 * m_area));
	}
}

void MeshMergeClippedTriangle::clipAABox(float x0, float y0, float x1, float y1) {
	clipVerticalPlane(x0, -1);
	clipHorizontalPlane(y0, -1);
	clipVerticalPlane(x1, 1);
	clipHorizontalPlane(y1, 1);
	computeAreaCentroid();
}

Vector2 MeshMergeClippedTriangle::centroid() {
	return m_centroid;
}

float MeshMergeClippedTriangle::area() {
	return m_area;
}

MeshMergeTriangle::MeshMergeTriangle(const Vector2 &p_v0, const Vector2 &p_v1, const Vector2 &p_v2, const Vector3 &p_t0, const Vector3 &p_t1, const Vector3 &p_t2) {
	// Init vertices.
	this->v1 = p_v0;
	this->v2 = p_v2;
	this->v3 = p_v1;
	// Set barycentric coordinates.
	this->t1 = p_t0;
	this->t2 = p_t2;
	this->t3 = p_t1;
	// make sure every triangle is front facing.
	flipBackface();
	// Compute deltas.
	computeDeltas();
	computeUnitInwardNormals();
}

bool MeshMergeTriangle::computeDeltas() {
	Vector2 e0 = v3 - v1;
	Vector2 e1 = v2 - v1;
	Vector3 de0 = t3 - t1;
	Vector3 de1 = t2 - t1;

	float denom = e0.y * e1.x - e1.y * e0.x;
	if (denom == 0.0f || !Math::is_finite(denom)) {
		return false;
	}

	float denom_inv = 1.0f / denom;
	float lambda1 = -e1.y * denom_inv;
	float lambda2 = e0.y * denom_inv;
	float lambda3 = e1.x * denom_inv;
	float lambda4 = -e0.x * denom_inv;

	dx = de0 * lambda1 + de1 * lambda2;
	dy = de0 * lambda3 + de1 * lambda4;

	return true;
}

void MeshMergeTriangle::flipBackface() {
	// check if triangle is backfacing, if so, swap two vertices
	if (((v3.x - v1.x) * (v2.y - v1.y) - (v3.y - v1.y) * (v2.x - v1.x)) < 0) {
		Vector2 hv = v1;
		v1 = v2;
		v2 = hv; // swap pos
		Vector3 ht = t1;
		t1 = t2;
		t2 = ht; // swap tex
	}
}

void MeshMergeTriangle::computeUnitInwardNormals() {
	n1 = v1 - v2;
	n1 = Vector2(-n1.y, n1.x);
	n1 = n1 * (1.0f / sqrtf(n1.x * n1.x + n1.y * n1.y));
	n2 = v2 - v3;
	n2 = Vector2(-n2.y, n2.x);
	n2 = n2 * (1.0f / sqrtf(n2.x * n2.x + n2.y * n2.y));
	n3 = v3 - v1;
	n3 = Vector2(-n3.y, n3.x);
	n3 = n3 * (1.0f / sqrtf(n3.x * n3.x + n3.y * n3.y));
}

bool MeshMergeTriangle::drawAA(MeshMergeSamplingCallback cb, void *param) {
	const float PX_INSIDE = 1.0f / sqrtf(2.0f);
	const float PX_OUTSIDE = -1.0f / sqrtf(2.0f);
	const float BK_SIZE = 8;
	const float BK_INSIDE = sqrtf(BK_SIZE * BK_SIZE / 2.0f);
	const float BK_OUTSIDE = -sqrtf(BK_SIZE * BK_SIZE / 2.0f);

	// Bounding rectangle
	float minx = floorf(MAX(MIN(v1.x, MIN(v2.x, v3.x)), 0.0f));
	float miny = floorf(MAX(MIN(v1.y, MIN(v2.y, v3.y)), 0.0f));
	float maxx = ceilf(MAX(v1.x, MAX(v2.x, v3.x)));
	float maxy = ceilf(MAX(v1.y, MAX(v2.y, v3.y)));

	// Align to texel centers
	minx += 0.5f;
	miny += 0.5f;
	maxx += 0.5f;
	maxy += 0.5f;

	// Half-edge constants
	float C1 = n1.x * (-v1.x) + n1.y * (-v1.y);
	float C2 = n2.x * (-v2.x) + n2.y * (-v2.y);
	float C3 = n3.x * (-v3.x) + n3.y * (-v3.y);

	// Loop through blocks
	for (float y0 = miny; y0 <= maxy; y0 += BK_SIZE) {
		for (float x0 = minx; x0 <= maxx; x0 += BK_SIZE) {
			// Corners of block
			float xc = (x0 + (BK_SIZE - 1) / 2.0f);
			float yc = (y0 + (BK_SIZE - 1) / 2.0f);

			// Evaluate half-space functions
			float aC = C1 + n1.x * xc + n1.y * yc;
			float bC = C2 + n2.x * xc + n2.y * yc;
			float cC = C3 + n3.x * xc + n3.y * yc;

			// Skip block when outside an edge
			if ((aC <= BK_OUTSIDE) || (bC <= BK_OUTSIDE) || (cC <= BK_OUTSIDE)) {
				continue;
			}

			// Calculate initial texture coordinates
			Vector3 texRow = t1 + dy * (y0 - v1.y) + dx * (x0 - v1.x);

			// Accept whole block when totally covered
			if ((aC >= BK_INSIDE) && (bC >= BK_INSIDE) && (cC >= BK_INSIDE)) {
				for (float y = y0; y < y0 + BK_SIZE; y++) {
					Vector3 tex = texRow;
					for (float x = x0; x < x0 + BK_SIZE; x++) {
						if (!cb(param, (int)x, (int)y, tex, dx, dy, 1.0f)) {
							return false;
						}
						tex += dx;
					}
					texRow += dy;
				}
			} else { // Partially covered block
				float CY1 = C1 + n1.x * x0 + n1.y * y0;
				float CY2 = C2 + n2.x * x0 + n2.y * y0;
				float CY3 = C3 + n3.x * x0 + n3.y * y0;

				for (float y = y0; y < y0 + BK_SIZE; y++) {
					float CX1 = CY1;
					float CX2 = CY2;
					float CX3 = CY3;
					Vector3 tex = texRow;

					for (float x = x0; x < x0 + BK_SIZE; x++) {
						Vector3 tex2 = t1 + dx * (x - v1.x) + dy * (y - v1.y);
						if (CX1 >= PX_INSIDE && CX2 >= PX_INSIDE && CX3 >= PX_INSIDE) {
							// pixel completely covered
							if (!cb(param, (int)x, (int)y, tex2, dx, dy, 1.0f)) {
								return false;
							}
						} else if ((CX1 >= PX_OUTSIDE) && (CX2 >= PX_OUTSIDE) && (CX3 >= PX_OUTSIDE)) {
							// triangle partially covers pixel. do clipping.
							MeshMergeClippedTriangle ct(v1 - Vector2(x, y), v2 - Vector2(x, y), v3 - Vector2(x, y));
							ct.clipAABox(-0.5, -0.5, 0.5, 0.5);
							float area = ct.area();
							if (area > 0.0f) {
								if (!cb(param, (int)x, (int)y, tex2, dx, dy, 0.0f)) {
									return false;
								}
							}
						}
						CX1 += n1.x;
						CX2 += n2.x;
						CX3 += n3.x;
						tex += dx;
					}
					CY1 += n1.y;
					CY2 += n2.y;
					CY3 += n3.y;
					texRow += dy;
				}
			}
		}
	}
	return true;
}
