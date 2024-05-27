/**************************************************************************/
/*  face_filler.h                                                         */
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

#ifndef FACE_FILLER_H
#define FACE_FILLER_H

#include "slicer_face.h"

// This just mimics logic found in TriangleMesh#Create
_FORCE_INLINE_ Vector3 snap_vertex(Vector3 v) {
	return v.snapped(Vector3(0.0001, 0.0001, 0.0001));
}

/**
 * Responsible for serializing data from vertex arrays, as they are
 * given from the visual server, into an array of SlicerFaces while
 * maintaining info about things such as normals and uvs etc.
 */
struct FaceFiller {
	bool has_normals;
	bool has_tangents;
	bool has_colors;
	bool has_bones;
	bool has_weights;
	bool has_uvs;
	bool has_uv2s;

	Vector<Vector3> vertices;

	Vector<Vector3> normals;

	Vector<real_t> tangents;

	Vector<Color> colors;

	Vector<real_t> bones;

	Vector<real_t> weights;

	Vector<Vector2> uvs;

	Vector<Vector2> uv2s;
	Vector<SlicerFace> faces;

	// Yuck. What an eye sore this constructor is
	FaceFiller(Vector<SlicerFace> &p_faces, const Array &surface_arrays) {
		faces = p_faces;
		Vector<Vector3> vertices = surface_arrays[Mesh::ARRAY_VERTEX];

		Vector<Vector3> normals = surface_arrays[Mesh::ARRAY_NORMAL];
		has_normals = normals.size() > 0 && normals.size() == vertices.size();

		Vector<real_t> tangents = surface_arrays[Mesh::ARRAY_TANGENT];
		has_tangents = tangents.size() > 0 && tangents.size() == vertices.size() * 4;

		Vector<Color> colors = surface_arrays[Mesh::ARRAY_COLOR];
		has_colors = colors.size() > 0 && colors.size() == vertices.size();

		Vector<real_t> bones = surface_arrays[Mesh::ARRAY_BONES];
		has_bones = bones.size() > 0 && bones.size() == vertices.size() * 4;

		Vector<real_t> weights = surface_arrays[Mesh::ARRAY_WEIGHTS];
		has_weights = weights.size() > 0 && weights.size() == vertices.size() * 4;

		Vector<Vector2> uvs = surface_arrays[Mesh::ARRAY_TEX_UV];
		has_uvs = uvs.size() > 0 && uvs.size() == vertices.size();

		Vector<Vector2> uv2s = surface_arrays[Mesh::ARRAY_TEX_UV2];
		has_uv2s = uv2s.size() > 0 && uv2s.size() == vertices.size();
	}

	/**
	 * Takes data from the vertex array using the lookup_idx and puts it into
	 * our face vector using set_idx
	 */
	_FORCE_INLINE_ void fill(int set_idx, int lookup_idx) {
		// Having this function work vertex by vertex makes the code a bit nicer,
		// especially with having to support indexed and non-indexed vertices,
		// but just performance-wise I hate it. There's no reason, besides uglier
		// and more complicated code, why we can't be doing these has_* checks on
		// a per face basis. Maybe even a per mesh basis. Admitidly it'll probably
		// all come out in the wash, but it bothers me conceptually. Let's put in
		// a TODO about it. Maybe there's something incredibly clever we can do with
		// macros that *won't* make me want to tear out what's left of my hair.
		int face_idx = set_idx / 3;
		int set_offset = set_idx % 3;

		if (set_offset == 0) {
			faces.write[face_idx].has_normals = has_normals;
			faces.write[face_idx].has_tangents = has_tangents;
			faces.write[face_idx].has_colors = has_colors;
			faces.write[face_idx].has_bones = has_bones;
			faces.write[face_idx].has_weights = has_weights;
			faces.write[face_idx].has_uvs = has_uvs;
			faces.write[face_idx].has_uv2s = has_uv2s;
		}

		faces.write[face_idx].vertex[set_offset] = snap_vertex(vertices[lookup_idx]);

		if (has_normals) {
			faces.write[face_idx].normal[set_offset] = normals[lookup_idx];
		}

		if (has_tangents) {
			faces.write[face_idx].tangent[set_offset] = SlicerVector4(
					tangents[lookup_idx * 4],
					tangents[lookup_idx * 4 + 1],
					tangents[lookup_idx * 4 + 2],
					tangents[lookup_idx * 4 + 3]);
		}

		if (has_colors) {
			faces.write[face_idx].color[set_offset] = colors[lookup_idx];
		}

		if (has_bones) {
			faces.write[face_idx].bones[set_offset] = SlicerVector4(
					bones[lookup_idx * 4],
					bones[lookup_idx * 4 + 1],
					bones[lookup_idx * 4 + 2],
					bones[lookup_idx * 4 + 3]);
		}

		if (has_weights) {
			faces.write[face_idx].weights[set_offset] = SlicerVector4(
					weights[lookup_idx],
					weights[lookup_idx * 4 + 1],
					weights[lookup_idx * 4 + 2],
					weights[lookup_idx * 4 + 3]);
		}

		if (has_uvs) {
			faces.write[face_idx].uv[set_offset] = uvs[lookup_idx];
		}

		if (has_uv2s) {
			faces.write[face_idx].uv2[set_offset] = uv2s[lookup_idx];
		}
	}

	~FaceFiller() {
	}
};

#endif // FACE_FILLER_H
