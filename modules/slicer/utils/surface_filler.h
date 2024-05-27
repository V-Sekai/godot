/**************************************************************************/
/*  surface_filler.h                                                      */
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

#ifndef SURFACE_FILLER_H
#define SURFACE_FILLER_H

#include "slicer_face.h"

/**
 * The inverse of FaceFiller, this struct is responsible for taking
 * SlicerFaces and serializing them back into vertex arrays for Godot
 * to read into a mesh surface
 */
struct SurfaceFiller {
	bool has_normals;
	bool has_tangents;
	bool has_colors;
	bool has_bones;
	bool has_weights;
	bool has_uvs;
	bool has_uv2s;

	Array arrays;
	Vector<Vector3> vertices;

	Vector<Vector3> normals;

	Vector<real_t> tangents;

	Vector<Color> colors;

	Vector<real_t> bones;

	Vector<real_t> weights;

	Vector<Vector2> uvs;

	Vector<Vector2> uv2s;

	Vector<SlicerFace> faces;

	SurfaceFiller(const Vector<SlicerFace> &p_faces) {
		faces = p_faces;
		SlicerFace first_face = faces[0];

		has_normals = first_face.has_normals;
		has_tangents = first_face.has_tangents;
		has_colors = first_face.has_colors;
		has_bones = first_face.has_bones;
		has_weights = first_face.has_weights;
		has_uvs = first_face.has_uvs;
		has_uv2s = first_face.has_uv2s;

		arrays.resize(Mesh::ARRAY_MAX);

		int array_length = faces.size() * 3;
		vertices.resize(array_length);

		// There's gotta be a less tedious way of doing this
		if (has_normals) {
			normals.resize(array_length);
		}

		if (has_tangents) {
			tangents.resize(array_length * 4);
		}

		if (has_colors) {
			colors.resize(array_length);
		}

		if (has_bones) {
			bones.resize(array_length * 4);
		}

		if (has_weights) {
			weights.resize(array_length * 4);
		}

		if (has_uvs) {
			uvs.resize(array_length);
		}

		if (has_uv2s) {
			uv2s.resize(array_length);
		}
	}

	/**
	 * Takes data from the faces using the lookup_idx and stores it
	 * to be saved into vertex arrays (see add_to_mesh for how to attach
	 * that information into a mesh)
	 */
	_FORCE_INLINE_ void fill(int lookup_idx, int set_idx) {
		// TODO - I think the function definition here with lookup_idx and set_idx
		// is reversed from FaceFiller#fill. We should make that more consistent
		//
		// As mentioned in the FaceFiller comments, while having this function work
		// on a vertex by vertex basis helps with cleaner code (especially, in this case,
		// when it comes to reversing the order of cross section verts), its conceptually
		// and perhaps performance drawback back by having to do these repeated calculations
		// and boolean checks (I'd hope the force_inline would help with the function invocation
		// cost but even then who knows).
		int face_idx = lookup_idx / 3;
		int idx_offset = lookup_idx % 3;

		SlicerFace face = faces[face_idx];

		vertices.write[set_idx] = face.vertex[idx_offset];

		if (has_normals) {
			normals.write[set_idx] = face.normal[idx_offset];
		}

		if (has_tangents) {
			tangents.write[set_idx * 4] = face.tangent[idx_offset][0];
			tangents.write[set_idx * 4 + 1] = face.tangent[idx_offset][1];
			tangents.write[set_idx * 4 + 2] = face.tangent[idx_offset][2];
			tangents.write[set_idx * 4 + 3] = face.tangent[idx_offset][3];
		}

		if (has_colors) {
			colors.write[set_idx] = face.color[idx_offset];
		}

		if (has_bones) {
			bones.write[set_idx * 4] = face.bones[idx_offset][0];
			bones.write[set_idx * 4 + 1] = face.bones[idx_offset][1];
			bones.write[set_idx * 4 + 2] = face.bones[idx_offset][2];
			bones.write[set_idx * 4 + 3] = face.bones[idx_offset][3];
		}

		if (has_weights) {
			weights.write[set_idx * 4] = face.weights[idx_offset][0];
			weights.write[set_idx * 4 + 1] = face.weights[idx_offset][1];
			weights.write[set_idx * 4 + 2] = face.weights[idx_offset][2];
			weights.write[set_idx * 4 + 3] = face.weights[idx_offset][3];
		}

		if (has_uvs) {
			uvs.write[set_idx] = face.uv[idx_offset];
		}

		if (has_uv2s) {
			uv2s.write[set_idx] = face.uv2[idx_offset];
		}
	}

	/**
	 * Adds the vertex information read from the "fill" as a new surface
	 * of the passed in mesh and sets the passed in material to the new
	 * surface
	 */
	void add_to_mesh(Ref<ArrayMesh> mesh, Ref<Material> material) {
		ERR_FAIL_COND(mesh.is_null());
		ERR_FAIL_COND(material.is_null());
		arrays[Mesh::ARRAY_VERTEX] = vertices;

		if (has_normals) {
			arrays[Mesh::ARRAY_NORMAL] = normals;
		}

		if (has_tangents) {
			arrays[Mesh::ARRAY_TANGENT] = tangents;
		}

		if (has_colors) {
			arrays[Mesh::ARRAY_COLOR] = colors;
		}

		if (has_bones) {
			arrays[Mesh::ARRAY_BONES] = bones;
		}

		if (has_weights) {
			arrays[Mesh::ARRAY_WEIGHTS] = weights;
		}

		if (has_uvs) {
			arrays[Mesh::ARRAY_TEX_UV] = uvs;
		}

		if (has_uv2s) {
			arrays[Mesh::ARRAY_TEX_UV2] = uv2s;
		}

		mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
		mesh->surface_set_material(mesh->get_surface_count() - 1, material);
	}

	~SurfaceFiller() {
	}
};

#endif // SURFACE_FILLER_H
