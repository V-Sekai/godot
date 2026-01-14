/**************************************************************************/
/*  csg_sculpted_ring.cpp                                                 */
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

#include "csg_sculpted_ring.h"

#include "core/math/geometry_3d.h"
#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/mesh.h"

// Helper function to generate profile points based on curve type

// Helper function to apply path transformations

void CSGSculptedRing3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_inner_radius", "inner_radius"), &CSGSculptedRing3D::set_inner_radius);
	ClassDB::bind_method(D_METHOD("get_inner_radius"), &CSGSculptedRing3D::get_inner_radius);

	ClassDB::bind_method(D_METHOD("set_outer_radius", "outer_radius"), &CSGSculptedRing3D::set_outer_radius);
	ClassDB::bind_method(D_METHOD("get_outer_radius"), &CSGSculptedRing3D::get_outer_radius);

	ClassDB::bind_method(D_METHOD("set_height", "height"), &CSGSculptedRing3D::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &CSGSculptedRing3D::get_height);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "inner_radius"), "set_inner_radius", "get_inner_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "outer_radius"), "set_outer_radius", "get_outer_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height"), "set_height", "get_height");
}

CSGSculptedRing3D::CSGSculptedRing3D() {
	profile_curve = PROFILE_CURVE_CIRCLE;
	path_curve = PATH_CURVE_CIRCLE;
}

void CSGSculptedRing3D::set_inner_radius(const real_t p_inner_radius) {
	inner_radius = p_inner_radius;
	_make_dirty();
}

real_t CSGSculptedRing3D::get_inner_radius() const {
	return inner_radius;
}

void CSGSculptedRing3D::set_outer_radius(const real_t p_outer_radius) {
	outer_radius = p_outer_radius;
	_make_dirty();
}

real_t CSGSculptedRing3D::get_outer_radius() const {
	return outer_radius;
}

void CSGSculptedRing3D::set_height(const real_t p_height) {
	height = p_height;
	_make_dirty();
}

real_t CSGSculptedRing3D::get_height() const {
	return height;
}

CSGBrush *CSGSculptedRing3D::_build_brush() {
	// Ring is similar to torus but with different proportions (thinner, more ring-like)
	CSGBrush *_brush = memnew(CSGBrush);

	Vector<Vector2> profile;
	Vector<Vector2> hollow_profile;
	int segments = 16;
	generate_profile_points(profile_curve, profile_begin, profile_end, hollow, hollow_shape, segments, profile, hollow_profile);

	int path_segments = 16;
	real_t path_range = path_end - path_begin;
	if (path_range <= 0.0) {
		path_range = 1.0;
	}

	real_t major_radius = (inner_radius + outer_radius) / 2.0;
	real_t minor_radius = height / 2.0; // Ring uses height as the minor radius

	// Determine if profile is closed (last point equals first) - needed before generating vertices
	bool profile_closed = profile.size() > 0 && profile[profile.size() - 1].is_equal_approx(profile[0]);
	int effective_profile_count = profile_closed ? profile.size() - 1 : profile.size();
	bool hollow_closed = hollow_profile.size() > 0 && hollow_profile[hollow_profile.size() - 1].is_equal_approx(hollow_profile[0]);
	int effective_hollow_count = (hollow > 0.0 && hollow_profile.size() > 0) ? (hollow_closed ? hollow_profile.size() - 1 : hollow_profile.size()) : 0;
	int total_profile = effective_profile_count + effective_hollow_count;

	Vector<Vector3> vertices;
	Vector<Vector2> uvs;
	Vector<int> indices;

	for (int p = 0; p <= path_segments; p++) {
		real_t path_pos = path_begin + (path_range * p / path_segments);
		real_t normalized_path = (path_pos - path_begin) / path_range;
		real_t twist = Math::lerp(twist_begin, twist_end, normalized_path);
		real_t path_angle = path_pos * revolutions * Math::TAU;

		// Only generate vertices for unique profile points (skip duplicate if closed)
		for (int i = 0; i < effective_profile_count; i++) {
			Vector2 prof = profile[i] * scale * minor_radius;
			// Apply path transformations (twist, taper, shear, radius_offset, skew)
			Vector3 transformed = apply_path_transform(prof, path_pos, path_curve, twist, taper, shear, radius_offset, revolutions, skew);
			// Transform to ring coordinates using the transformed profile
			real_t radius_at_angle = major_radius + transformed.x;
			Vector3 ring_pos = Vector3(
					radius_at_angle * Math::cos(path_angle),
					transformed.y,
					radius_at_angle * Math::sin(path_angle));
			// Apply skew offset in Z
			ring_pos.z += transformed.z - (normalized_path - 0.5);
			vertices.push_back(ring_pos);
			uvs.push_back(Vector2((real_t)i / effective_profile_count, path_pos));
		}

		if (hollow > 0.0 && effective_hollow_count > 0) {
			// Only generate vertices for unique hollow profile points
			for (int i = 0; i < effective_hollow_count; i++) {
				Vector2 prof = hollow_profile[i] * scale * minor_radius;
				// Apply path transformations
				Vector3 transformed = apply_path_transform(prof, path_pos, path_curve, twist, taper, shear, radius_offset, revolutions, skew);
				real_t radius_at_angle = major_radius + transformed.x;
				Vector3 ring_pos = Vector3(
						radius_at_angle * Math::cos(path_angle),
						transformed.y,
						radius_at_angle * Math::sin(path_angle));
				ring_pos.z += transformed.z - (normalized_path - 0.5);
				vertices.push_back(ring_pos);
				uvs.push_back(Vector2((real_t)i / effective_hollow_count, path_pos));
			}
		}
	}

	// Generate faces
	// Note: effective_profile_count, effective_hollow_count, and total_profile are already calculated above

	// For circular paths, close the path loop as well
	bool path_closed = (path_curve == PATH_CURVE_CIRCLE || path_curve == PATH_CURVE_CIRCLE_33 || path_curve == PATH_CURVE_CIRCLE2) && path_begin == 0.0 && path_end == 1.0;

	for (int p = 0; p < path_segments; p++) {
		int base1 = p * total_profile;
		int base2 = path_closed ? ((p + 1) % (path_segments + 1)) * total_profile : (p + 1) * total_profile; // Wrap around for closed paths

		// Outer faces - close the loop
		// Match CSGCylinder3D pattern exactly: (bottom_i, bottom_next_i, top_next_i) and (top_next_i, top_i, bottom_i)
		// The manifold library reorders with {0,2,1} when packing and unpacking, so the net effect cancels
		// Provide vertices in counter-clockwise order as they should appear in the final mesh
		for (int i = 0; i < effective_profile_count; i++) {
			int next_i = (i + 1) % effective_profile_count;
			// Triangle 1: (base1+i, base1+next_i, base2+next_i) - counter-clockwise from outside
			indices.push_back(base1 + i);
			indices.push_back(base1 + next_i);
			indices.push_back(base2 + next_i);

			// Triangle 2: (base2+next_i, base2+i, base1+i) - counter-clockwise from outside
			// Shares edge (base1+i, base2+next_i) in opposite order
			indices.push_back(base2 + next_i);
			indices.push_back(base2 + i);
			indices.push_back(base1 + i);
		}

		if (hollow > 0.0 && effective_hollow_count > 0) {
			int hollow_base1 = base1 + effective_profile_count;
			int hollow_base2 = base2 + effective_profile_count;

			for (int i = 0; i < effective_hollow_count; i++) {
				int next_i = (i + 1) % effective_hollow_count;
				// Hollow faces: counter-clockwise from outside (same pattern as outer faces)
				// Triangle 1: (hollow_bottom_i, hollow_bottom_next_i, hollow_top_next_i)
				indices.push_back(hollow_base1 + i);
				indices.push_back(hollow_base1 + next_i);
				indices.push_back(hollow_base2 + next_i);

				// Triangle 2: (hollow_top_next_i, hollow_top_i, hollow_bottom_i)
				// Shares edge (hollow_bottom_i, hollow_top_next_i) in opposite order
				indices.push_back(hollow_base2 + next_i);
				indices.push_back(hollow_base2 + i);
				indices.push_back(hollow_base1 + i);
			}
		}
	}

	// Add end caps for open paths (linear or partial revolution) to ensure manifold geometry
	bool needs_caps = !path_closed;
	cap_open_ends(indices, total_profile, effective_profile_count, effective_hollow_count, path_segments, hollow, needs_caps);

	// Convert to CSGBrush format
	Vector<Vector3> faces;
	Vector<Vector2> face_uvs;
	Vector<bool> smooth;
	Vector<Ref<Material>> materials;
	Vector<bool> invert;

	int face_count = indices.size() / 3;
	faces.resize(face_count * 3);
	face_uvs.resize(face_count * 3);
	smooth.resize(face_count);
	materials.resize(face_count);
	invert.resize(face_count);

	{
		Vector3 *facesw = faces.ptrw();
		Vector2 *face_uvsw = face_uvs.ptrw();
		bool *smoothw = smooth.ptrw();
		Ref<Material> *materialsw = materials.ptrw();
		bool *invertw = invert.ptrw();

		bool flip = get_flip_faces();
		for (int i = 0; i < face_count; i++) {
			int idx = i * 3;
			if (flip) {
				facesw[idx] = vertices[indices[idx + 2]];
				facesw[idx + 1] = vertices[indices[idx + 1]];
				facesw[idx + 2] = vertices[indices[idx]];
				face_uvsw[idx] = uvs[indices[idx + 2]];
				face_uvsw[idx + 1] = uvs[indices[idx + 1]];
				face_uvsw[idx + 2] = uvs[indices[idx]];
			} else {
				facesw[idx] = vertices[indices[idx]];
				facesw[idx + 1] = vertices[indices[idx + 1]];
				facesw[idx + 2] = vertices[indices[idx + 2]];
				face_uvsw[idx] = uvs[indices[idx]];
				face_uvsw[idx + 1] = uvs[indices[idx + 1]];
				face_uvsw[idx + 2] = uvs[indices[idx + 2]];
			}
			smoothw[i] = true;
			materialsw[i] = material;
			invertw[i] = flip;
		}
	}

	_brush->build_from_faces(faces, face_uvs, smooth, materials, invert);

	// Debug output for testing
	print_verbose("CSGSculptedRing3D::_build_brush() debug:");
	print_verbose(vformat("  Profile size: %d, effective_profile_count: %d", profile.size(), effective_profile_count));
	print_verbose(vformat("  Hollow profile size: %d, effective_hollow_count: %d", hollow_profile.size(), effective_hollow_count));
	print_verbose(vformat("  Total profile: %d, path_segments: %d", total_profile, path_segments));
	print_verbose(vformat("  Vertices generated: %d", vertices.size()));
	print_verbose(vformat("  Indices generated: %d (face_count: %d)", indices.size(), face_count));
	print_verbose(vformat("  Faces array size: %d", faces.size()));
	print_verbose(vformat("  Brush faces after build_from_faces: %d", _brush->faces.size()));
	if (_brush->faces.size() > 0) {
		print_verbose(vformat("  First face vertices: (%f, %f, %f), (%f, %f, %f), (%f, %f, %f)",
				_brush->faces[0].vertices[0].x, _brush->faces[0].vertices[0].y, _brush->faces[0].vertices[0].z,
				_brush->faces[0].vertices[1].x, _brush->faces[0].vertices[1].y, _brush->faces[0].vertices[1].z,
				_brush->faces[0].vertices[2].x, _brush->faces[0].vertices[2].y, _brush->faces[0].vertices[2].z));
	}

	// Validate manifold geometry requirements
	Ref<ArrayMesh> test_mesh;
	test_mesh.instantiate();
	Array arrays;
	arrays.resize(RS::ARRAY_MAX);
	arrays[RS::ARRAY_VERTEX] = vertices;
	arrays[RS::ARRAY_INDEX] = indices;
	test_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
	Dictionary validation_result = CSGShape3D::validate_manifold_mesh(test_mesh);
	if (!(bool)validation_result["valid"]) {
		print_verbose(vformat("CSGSculptedRing3D::_build_brush() - MANIFOLD VALIDATION FAILED"));
		Array errors = validation_result["errors"];
		for (int i = 0; i < errors.size(); i++) {
			print_verbose(vformat("  ERROR: %s", (String)errors[i]));
		}
		print_verbose("CSGSculptedRing3D::_build_brush() - This may cause CSG operations to fail!");
		// Don't return nullptr - let the manifold library catch issues later
		// But warn the user that there are problems
	} else {
		print_verbose("CSGSculptedRing3D::_build_brush() - Manifold validation passed");
	}

	return _brush;
}
