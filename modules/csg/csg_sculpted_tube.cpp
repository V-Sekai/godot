/**************************************************************************/
/*  csg_sculpted_tube.cpp                                                 */
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

#include "csg_sculpted_tube.h"

#include "core/math/geometry_3d.h"

// Helper function to generate profile points based on curve type
static void generate_profile_points(CSGSculptedPrimitive3D::ProfileCurve p_curve, real_t p_begin, real_t p_end, real_t p_hollow, CSGSculptedPrimitive3D::HollowShape p_hollow_shape, int p_segments, Vector<Vector2> &r_profile, Vector<Vector2> &r_hollow_profile) {
	r_profile.clear();
	r_hollow_profile.clear();

	real_t begin_angle = p_begin * Math::TAU;
	real_t end_angle = p_end * Math::TAU;
	real_t angle_range = end_angle - begin_angle;

	if (angle_range <= 0.0) {
		angle_range = Math::TAU;
	}

	switch (p_curve) {
		case CSGSculptedPrimitive3D::PROFILE_CURVE_CIRCLE:
		case CSGSculptedPrimitive3D::PROFILE_CURVE_CIRCLE_HALF: {
			int segments = p_segments;
			if (p_curve == CSGSculptedPrimitive3D::PROFILE_CURVE_CIRCLE_HALF) {
				segments = segments / 2;
			}

			// Generate points, ensuring the profile closes if it's a full circle
			bool is_full_circle = (p_begin == 0.0 && p_end == 1.0) || angle_range >= Math::TAU - 0.001;
			int point_count = is_full_circle ? segments : segments + 1;

			for (int i = 0; i < point_count; i++) {
				real_t angle = begin_angle + (angle_range * i / segments);
				r_profile.push_back(Vector2(Math::cos(angle), Math::sin(angle)));
			}

			// Close the loop if it's a full circle
			if (is_full_circle && r_profile.size() > 0) {
				r_profile.push_back(r_profile[0]);
			}

			if (p_hollow > 0.0) {
				real_t hollow_radius = 1.0 - p_hollow;
				for (int i = 0; i < point_count; i++) {
					real_t angle = begin_angle + (angle_range * i / segments);
					r_hollow_profile.push_back(Vector2(Math::cos(angle) * hollow_radius, Math::sin(angle) * hollow_radius));
				}
				// Close the hollow loop if it's a full circle
				if (is_full_circle && r_hollow_profile.size() > 0) {
					r_hollow_profile.push_back(r_hollow_profile[0]);
				}
			}
		} break;

		case CSGSculptedPrimitive3D::PROFILE_CURVE_SQUARE: {
			// Square profile
			real_t step = angle_range / 4.0;
			for (int i = 0; i < 4; i++) {
				real_t angle = begin_angle + step * i;
				r_profile.push_back(Vector2(Math::cos(angle), Math::sin(angle)).normalized());
			}
			r_profile.push_back(r_profile[0]); // Close the loop

			if (p_hollow > 0.0) {
				real_t hollow_radius = 1.0 - p_hollow;
				for (int i = 0; i < 4; i++) {
					real_t angle = begin_angle + step * i;
					r_hollow_profile.push_back(Vector2(Math::cos(angle), Math::sin(angle)).normalized() * hollow_radius);
				}
				r_hollow_profile.push_back(r_hollow_profile[0]);
			}
		} break;

		case CSGSculptedPrimitive3D::PROFILE_CURVE_ISOTRI:
		case CSGSculptedPrimitive3D::PROFILE_CURVE_EQUALTRI:
		case CSGSculptedPrimitive3D::PROFILE_CURVE_RIGHTTRI: {
			// Triangle profiles
			int sides = 3;
			real_t step = angle_range / sides;
			for (int i = 0; i < sides; i++) {
				real_t angle = begin_angle + step * i;
				r_profile.push_back(Vector2(Math::cos(angle), Math::sin(angle)).normalized());
			}
			r_profile.push_back(r_profile[0]);

			if (p_hollow > 0.0) {
				real_t hollow_radius = 1.0 - p_hollow;
				for (int i = 0; i < sides; i++) {
					real_t angle = begin_angle + step * i;
					r_hollow_profile.push_back(Vector2(Math::cos(angle), Math::sin(angle)).normalized() * hollow_radius);
				}
				r_hollow_profile.push_back(r_hollow_profile[0]);
			}
		} break;
	}
}

// Helper function to apply path transformations
static Vector3 apply_path_transform(const Vector2 &p_profile_point, real_t p_path_pos, const CSGSculptedPrimitive3D::PathCurve p_path_curve, real_t p_twist, const Vector2 &p_taper, const Vector2 &p_shear, real_t p_radius_offset, real_t p_revolutions, real_t p_skew) {
	Vector3 result;

	// Normalize path_pos to 0.0-1.0 range for calculations
	real_t normalized_path = CLAMP(p_path_pos, 0.0, 1.0);

	// Apply taper (taper.x is at begin, taper.y is at end)
	real_t taper_factor = 1.0 - (p_taper.x * (1.0 - normalized_path) + p_taper.y * normalized_path);
	Vector2 scaled_profile = p_profile_point * taper_factor;

	// Apply radius offset
	real_t radius = scaled_profile.length() + p_radius_offset;
	if (radius < 0.0) {
		radius = 0.0;
	}

	// Apply twist
	real_t twist_angle = p_twist * normalized_path * Math::TAU;
	Vector2 twisted_profile = Vector2(
			scaled_profile.x * Math::cos(twist_angle) - scaled_profile.y * Math::sin(twist_angle),
			scaled_profile.x * Math::sin(twist_angle) + scaled_profile.y * Math::cos(twist_angle));

	// Apply path curve
	switch (p_path_curve) {
		case CSGSculptedPrimitive3D::PATH_CURVE_LINE: {
			result = Vector3(twisted_profile.x, twisted_profile.y, normalized_path - 0.5);
		} break;

		case CSGSculptedPrimitive3D::PATH_CURVE_CIRCLE:
		case CSGSculptedPrimitive3D::PATH_CURVE_CIRCLE_33:
		case CSGSculptedPrimitive3D::PATH_CURVE_CIRCLE2: {
			real_t path_angle = normalized_path * p_revolutions * Math::TAU;
			real_t path_radius = radius;
			result = Vector3(
					path_radius * Math::cos(path_angle),
					twisted_profile.y,
					path_radius * Math::sin(path_angle));
		} break;
	}

	// Apply shear
	result.x += p_shear.x * normalized_path;
	result.y += p_shear.y * normalized_path;

	// Apply skew
	result.z += p_skew * normalized_path;

	return result;
}

void CSGSculptedTube3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_inner_radius", "inner_radius"), &CSGSculptedTube3D::set_inner_radius);
	ClassDB::bind_method(D_METHOD("get_inner_radius"), &CSGSculptedTube3D::get_inner_radius);

	ClassDB::bind_method(D_METHOD("set_outer_radius", "outer_radius"), &CSGSculptedTube3D::set_outer_radius);
	ClassDB::bind_method(D_METHOD("get_outer_radius"), &CSGSculptedTube3D::get_outer_radius);

	ClassDB::bind_method(D_METHOD("set_height", "height"), &CSGSculptedTube3D::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &CSGSculptedTube3D::get_height);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "inner_radius"), "set_inner_radius", "get_inner_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "outer_radius"), "set_outer_radius", "get_outer_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height"), "set_height", "get_height");
}

CSGSculptedTube3D::CSGSculptedTube3D() {
	profile_curve = PROFILE_CURVE_CIRCLE;
	path_curve = PATH_CURVE_LINE;
}

void CSGSculptedTube3D::set_inner_radius(const real_t p_inner_radius) {
	inner_radius = p_inner_radius;
	_make_dirty();
}

real_t CSGSculptedTube3D::get_inner_radius() const {
	return inner_radius;
}

void CSGSculptedTube3D::set_outer_radius(const real_t p_outer_radius) {
	outer_radius = p_outer_radius;
	_make_dirty();
}

real_t CSGSculptedTube3D::get_outer_radius() const {
	return outer_radius;
}

void CSGSculptedTube3D::set_height(const real_t p_height) {
	height = p_height;
	_make_dirty();
}

real_t CSGSculptedTube3D::get_height() const {
	return height;
}

CSGBrush *CSGSculptedTube3D::_build_brush() {
	// Tube is a hollow cylinder - similar to cylinder but with explicit inner/outer radius
	CSGBrush *brush = memnew(CSGBrush);

	Vector<Vector2> profile;
	Vector<Vector2> hollow_profile;
	int segments = 16;
	// For tube, we always have a hollow, so set hollow to represent the inner radius
	real_t effective_hollow = inner_radius / outer_radius;
	generate_profile_points(profile_curve, profile_begin, profile_end, effective_hollow, hollow_shape, segments, profile, hollow_profile);

	int path_segments = 16;
	real_t path_range = path_end - path_begin;
	if (path_range <= 0.0) {
		path_range = 1.0;
	}

	// Determine if profile is closed (last point equals first) - needed before generating vertices
	bool profile_closed = profile.size() > 0 && profile[profile.size() - 1].is_equal_approx(profile[0]);
	int effective_profile_count = profile_closed ? profile.size() - 1 : profile.size();
	bool hollow_closed = hollow_profile.size() > 0 && hollow_profile[hollow_profile.size() - 1].is_equal_approx(hollow_profile[0]);
	int effective_hollow_count = hollow_closed ? hollow_profile.size() - 1 : hollow_profile.size();
	int total_profile = effective_profile_count + effective_hollow_count;

	Vector<Vector3> vertices;
	Vector<Vector2> uvs;
	Vector<int> indices;

	for (int p = 0; p <= path_segments; p++) {
		real_t path_pos = path_begin + (path_range * p / path_segments);
		real_t normalized_path = (path_pos - path_begin) / path_range;
		real_t twist = Math::lerp(twist_begin, twist_end, normalized_path);

		// Only generate vertices for unique profile points (skip duplicate if closed)
		for (int i = 0; i < effective_profile_count; i++) {
			Vector2 prof = profile[i] * scale * outer_radius;
			Vector3 vertex = apply_path_transform(prof, path_pos, path_curve, twist, taper, shear, radius_offset, revolutions, skew);
			vertex.y *= height;
			vertices.push_back(vertex);
			uvs.push_back(Vector2((real_t)i / effective_profile_count, path_pos));
		}

		if (effective_hollow_count > 0) {
			// Only generate vertices for unique hollow profile points
			for (int i = 0; i < effective_hollow_count; i++) {
				Vector2 prof = hollow_profile[i] * scale * outer_radius;
				Vector3 vertex = apply_path_transform(prof, path_pos, path_curve, twist, taper, shear, radius_offset, revolutions, skew);
				vertex.y *= height;
				vertices.push_back(vertex);
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

		if (effective_hollow_count > 0) {
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

	// Add end caps for linear paths to ensure manifold geometry
	if (path_curve == PATH_CURVE_LINE && path_begin == 0.0 && path_end == 1.0) {
		// Bottom cap (at path_begin) - outer profile
		int bottom_base = 0;
		if (effective_profile_count >= 3) {
			// Triangulate the bottom face
			for (int i = 1; i < effective_profile_count - 1; i++) {
				indices.push_back(bottom_base);
				indices.push_back(bottom_base + i + 1);
				indices.push_back(bottom_base + i);
			}
		}

		// Top cap (at path_end) - outer profile
		int top_base = path_segments * total_profile;
		if (effective_profile_count >= 3) {
			// Triangulate the top face (reverse winding for opposite side)
			for (int i = 1; i < effective_profile_count - 1; i++) {
				indices.push_back(top_base);
				indices.push_back(top_base + i);
				indices.push_back(top_base + i + 1);
			}
		}

		// Bottom cap - inner profile (hollow)
		if (effective_hollow_count > 0) {
			int hollow_bottom_base = effective_profile_count;
			if (effective_hollow_count >= 3) {
				for (int i = 1; i < effective_hollow_count - 1; i++) {
					indices.push_back(hollow_bottom_base);
					indices.push_back(hollow_bottom_base + i);
					indices.push_back(hollow_bottom_base + i + 1);
				}
			}
		}

		// Top cap - inner profile (hollow)
		if (effective_hollow_count > 0) {
			int hollow_top_base = path_segments * total_profile + effective_profile_count;
			if (effective_hollow_count >= 3) {
				for (int i = 1; i < effective_hollow_count - 1; i++) {
					indices.push_back(hollow_top_base);
					indices.push_back(hollow_top_base + i + 1);
					indices.push_back(hollow_top_base + i);
				}
			}
		}
	}

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

	brush->build_from_faces(faces, face_uvs, smooth, materials, invert);

	// Debug output for testing
	print_verbose("CSGSculptedTube3D::_build_brush() debug:");
	print_verbose(vformat("  Profile size: %d, effective_profile_count: %d", profile.size(), effective_profile_count));
	print_verbose(vformat("  Hollow profile size: %d, effective_hollow_count: %d", hollow_profile.size(), effective_hollow_count));
	print_verbose(vformat("  Total profile: %d, path_segments: %d", total_profile, path_segments));
	print_verbose(vformat("  Vertices generated: %d", vertices.size()));
	print_verbose(vformat("  Indices generated: %d (face_count: %d)", indices.size(), face_count));
	print_verbose(vformat("  Faces array size: %d", faces.size()));
	print_verbose(vformat("  Brush faces after build_from_faces: %d", brush->faces.size()));
	if (brush->faces.size() > 0) {
		print_verbose(vformat("  First face vertices: (%f, %f, %f), (%f, %f, %f), (%f, %f, %f)",
				brush->faces[0].vertices[0].x, brush->faces[0].vertices[0].y, brush->faces[0].vertices[0].z,
				brush->faces[0].vertices[1].x, brush->faces[0].vertices[1].y, brush->faces[0].vertices[1].z,
				brush->faces[0].vertices[2].x, brush->faces[0].vertices[2].y, brush->faces[0].vertices[2].z));
	}

	return brush;
}
