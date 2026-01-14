/**************************************************************************/
/*  csg_sculpted_texture.cpp                                              */
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

#include "csg_sculpted_texture.h"

#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/mesh.h"

void CSGSculptedTexture3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_sculpt_texture", "texture"), &CSGSculptedTexture3D::set_sculpt_texture);
	ClassDB::bind_method(D_METHOD("get_sculpt_texture"), &CSGSculptedTexture3D::get_sculpt_texture);

	ClassDB::bind_method(D_METHOD("set_mirror", "mirror"), &CSGSculptedTexture3D::set_mirror);
	ClassDB::bind_method(D_METHOD("get_mirror"), &CSGSculptedTexture3D::get_mirror);

	ClassDB::bind_method(D_METHOD("set_invert", "invert"), &CSGSculptedTexture3D::set_invert);
	ClassDB::bind_method(D_METHOD("get_invert"), &CSGSculptedTexture3D::get_invert);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "sculpt_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_sculpt_texture", "get_sculpt_texture");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "mirror"), "set_mirror", "get_mirror");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "invert"), "set_invert", "get_invert");
}

void CSGSculptedTexture3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (sculpt_texture.is_valid()) {
				sculpt_texture->connect("changed", callable_mp(this, &CSGSculptedTexture3D::_texture_changed));
			}
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (sculpt_texture.is_valid()) {
				sculpt_texture->disconnect("changed", callable_mp(this, &CSGSculptedTexture3D::_texture_changed));
			}
		} break;
	}
}

CSGSculptedTexture3D::CSGSculptedTexture3D() {
	profile_curve = PROFILE_CURVE_CIRCLE;
	path_curve = PATH_CURVE_LINE;
}

void CSGSculptedTexture3D::_texture_changed() {
	_make_dirty();
}

void CSGSculptedTexture3D::set_sculpt_texture(const Ref<Texture2D> &p_texture) {
	if (sculpt_texture.is_valid()) {
		if (is_inside_tree()) {
			sculpt_texture->disconnect("changed", callable_mp(this, &CSGSculptedTexture3D::_texture_changed));
		}
	}
	sculpt_texture = p_texture;
	if (sculpt_texture.is_valid()) {
		if (is_inside_tree()) {
			sculpt_texture->connect("changed", callable_mp(this, &CSGSculptedTexture3D::_texture_changed));
		}
	}
	_make_dirty();
}

Ref<Texture2D> CSGSculptedTexture3D::get_sculpt_texture() const {
	return sculpt_texture;
}

void CSGSculptedTexture3D::set_mirror(bool p_mirror) {
	mirror = p_mirror;
	_make_dirty();
}

bool CSGSculptedTexture3D::get_mirror() const {
	return mirror;
}

void CSGSculptedTexture3D::set_invert(bool p_invert) {
	invert = p_invert;
	_make_dirty();
}

bool CSGSculptedTexture3D::get_invert() const {
	return invert;
}

CSGBrush *CSGSculptedTexture3D::_build_brush() {
	CSGBrush *_brush = memnew(CSGBrush);

	if (!sculpt_texture.is_valid()) {
		// Return empty brush if no texture
		return _brush;
	}

	Ref<Image> image = sculpt_texture->get_image();
	if (!image.is_valid() || image->is_empty()) {
		return _brush;
	}

	int width = image->get_width();
	int height = image->get_height();

	if (width < 2 || height < 2) {
		return _brush;
	}

	// Convert texture RGB values to 3D coordinates
	// Texture format: R=X, G=Y, B=Z, each mapped from 0-255 to -1 to 1
	Vector<Vector3> vertices;
	Vector<Vector2> uvs;
	Vector<int> indices;

	// Generate vertices from texture
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			Color pixel = image->get_pixel(x, y);

			// Convert RGB (0-1) to XYZ (-1 to 1)
			// Use base scale parameter for scaling
			real_t x_coord = (pixel.r * 2.0 - 1.0) * scale.x;
			real_t y_coord = (pixel.g * 2.0 - 1.0) * scale.y;
			real_t z_coord = (pixel.b * 2.0 - 1.0);

			// Apply invert flag
			if (invert) {
				z_coord = -z_coord;
			}

			// Apply mirror flag
			if (mirror) {
				x_coord = -x_coord;
			}

			vertices.push_back(Vector3(x_coord, y_coord, z_coord));
			uvs.push_back(Vector2((real_t)x / (width - 1), (real_t)y / (height - 1)));
		}
	}

	// Generate triangle indices (grid triangulation)
	for (int y = 0; y < height - 1; y++) {
		for (int x = 0; x < width - 1; x++) {
			int i0 = y * width + x;
			int i1 = y * width + (x + 1);
			int i2 = (y + 1) * width + x;
			int i3 = (y + 1) * width + (x + 1);

			// First triangle
			indices.push_back(i0);
			indices.push_back(i2);
			indices.push_back(i1);

			// Second triangle
			indices.push_back(i1);
			indices.push_back(i2);
			indices.push_back(i3);
		}
	}

	// Convert to CSGBrush format
	Vector<Vector3> faces;
	Vector<Vector2> face_uvs;
	Vector<bool> smooth;
	Vector<Ref<Material>> materials;
	Vector<bool> invert_faces;

	int face_count = indices.size() / 3;
	faces.resize(face_count * 3);
	face_uvs.resize(face_count * 3);
	smooth.resize(face_count);
	materials.resize(face_count);
	invert_faces.resize(face_count);

	{
		Vector3 *facesw = faces.ptrw();
		Vector2 *face_uvsw = face_uvs.ptrw();
		bool *smoothw = smooth.ptrw();
		Ref<Material> *materialsw = materials.ptrw();
		bool *invertw = invert_faces.ptrw();

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

	_brush->build_from_faces(faces, face_uvs, smooth, materials, invert_faces);

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
		print_verbose(vformat("CSGSculptedTexture3D::_build_brush() - MANIFOLD VALIDATION FAILED"));
		Array errors = validation_result["errors"];
		for (int i = 0; i < errors.size(); i++) {
			print_verbose(vformat("  ERROR: %s", (String)errors[i]));
		}
		print_verbose("CSGSculptedTexture3D::_build_brush() - This may cause CSG operations to fail!");
		// Don't return nullptr - let the manifold library catch issues later
		// But warn the user that there are problems
	} else {
		print_verbose("CSGSculptedTexture3D::_build_brush() - Manifold validation passed");
	}

	return _brush;
}
