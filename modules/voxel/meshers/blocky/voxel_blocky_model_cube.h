/**************************************************************************/
/*  voxel_blocky_model_cube.h                                             */
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

#pragma once

#include "voxel_blocky_model.h"

namespace zylann::voxel {

// Cubic model, with configurable tiles on each side
// TODO Would it be better to add a new PrimitiveMesh doing this, and use VoxelBlockyMesh?
class VoxelBlockyModelCube : public VoxelBlockyModel {
	GDCLASS(VoxelBlockyModelCube, VoxelBlockyModel)
public:
	VoxelBlockyModelCube();

	static Cube::Side name_to_side(const String &s);

	Vector2i get_tile(VoxelBlockyModel::Side side) const {
		return _tiles[side];
	}

	void set_tile(VoxelBlockyModel::Side side, Vector2i pos);

	void set_height(float h);
	float get_height() const;

	void set_atlas_size_in_tiles(Vector2i s);
	Vector2i get_atlas_size_in_tiles() const;

	void bake(blocky::ModelBakingContext &ctx) const override;
	bool is_empty() const override;

	Ref<Mesh> get_preview_mesh() const override;

	void rotate_tiles_90(const math::Axis axis, const bool clockwise);
	void rotate_tiles_ortho(const math::OrthoBasis ortho_basis);

private:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	static void _bind_methods();

	FixedArray<Vector2i, Cube::SIDE_COUNT> _tiles;
	float _height = 1.f;

	Vector2i _atlas_size_in_tiles;

	uint8_t _mesh_ortho_rotation = 0;
};

void make_cube_side_vertices(StdVector<Vector3f> &positions, const unsigned int side_index, const float height);
void make_cube_side_indices(StdVector<int> &indices, const unsigned int side_index);
void make_cube_side_tangents(StdVector<float> &tangents, const unsigned int side_index);

} // namespace zylann::voxel
