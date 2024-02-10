/**************************************************************************/
/*  vox_importer.cpp                                                      */
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

#include "vox_importer.h"
#include "constants/voxel_string_names.h"
#include "meshers/cubes/voxel_mesher_cubes.h"
#include "modules/voxel_mesh/util/math/vector3i.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/material.h"
#include "storage/voxel_buffer.h"
#include "streams/vox_data.h"
#include "util/godot/funcs.h"

#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/resources/importer_mesh.h"
#include <core/io/file_access.h>
#include <scene/3d/mesh_instance_3d.h>
#include <scene/3d/node_3d.h>
#include <scene/resources/mesh.h>
#include <scene/resources/packed_scene.h>

Error VoxelVoxImporter::process_scene_node_recursively(const vox::Data &data, int node_id,
		Node3D *parent_node,
		Node3D *&root_node, int depth,
		const Vector<VoxMesh> &meshes) {
	ERR_FAIL_COND_V(depth > 10, ERR_INVALID_DATA);
	const vox::Node *vox_node = data.get_node(node_id);

	switch (vox_node->type) {
		case vox::Node::TYPE_TRANSFORM: {
			Node3D *node = memnew(Node3D);
			if (root_node == nullptr) {
				root_node = node;
			} else {
				ERR_FAIL_COND_V(parent_node == nullptr, ERR_BUG);
				parent_node->add_child(node, true);
				node->set_owner(root_node);
			}
			const vox::TransformNode *vox_transform_node =
					reinterpret_cast<const vox::TransformNode *>(vox_node);
			node->set_transform(Transform3D(vox_transform_node->rotation.basis,
					vox_transform_node->position.to_vec3()));
			process_scene_node_recursively(data, vox_transform_node->child_node_id,
					node, root_node, depth + 1, meshes);
		} break;

		case vox::Node::TYPE_GROUP: {
			const vox::GroupNode *vox_group_node =
					reinterpret_cast<const vox::GroupNode *>(vox_node);
			for (unsigned int i = 0; i < vox_group_node->child_node_ids.size(); ++i) {
				const int child_node_id = vox_group_node->child_node_ids[i];
				process_scene_node_recursively(data, child_node_id, parent_node,
						root_node, depth + 1, meshes);
			}
		} break;

		case vox::Node::TYPE_SHAPE: {
			ERR_FAIL_COND_V(parent_node == nullptr, ERR_BUG);
			ERR_FAIL_COND_V(root_node == nullptr, ERR_BUG);
			const vox::ShapeNode *vox_shape_node =
					reinterpret_cast<const vox::ShapeNode *>(vox_node);
			const VoxMesh &mesh_data = meshes[vox_shape_node->model_id];
			ERR_FAIL_COND_V(mesh_data.mesh.is_null(), ERR_BUG);
			const Vector3 offset = -mesh_data.pivot;
			add_mesh_instance(mesh_data.mesh, parent_node, root_node, offset);
		} break;

		default:
			ERR_FAIL_V(ERR_INVALID_DATA);
			break;
	}

	return OK;
}

Ref<ImporterMesh>
VoxelVoxImporter::build_mesh(VoxelBuffer &voxels, VoxelMesher &mesher,
		Ref<Image> &out_atlas) {
	//
	VoxelMesher::Output output;
	VoxelMesher::Input input = { voxels, 0 };
	mesher.build(output, input);

	if (output.surfaces.is_empty()) {
		return Ref<ImporterMesh>();
	}

	Ref<ImporterMesh> mesh;
	mesh.instantiate();

	for (int i = 0; i < output.surfaces.size(); ++i) {
		Array surface = output.surfaces[i];

		if (surface.is_empty()) {
			continue;
		}

		CRASH_COND(surface.size() != Mesh::ARRAY_MAX);
		if (!is_surface_triangulated(surface)) {
			continue;
		}

		Ref<StandardMaterial3D> material;
		material.instantiate();
		material->set_roughness(1.f);
		material->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
		if (output.atlas_image.is_valid()) {
			out_atlas = output.atlas_image;
		}
		Ref<ImageTexture> texture;
		texture.instantiate();
		texture->set_image(out_atlas);
		material->set_texture(BaseMaterial3D::TEXTURE_ALBEDO, texture);
		material->set_texture_filter(BaseMaterial3D::TEXTURE_FILTER_NEAREST_WITH_MIPMAPS);
		mesh->add_surface(output.primitive_type, surface, Array(), Dictionary(), material);
	}

	return mesh;
}

void VoxelVoxImporter::add_mesh_instance(Ref<ImporterMesh> mesh, Node *parent, Node *owner,
		Vector3 offset) {
	ImporterMeshInstance3D *mesh_instance = memnew(ImporterMeshInstance3D);
	mesh_instance->set_mesh(mesh);
	mesh_instance->set_name(MeshInstance3D::get_class_static());
	parent->add_child(mesh_instance, true);
	mesh_instance->set_owner(owner);
	mesh_instance->set_position(offset);
}

Node *VoxelVoxImporter::import_scene(const String &p_path, uint32_t p_flags, const HashMap<StringName, Variant> &p_options, List<String> *r_missing_deps, Error *r_err) {
	vox::Data data;
	const Error load_err = data.load_from_file(p_path);
	if (load_err != OK && r_err) {
		*r_err = load_err;
		return nullptr;
	}

	Vector<VoxMesh> meshes;
	meshes.resize(data.get_model_count());

	Ref<VoxelColorPalette> palette;
	palette.instantiate();
	for (unsigned int i = 0; i < data.get_palette().size(); ++i) {
		Color8 color = data.get_palette()[i];
		palette->set_color8(i, color);
	}

	Ref<VoxelMesherCubes> mesher;
	mesher.instantiate();
	mesher->set_color_mode(VoxelMesherCubes::COLOR_MESHER_PALETTE);
	mesher->set_palette(palette);
	mesher->set_greedy_meshing_enabled(true);
	mesher->set_store_colors_in_texture(true);

	for (unsigned int model_index = 0; model_index < data.get_model_count();
			++model_index) {
		const vox::Model &model = data.get_model(model_index);
		Ref<VoxelBuffer> voxels;
		voxels.instantiate();
		voxels->create(model.size + VoxelVector3i(VoxelMesherCubes::PADDING * 2));
		voxels->decompress_channel(VoxelBuffer::CHANNEL_COLOR);
		Span<uint8_t> dst_color_indices;
		if (!voxels->get_channel_raw(VoxelBuffer::CHANNEL_COLOR, dst_color_indices)) {
			if (r_err) {
				*r_err = ERR_BUG;
			}
			return nullptr;
		}
		Span<const uint8_t> src_color_indices = to_span_const(model.color_indexes);
		copy_3d_region_zxy(dst_color_indices, voxels->get_size(),
				VoxelVector3i(VoxelMesherCubes::PADDING),
				src_color_indices, model.size, VoxelVector3i(),
				model.size);
		Ref<Image> atlas;
		Ref<ImporterMesh> mesh = build_mesh(**voxels, **mesher, atlas);

		if (mesh.is_null()) {
			continue;
		}

		VoxMesh mesh_info;
		mesh_info.mesh = mesh;
		// In MagicaVoxel scene graph, pivots are at the center of models, not at
		// the lower corner.
		mesh_info.pivot = (voxels->get_size() / VoxelVector3i(2, 2, 2) - VoxelVector3i(1)).to_vec3();
		meshes.write[model_index] = mesh_info;
	}

	Node3D *root_node = memnew(Node3D);
	if (data.get_root_node_id() != -1) {
		// Convert scene graph into a node tree
		process_scene_node_recursively(data, data.get_root_node_id(), root_node,
				root_node, 0, meshes);
	} else if (meshes.size() > 0) {
		// Some vox files don't have a scene graph
		const VoxMesh &mesh0 = meshes[0];
		add_mesh_instance(mesh0.mesh, root_node, root_node, Vector3());
	}
	return root_node;
}
