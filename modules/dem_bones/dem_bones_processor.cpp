/**************************************************************************/
/*  dem_bones_processor.cpp                                               */
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

#include "dem_bones_processor.h"

#include "dem_bones_extension.h"

#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/animation/animation_player.h"

void DemBonesProcessor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("process_animation", "animation_player", "node", "animation_name"), &DemBonesProcessor::process_animation);
	ClassDB::bind_method(D_METHOD("get_rest_vertices"), &DemBonesProcessor::get_rest_vertices);
	ClassDB::bind_method(D_METHOD("get_skinning_weights"), &DemBonesProcessor::get_skinning_weights);
	ClassDB::bind_method(D_METHOD("get_bone_transforms"), &DemBonesProcessor::get_bone_transforms);
	ClassDB::bind_method(D_METHOD("get_bone_count"), &DemBonesProcessor::get_bone_count);
}

Error DemBonesProcessor::process_animation(AnimationPlayer *p_animation_player, Node3D *p_node, const StringName &p_animation_name) {
	ERR_FAIL_NULL_V(p_animation_player, ERR_INVALID_PARAMETER);
	ERR_FAIL_NULL_V(p_node, ERR_INVALID_PARAMETER);

	Ref<Animation> animation = p_animation_player->get_animation(p_animation_name);
	ERR_FAIL_COND_V(animation.is_null(), ERR_INVALID_PARAMETER);

	Ref<Resource> mesh_resource;
	if (MeshInstance3D *mesh_instance = Object::cast_to<MeshInstance3D>(p_node)) {
		mesh_resource = mesh_instance->get_mesh();
	} else if (ImporterMeshInstance3D *importer_mesh_instance = Object::cast_to<ImporterMeshInstance3D>(p_node)) {
		mesh_resource = importer_mesh_instance->get_mesh();
	} else {
		ERR_FAIL_V_MSG(ERR_INVALID_PARAMETER, "Node must be MeshInstance3D or ImporterMeshInstance3D");
	}
	ERR_FAIL_COND_V(mesh_resource.is_null(), ERR_INVALID_PARAMETER);

	Mesh *mesh = Object::cast_to<Mesh>(mesh_resource.ptr());
	ImporterMesh *importer_mesh = Object::cast_to<ImporterMesh>(mesh_resource.ptr());
	ERR_FAIL_COND_V(!mesh && !importer_mesh, ERR_INVALID_PARAMETER);

	int blend_shape_count = mesh ? mesh->get_blend_shape_count() : importer_mesh->get_blend_shape_count();
	ERR_FAIL_COND_V(blend_shape_count == 0, ERR_INVALID_DATA);

	// Extract blend shape data from animation
	HashMap<NodePath, Vector<Vector3>> blend_shapes;
	NodePath mesh_path = p_animation_player->get_path_to(p_node);

	// Get animation length and frame rate
	float length = animation->get_length();
	constexpr float fps = 30.0f;
	int frame_count = Math::ceil(length * fps);

	// Initialize DemBones
	Dem::DemBonesExt<double, float> bones;
	bones.num_subjects = 1;
	bones.num_total_frames = frame_count;
	Array surface_arrays = mesh ? mesh->surface_get_arrays(0) : importer_mesh->get_surface_arrays(0);
	PackedVector3Array vertex_array = surface_arrays[ArrayMesh::ARRAY_VERTEX];
	bones.num_vertices = vertex_array.size();
	ERR_FAIL_COND_V(bones.num_vertices == 0, ERR_INVALID_DATA);

	// Set up frame data
	bones.frame_start_index.resize(2);
	bones.frame_start_index[0] = 0;
	bones.frame_start_index[1] = frame_count;

	bones.frame_subject_id.resize(frame_count);
	for (int i = 0; i < frame_count; i++) {
		bones.frame_subject_id[i] = 0;
	}

	// Set rest pose geometry
	bones.rest_pose_geometry.resize(3, bones.num_vertices);
	PackedVector3Array rest_vertices = surface_arrays[ArrayMesh::ARRAY_VERTEX];
	for (int i = 0; i < bones.num_vertices; i++) {
		bones.rest_pose_geometry(0, i) = rest_vertices[i].x;
		bones.rest_pose_geometry(1, i) = rest_vertices[i].y;
		bones.rest_pose_geometry(2, i) = rest_vertices[i].z;
	}

	// Set animated mesh data
	bones.vertex.resize(3 * frame_count, bones.num_vertices);

	float time_step = 1.0f / fps;
	for (int frame = 0; frame < frame_count; frame++) {
		float time = frame * time_step;
		if (time > length) {
			time = length;
		}

		// Sample animation at this time
		p_animation_player->seek(time, true);

		// Get deformed vertices
		for (int blend_i = 0; blend_i < blend_shape_count; blend_i++) {
			StringName blend_name = StringName(mesh ? String(mesh->get_blend_shape_name(blend_i)) : importer_mesh->get_blend_shape_name(blend_i));
			String track_path = String(mesh_path) + ":" + String(blend_name);

			// Find the track index for this blend shape
			int track_index = -1;
			for (int t = 0; t < animation->get_track_count(); t++) {
				if (animation->track_get_path(t) == NodePath(track_path)) {
					track_index = t;
					break;
				}
			}

			if (track_index == -1) {
				continue;
			}

			float weight = animation->blend_shape_track_interpolate(track_index, time);

			if (Math::is_zero_approx(weight)) {
				continue;
			}

			Array blend_array;
			if (mesh) {
				Array blend_arrays = mesh->surface_get_blend_shape_arrays(0);
				if (blend_i >= blend_arrays.size()) {
					continue;
				}
				blend_array = blend_arrays[blend_i];
			} else {
				blend_array = importer_mesh->get_surface_blend_shape_arrays(0, blend_i);
			}
			if (blend_array.size() <= ArrayMesh::ARRAY_VERTEX) {
				continue;
			}

			PackedVector3Array blend_vertices = blend_array[ArrayMesh::ARRAY_VERTEX];
			for (int v = 0; v < bones.num_vertices && v < blend_vertices.size(); v++) {
				rest_vertices.write[v] += blend_vertices[v] * weight;
			}
		}

		// Store frame data
		for (int v = 0; v < bones.num_vertices; v++) {
			bones.vertex(3 * frame + 0, v) = rest_vertices[v].x;
			bones.vertex(3 * frame + 1, v) = rest_vertices[v].y;
			bones.vertex(3 * frame + 2, v) = rest_vertices[v].z;
		}
	}

	// Set mesh topology (simplified - assume triangles)
	int surface_count = mesh->get_surface_count();
	bones.fv.clear();
	for (int s = 0; s < surface_count; s++) {
		Array surface_data = mesh ? mesh->surface_get_arrays(s) : importer_mesh->get_surface_arrays(s);
		if (surface_data.size() <= ArrayMesh::ARRAY_INDEX) {
			continue;
		}

		PackedInt32Array indices = surface_data[ArrayMesh::ARRAY_INDEX];
		for (int i = 0; i < indices.size(); i += 3) {
			std::vector<int> triangle;
			triangle.push_back(indices[i]);
			triangle.push_back(indices[i + 1]);
			triangle.push_back(indices[i + 2]);
			bones.fv.push_back(triangle);
		}
	}

	// Set bone count (start with reasonable default)
	bones.num_bones = 10; // This could be parameterized

	// Run DemBones computation
	bones.init();
	for (int iter = 0; iter < bones.nIters; iter++) {
		bones.computeTranformations();
		bones.computeWeights();
	}

	// Store results
	bone_count = bones.num_bones;

	// Convert results to Godot format
	rest_vertices.resize(bones.num_vertices);
	for (int i = 0; i < bones.num_vertices; i++) {
		rest_vertices.write[i] = Vector3(
				bones.rest_pose_geometry(0, i),
				bones.rest_pose_geometry(1, i),
				bones.rest_pose_geometry(2, i));
	}
	this->rest_vertices = rest_vertices;

	// Convert skinning weights
	skinning_weights.clear();
	for (int j = 0; j < bones.num_bones; j++) {
		Array bone_weights;
		for (int i = 0; i < bones.num_vertices; i++) {
			bone_weights.push_back(bones.skinning_weights.coeff(j, i));
		}
		skinning_weights.push_back(bone_weights);
	}

	// Convert bone transforms
	bone_transforms.clear();
	for (int k = 0; k < frame_count; k++) {
		Array frame_transforms;
		for (int j = 0; j < bones.num_bones; j++) {
			Transform3D transform;
			Eigen::Block<Eigen::MatrixXd, 4, 4> bone_mat = bones.bone_transform_mat.block<4, 4>(4 * k, 4 * j);
			transform.basis = Basis(
				Vector3(bone_mat(0, 0), bone_mat(1, 0), bone_mat(2, 0)),
				Vector3(bone_mat(0, 1), bone_mat(1, 1), bone_mat(2, 1)),
				Vector3(bone_mat(0, 2), bone_mat(1, 2), bone_mat(2, 2)));
			transform.origin = Vector3(bone_mat(0, 3), bone_mat(1, 3), bone_mat(2, 3));
			frame_transforms.push_back(transform);
		}
		bone_transforms.push_back(frame_transforms);
	}

	return OK;
}

PackedVector3Array DemBonesProcessor::get_rest_vertices() const {
	return rest_vertices;
}

Array DemBonesProcessor::get_skinning_weights() const {
	return skinning_weights;
}

Array DemBonesProcessor::get_bone_transforms() const {
	return bone_transforms;
}

int DemBonesProcessor::get_bone_count() const {
	return bone_count;
}

DemBonesProcessor::DemBonesProcessor() {
}
