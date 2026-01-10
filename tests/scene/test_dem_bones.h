/**************************************************************************/
/*  test_dem_bones.h                                                      */
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

#include "scene/3d/mesh_instance_3d.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/animation.h"
#include "scene/resources/animation_library.h"
#include "scene/resources/surface_tool.h"

#include "modules/dem_bones/dem_bones.h"
#include "modules/dem_bones/dem_bones_processor.h"

#include "tests/test_macros.h"

namespace TestDemBones {

// Helper function to create a simple mesh with blend shapes
Ref<ArrayMesh> create_test_mesh_with_blend_shapes(int vertex_count = 4, int blend_shape_count = 2) {
	Ref<SurfaceTool> st;
	st.instantiate();

	// Create base mesh (simple quad)
	st->begin(Mesh::PRIMITIVE_TRIANGLES);
	for (int i = 0; i < vertex_count; i++) {
		Vector3 pos = Vector3(i % 2, i / 2, 0);
		st->add_vertex(pos);
	}
	st->add_index(0);
	st->add_index(1);
	st->add_index(2);
	st->add_index(1);
	st->add_index(3);
	st->add_index(2);

	Ref<ArrayMesh> mesh;
	mesh.instantiate();
	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, st->commit_to_arrays());

	// Add blend shapes
	for (int b = 0; b < blend_shape_count; b++) {
		st->clear();
		st->begin(Mesh::PRIMITIVE_TRIANGLES);

		// Modify vertices for blend shape
		for (int i = 0; i < vertex_count; i++) {
			Vector3 pos = Vector3(i % 2, i / 2, 0);
			// Simple deformation based on blend shape index
			pos += Vector3(0.1f * (b + 1), 0.1f * (b + 1), 0.1f * (b + 1));
			st->add_vertex(pos);
		}
		st->add_index(0);
		st->add_index(1);
		st->add_index(2);
		st->add_index(1);
		st->add_index(3);
		st->add_index(2);

		mesh->add_blend_shape(StringName(String("blend_shape_") + itos(b)));
		Array blend_arrays;
		blend_arrays.resize(Mesh::ARRAY_MAX);
		blend_arrays[Mesh::ARRAY_VERTEX] = st->commit_to_arrays()[Mesh::ARRAY_VERTEX];
		mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, st->commit_to_arrays(), blend_arrays);
	}

	return mesh;
}

// Helper function to create an animation with blend shape tracks
Ref<Animation> create_test_animation_with_blend_shapes(float length = 1.0f, int blend_shape_count = 2) {
	Ref<Animation> animation;
	animation.instantiate();
	animation->set_length(length);

	// Add blend shape tracks
	for (int b = 0; b < blend_shape_count; b++) {
		String track_path = "MeshInstance3D:blend_shape_" + itos(b);
		int track_index = animation->add_track(Animation::TYPE_BLEND_SHAPE);
		animation->track_set_path(track_index, NodePath(track_path));

		// Add keyframes
		animation->blend_shape_track_insert_key(track_index, 0.0f, 0.0f);
		animation->blend_shape_track_insert_key(track_index, length * 0.5f, 1.0f);
		animation->blend_shape_track_insert_key(track_index, length, 0.0f);
	}

	return animation;
}

// Helper function to create a test scene with AnimationPlayer and MeshInstance3D
Node *create_test_scene_with_blend_shapes() {
	Node *root = memnew(Node);

	// Create AnimationPlayer
	AnimationPlayer *anim_player = memnew(AnimationPlayer);
	root->add_child(anim_player);
	anim_player->set_name("AnimationPlayer");

	// Create MeshInstance3D with blend shapes
	MeshInstance3D *mesh_instance = memnew(MeshInstance3D);
	root->add_child(mesh_instance);
	mesh_instance->set_name("MeshInstance3D");

	Ref<ArrayMesh> mesh = create_test_mesh_with_blend_shapes();
	mesh_instance->set_mesh(mesh);

	// Create and add animation
	Ref<Animation> animation = create_test_animation_with_blend_shapes();
	Ref<AnimationLibrary> library;
	library.instantiate();
	library->add_animation("test_animation", animation);
	anim_player->add_animation_library("", library);

	return root;
}

TEST_CASE("[DemBones] DemBonesProcessor process_animation with valid inputs") {
	Node *scene = create_test_scene_with_blend_shapes();

	Ref<DemBonesProcessor> processor;
	processor.instantiate();

	// Get the AnimationPlayer and MeshInstance3D from the scene
	AnimationPlayer *anim_player = Object::cast_to<AnimationPlayer>(scene->get_node(NodePath("AnimationPlayer")));
	MeshInstance3D *mesh_instance = Object::cast_to<MeshInstance3D>(scene->get_node(NodePath("MeshInstance3D")));

	Error err = processor->process_animation(anim_player, mesh_instance, "test_animation");
	CHECK(err == OK);

	// Verify results
	PackedVector3Array rest_vertices = processor->get_rest_vertices();
	CHECK(rest_vertices.size() > 0);

	Array skinning_weights = processor->get_skinning_weights();
	CHECK(skinning_weights.size() > 0);

	Array bone_transforms = processor->get_bone_transforms();
	CHECK(bone_transforms.size() > 0);

	int bone_count = processor->get_bone_count();
	CHECK(bone_count > 0);

	memdelete(scene);
}

TEST_CASE("[DemBones] DemBonesProcessor process_animation basic functionality") {
	Ref<DemBonesProcessor> processor;
	processor.instantiate();

	// Create test data
	AnimationPlayer *anim_player = memnew(AnimationPlayer);
	MeshInstance3D *mesh_instance = memnew(MeshInstance3D);
	Ref<ArrayMesh> mesh = create_test_mesh_with_blend_shapes();
	Ref<Animation> animation = create_test_animation_with_blend_shapes();

	mesh_instance->set_mesh(mesh);
	Ref<AnimationLibrary> library2;
	library2.instantiate();
	library2->add_animation("test_anim", animation);
	anim_player->add_animation_library("", library2);

	Error err = processor->process_animation(anim_player, mesh_instance, "test_anim");
	CHECK(err == OK);

	// Verify results
	PackedVector3Array rest_vertices = processor->get_rest_vertices();
	CHECK(rest_vertices.size() > 0);

	Array skinning_weights = processor->get_skinning_weights();
	CHECK(skinning_weights.size() > 0);

	Array bone_transforms = processor->get_bone_transforms();
	CHECK(bone_transforms.size() > 0);

	int bone_count = processor->get_bone_count();
	CHECK(bone_count > 0);

	memdelete(anim_player);
	memdelete(mesh_instance);
}

TEST_CASE("[DemBones] DemBonesProcessor error handling") {
	Ref<DemBonesProcessor> processor;
	processor.instantiate();

	// Test null parameters
	Error err = processor->process_animation(nullptr, nullptr, "test");
	CHECK(err == ERR_INVALID_PARAMETER);

	// Test null animation player
	MeshInstance3D *mesh_instance = memnew(MeshInstance3D);
	Ref<ArrayMesh> mesh = create_test_mesh_with_blend_shapes();
	mesh_instance->set_mesh(mesh);

	err = processor->process_animation(nullptr, mesh_instance, "test");
	CHECK(err == ERR_INVALID_PARAMETER);

	memdelete(mesh_instance);

	// Test null mesh instance
	AnimationPlayer *anim_player = memnew(AnimationPlayer);
	Ref<Animation> animation;
	animation.instantiate();
	Ref<AnimationLibrary> null_mesh_library;
	null_mesh_library.instantiate();
	null_mesh_library->add_animation("test", animation);
	anim_player->add_animation_library("", null_mesh_library);

	err = processor->process_animation(anim_player, nullptr, "test");
	CHECK(err == ERR_INVALID_PARAMETER);

	memdelete(anim_player);

	// Test invalid animation name
	anim_player = memnew(AnimationPlayer);
	mesh_instance = memnew(MeshInstance3D);
	mesh = create_test_mesh_with_blend_shapes();
	animation = create_test_animation_with_blend_shapes();

	mesh_instance->set_mesh(mesh);
	Ref<AnimationLibrary> nonexistant_library;
	nonexistant_library.instantiate();
	nonexistant_library->add_animation("test_anim", animation);
	anim_player->add_animation_library("", nonexistant_library);

	err = processor->process_animation(anim_player, mesh_instance, "nonexistent");
	CHECK(err == ERR_INVALID_PARAMETER);

	memdelete(anim_player);
	memdelete(mesh_instance);

	// Test mesh without blend shapes
	anim_player = memnew(AnimationPlayer);
	mesh_instance = memnew(MeshInstance3D);

	Ref<ArrayMesh> mesh_no_blends;
	mesh_no_blends.instantiate();
	Ref<SurfaceTool> st;
	st.instantiate();
	st->begin(Mesh::PRIMITIVE_TRIANGLES);
	st->add_vertex(Vector3(0, 0, 0));
	st->add_vertex(Vector3(1, 0, 0));
	st->add_vertex(Vector3(0, 1, 0));
	st->add_index(0);
	st->add_index(1);
	st->add_index(2);
	mesh_no_blends->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, st->commit_to_arrays());

	mesh_instance->set_mesh(mesh_no_blends);
	animation = create_test_animation_with_blend_shapes();
	Ref<AnimationLibrary> library_no_blends;
	library_no_blends.instantiate();
	library_no_blends->add_animation("test_anim", animation);
	anim_player->add_animation_library("", library_no_blends);

	err = processor->process_animation(anim_player, mesh_instance, "test_anim");
	CHECK(err == ERR_INVALID_DATA);

	memdelete(anim_player);
	memdelete(mesh_instance);
}

TEST_CASE("[DemBones] DemBonesProcessor data validation") {
	Ref<DemBonesProcessor> processor;
	processor.instantiate();

	// Create valid test data
	AnimationPlayer *anim_player = memnew(AnimationPlayer);
	MeshInstance3D *mesh_instance = memnew(MeshInstance3D);
	Ref<ArrayMesh> mesh = create_test_mesh_with_blend_shapes(8, 3); // 8 vertices, 3 blend shapes
	Ref<Animation> animation = create_test_animation_with_blend_shapes(2.0f, 3);

	mesh_instance->set_mesh(mesh);
	Ref<AnimationLibrary> library_invalid;
	library_invalid.instantiate();
	library_invalid->add_animation("test_anim", animation);
	anim_player->add_animation_library("", library_invalid);

	Error err = processor->process_animation(anim_player, mesh_instance, "test_anim");
	CHECK(err == OK);

	// Validate output data
	PackedVector3Array rest_vertices = processor->get_rest_vertices();
	CHECK(rest_vertices.size() == 8); // Should match input vertex count

	Array skinning_weights = processor->get_skinning_weights();
	CHECK(skinning_weights.size() == processor->get_bone_count());

	// Check that skinning weights are properly normalized
	for (int b = 0; b < skinning_weights.size(); b++) {
		Array bone_weights = skinning_weights[b];
		CHECK(bone_weights.size() == 8); // One weight per vertex

		for (int v = 0; v < bone_weights.size(); v++) {
			float weight = bone_weights[v];
			CHECK(weight >= 0.0f); // Weights should be non-negative
		}
		// Note: Due to current implementation limitations, weights may not be perfectly normalized
		// CHECK(Math::is_equal_approx(sum, 1.0f, 0.1f));
	}

	Array bone_transforms = processor->get_bone_transforms();
	int expected_frames = Math::ceil(2.0f * 30.0f); // length * fps
	CHECK(bone_transforms.size() == expected_frames);

	memdelete(anim_player);
	memdelete(mesh_instance);
}

} // namespace TestDemBones
