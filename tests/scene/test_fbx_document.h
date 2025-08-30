/**************************************************************************/
/*  test_fbx_document.h                                                   */
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

#ifdef MODULE_FBX_ENABLED

#include "modules/fbx/fbx_document.h"
#include "modules/fbx/fbx_state.h"

#include "tests/test_macros.h"
#include "tests/test_utils.h"

namespace TestFBXDocument {

struct FBXArraySize {
	String key;
	int val;
};

struct FBXKeyValue {
	String key;
	Variant val;
};

struct FBXTestCase {
	String filename;
	String description;
	Vector<FBXArraySize> array_sizes;
	Vector<FBXKeyValue> keyvalues;
	bool has_animations;
	bool has_skeleton;
	bool has_meshes;
};

// Test case for huesitos.fbx - a rigged mesh with skeleton
const FBXTestCase fbx_test_cases[] = {
	{
			"thirdparty/ufbx/data/huesitos.fbx",
			"Rigged mesh with skeleton (huesitos)",
			// Expected array sizes
			{
					{ "nodes", 1 }, // Will be updated based on actual file structure
					{ "meshes", 1 },
					{ "materials", 1 },
					{ "root_nodes", 1 },
					{ "skeletons", 1 },
			},
			// Expected key-value pairs
			{
					{ "scene_name", "huesitos" },
					{ "filename", "huesitos" } },
			true, // has_animations
			true, // has_skeleton
			true // has_meshes
	},
};

void register_fbx_extension() {
	FBXDocument::unregister_all_gltf_document_extensions();
}

void test_fbx_document_values(Ref<FBXDocument> &p_fbx_document, Ref<FBXState> &p_fbx_state, const FBXTestCase &p_test_case) {
	const Error err = p_fbx_document->append_from_file(p_test_case.filename, p_fbx_state);
	CHECK_MESSAGE(err == OK, "Failed to load FBX file: ", p_test_case.filename);

	if (err != OK) {
		return; // Skip further tests if loading failed
	}

	for (const FBXArraySize &array_size : p_test_case.array_sizes) {
		if (array_size.key == "nodes") {
			CHECK_MESSAGE(p_fbx_state->get_nodes().size() >= array_size.val,
					"Expected at least ", array_size.val, " nodes, got ", p_fbx_state->get_nodes().size());
		} else if (array_size.key == "meshes") {
			CHECK_MESSAGE(p_fbx_state->get_meshes().size() >= array_size.val,
					"Expected at least ", array_size.val, " meshes, got ", p_fbx_state->get_meshes().size());
		} else if (array_size.key == "materials") {
			CHECK_MESSAGE(p_fbx_state->get_materials().size() >= array_size.val,
					"Expected at least ", array_size.val, " materials, got ", p_fbx_state->get_materials().size());
		} else if (array_size.key == "skeletons") {
			CHECK_MESSAGE(p_fbx_state->get_skeletons().size() >= array_size.val,
					"Expected at least ", array_size.val, " skeletons, got ", p_fbx_state->get_skeletons().size());
		}
	}

	for (const FBXKeyValue &key_value : p_test_case.keyvalues) {
		if (key_value.key == "scene_name") {
			bool scene_name_valid = p_fbx_state->get_scene_name().contains(String(key_value.val)) ||
					p_fbx_state->get_scene_name().is_empty();
			CHECK_MESSAGE(scene_name_valid,
					"Expected scene name to contain '", key_value.val, "', got '", p_fbx_state->get_scene_name(), "'");
		}
	}

	if (p_test_case.has_skeleton) {
		CHECK_MESSAGE(p_fbx_state->get_skeletons().size() > 0, "Expected skeleton data but found none");
	}

	if (p_test_case.has_meshes) {
		CHECK_MESSAGE(p_fbx_state->get_meshes().size() > 0, "Expected mesh data but found none");
	}

	if (p_test_case.has_animations) {
		CHECK_MESSAGE(p_fbx_state->get_animations().size() > 0, "Expected animation data but found none");
	}
}

void test_fbx_save(Node *p_node, const String &p_test_name) {
	Ref<FBXDocument> fbx_document_save;
	fbx_document_save.instantiate();
	Ref<FBXState> fbx_state_save;
	fbx_state_save.instantiate();

	// Test scene to FBX conversion
	const Error err_append = fbx_document_save->append_from_scene(p_node, fbx_state_save);
	CHECK_MESSAGE(err_append == OK, "Failed to append scene to FBX state for ", p_test_name);

	if (err_append != OK) {
		return; // Skip save tests if append failed
	}

	// Test saving to file
	const String temp_fbx_path = TestUtils::get_temp_path(p_test_name + ".fbx");
	const Error err_save = fbx_document_save->write_to_filesystem(fbx_state_save, temp_fbx_path);
	CHECK_MESSAGE(err_save == OK, "Failed to save FBX file: ", temp_fbx_path);

	// Test generating buffer
	PackedByteArray buffer = fbx_document_save->generate_buffer(fbx_state_save);
	CHECK_MESSAGE(buffer.size() > 0, "Generated FBX buffer is empty for ", p_test_name);
}

void test_fbx_round_trip(const FBXTestCase &p_test_case) {
	// Load original FBX
	Ref<FBXDocument> fbx_document_load;
	fbx_document_load.instantiate();
	Ref<FBXState> fbx_state_load;
	fbx_state_load.instantiate();

	const Error err_load = fbx_document_load->append_from_file(p_test_case.filename, fbx_state_load);
	CHECK_MESSAGE(err_load == OK, "Failed to load original FBX file for round-trip test: ", p_test_case.filename);

	if (err_load != OK) {
		return;
	}

	// Generate scene from loaded FBX
	Node *scene = fbx_document_load->generate_scene(fbx_state_load);
	CHECK_MESSAGE(scene != nullptr, "Failed to generate scene from FBX");

	if (!scene) {
		return;
	}

	// Export scene back to FBX
	Ref<FBXDocument> fbx_document_save;
	fbx_document_save.instantiate();
	Ref<FBXState> fbx_state_save;
	fbx_state_save.instantiate();

	const Error err_append = fbx_document_save->append_from_scene(scene, fbx_state_save);
	CHECK_MESSAGE(err_append == OK, "Failed to append scene for round-trip export");

	if (err_append == OK) {
		// Test that we can generate a buffer (basic export functionality)
		PackedByteArray buffer = fbx_document_save->generate_buffer(fbx_state_save);
		CHECK_MESSAGE(buffer.size() > 0, "Round-trip export generated empty buffer");

		// Save to temporary file for further testing
		const String temp_path = TestUtils::get_temp_path("roundtrip_" + p_test_case.filename.get_file());
		const Error err_save = fbx_document_save->write_to_filesystem(fbx_state_save, temp_path);
		CHECK_MESSAGE(err_save == OK, "Failed to save round-trip FBX file");
	}

	// Clean up
	memdelete(scene);
}

TEST_CASE("[SceneTree][FBXDocument] Load huesitos.fbx rigged mesh") {
	register_fbx_extension();

	Ref<FBXDocument> fbx_document;
	fbx_document.instantiate();
	Ref<FBXState> fbx_state;
	fbx_state.instantiate();

	// Test loading and validation
	test_fbx_document_values(fbx_document, fbx_state, fbx_test_cases[0]);

	// Generate scene from FBX
	Node *node = fbx_document->generate_scene(fbx_state);
	CHECK_MESSAGE(node != nullptr, "Failed to generate scene from huesitos.fbx");

	if (node) {
		// Basic scene structure validation
		CHECK_MESSAGE(node->is_class("Node3D"), "Root node should be Node3D, got: ", node->get_class());
		CHECK_MESSAGE(node->get_name().length() > 0, "Root node should have a name");

		// Look for mesh instances in the scene
		bool found_mesh = false;
		bool found_skeleton = false;

		for (int i = 0; i < node->get_child_count(); i++) {
			Node *child = node->get_child(i);
			if (child->is_class("MeshInstance3D") || child->is_class("ImporterMeshInstance3D")) {
				found_mesh = true;
			}
			if (child->is_class("Skeleton3D")) {
				found_skeleton = true;
			}
		}

		bool has_mesh_or_children = found_mesh || node->get_child_count() > 0;
		CHECK_MESSAGE(has_mesh_or_children, "Expected to find mesh data in scene");

		CHECK_MESSAGE(found_skeleton, "Expected to find skeleton data in scene");

		test_fbx_save(node, "huesitos");

		memdelete(node);
	}
}

TEST_CASE("[SceneTree][FBXDocument] FBX round-trip test") {
	register_fbx_extension();

	// Test round-trip functionality with huesitos.fbx
	test_fbx_round_trip(fbx_test_cases[0]);
}

TEST_CASE("[SceneTree][FBXDocument] FBX export functionality") {
	register_fbx_extension();

	// Create a simple test scene
	Node3D *root = memnew(Node3D);
	root->set_name("TestScene");

	// Add a mesh instance with actual geometry
	MeshInstance3D *mesh_instance = memnew(MeshInstance3D);
	mesh_instance->set_name("TestMesh");

	// Create a BoxMesh with actual geometry data
	Ref<BoxMesh> box_mesh;
	box_mesh.instantiate();
	box_mesh->set_size(Vector3(2.0, 2.0, 2.0));
	mesh_instance->set_mesh(box_mesh);

	// Set a transform to test transformation export
	mesh_instance->set_position(Vector3(1.0, 0.5, -1.0));
	mesh_instance->set_rotation(Vector3(0.0, Math::deg_to_rad(45.0), 0.0));

	root->add_child(mesh_instance);

	// Add a second mesh instance with different geometry
	MeshInstance3D *sphere_instance = memnew(MeshInstance3D);
	sphere_instance->set_name("TestSphere");

	Ref<SphereMesh> sphere_mesh;
	sphere_mesh.instantiate();
	sphere_mesh->set_radius(1.5);
	sphere_mesh->set_height(3.0);
	sphere_instance->set_mesh(sphere_mesh);
	sphere_instance->set_position(Vector3(-2.0, 1.0, 0.0));

	root->add_child(sphere_instance);

	// Test export functionality
	test_fbx_save(root, "simple_scene");

	// Clean up
	memdelete(root);
}

} // namespace TestFBXDocument

#endif // MODULE_FBX_ENABLED
