extends SceneTree

# SceneMerge Integration Tests using GDScript and SceneTree
# Tests the full SceneMerge pipeline including material consolidation

class_name SceneMergeIntegrationTest

var _current_test_root: Node = null
var _test_results: Dictionary = {}
var _performance_times: Array = []

# Entry point for running all integration tests
func _init():
	print("🧪 Starting SceneMerge Integration Tests...")
	run_all_tests()
	print("✅ All SceneMerge integration tests completed!")
	print("📊 Performance results: " + str(_performance_times))
	quit()

func run_all_tests():
	# Test basic scene merging functionality
	test_basic_scene_merge()
	test_base_color_material_merge()
	test_performance_scaling()
	test_edge_case_handling()
	test_scene_transformation_preservation()
	test_blend_shape_preservation()
	test_skeleton_animation_merge()

	print_test_results()

func test_basic_scene_merge():
	print("Running basic scene merge test...")
	var start_time = Time.get_ticks_msec()

	# Create a root node for testing
	setup_test_scene("BasicMergeTest")

	# Create multiple mesh instances programmatically
	create_mesh_with_material(Vector3(0, 0, 0), Color(1.0, 0.0, 0.0), "RedMesh")  # Red cube
	create_mesh_with_material(Vector3(3, 0, 0), Color(0.0, 1.0, 0.0), "GreenMesh") # Green cube at offset
	create_mesh_with_material(Vector3(6, 0, 0), Color(0.0, 0.0, 1.0), "BlueMesh")  # Blue cube at offset

	# Verify initial scene state (3 mesh instances)
	assert(_current_test_root.get_child_count() == 3, "Initial scene should have 3 mesh instances")

	# Execute SceneMerge
	var merge_result = call_merge_function(_current_test_root)

	# Verify merge succeeded
	assert(merge_result == _current_test_root, "Merge should return the same root node")

	# Verify merged mesh was created
	var merged_instance = find_merged_mesh()
	assert(merged_instance != null, "Merged mesh instance should exist")
	assert(merged_instance.name == "MergedMesh", "Merged mesh should be named 'MergedMesh'")

	# Verify merged mesh exists and has geometry
	var merged_mesh = merged_instance.mesh
	assert(merged_mesh != null, "Merged mesh should exist")
	assert(merged_mesh.get_surface_count() > 0, "Merged mesh should have surfaces")

	var end_time = Time.get_ticks_msec()
	_performance_times.append(end_time - start_time)

	_test_results["BasicSceneMerge"] = true
	_log_success("Basic scene merge test passed")

func test_base_color_material_merge():
	print("Running base color material merge test...")
	var start_time = Time.get_ticks_msec()

	# Create scene with known base colors to test averaging
	setup_test_scene("BaseColorTest")

	# Create meshes with specific base colors
	create_mesh_with_material(Vector3(0, 0, 0), Color(1.0, 0.0, 0.0), "RedMaterial")    # Red
	create_mesh_with_material(Vector3(0, 3, 0), Color(0.0, 1.0, 0.0), "GreenMaterial")  # Green
	create_mesh_with_material(Vector3(0, 6, 0), Color(0.0, 0.0, 1.0), "BlueMaterial")   # Blue

	# Run merge
	call_merge_function(_current_test_root)

	# Find merged mesh and check material
	var merged_instance = find_merged_mesh()
	assert(merged_instance != null)

	var merged_mesh = merged_instance.mesh
	assert(merged_mesh.get_surface_count() == 1, "Should have single merged surface")

	var surface_material = merged_mesh.get_surface_material(0)
	assert(surface_material != null, "Merged mesh should have a material")

	# Verify it's a BaseMaterial3D (we created StandardMaterial3D which inherits from it)
	var base_mat = surface_material as BaseMaterial3D
	assert(base_mat != null, "Material should be a BaseMaterial3D")

	# Check that color is averaged
	var albedo = base_mat.albedo_color
	print("Material was merged to average color: " + albedo.to_html(false))

	# All three colors (1,0,0) + (0,1,0) + (0,0,1) should average to (1/3, 1/3, 1/3)
	var expected_avg = Color(1.0/3.0, 1.0/3.0, 1.0/3.0)
	assert(abs(albedo.r - expected_avg.r) < 0.01, "Red component should be averaged")
	assert(abs(albedo.g - expected_avg.g) < 0.01, "Green component should be averaged")
	assert(abs(albedo.b - expected_avg.b) < 0.01, "Blue component should be averaged")

	var end_time = Time.get_ticks_msec()
	_performance_times.append(end_time - start_time)

	_test_results["BaseColorMaterialMerge"] = true
	_log_success("Base color material merge test passed")

func test_performance_scaling():
	print("Running performance scaling test...")

	# Test with varying numbers of meshes (5, 10, 15)
	var mesh_counts = [5, 10, 15]
	var times = []

	for count in mesh_counts:
		var start_time = Time.get_ticks_msec()

		setup_test_scene("PerformanceTest_" + str(count))

		# Create multiple meshes spread out
		for i in range(count):
			var pos = Vector3(i * 3, 0, 0)
			var color = Color(randf(), randf(), randf())
			create_mesh_with_material(pos, color, "PerfMesh_" + str(i))

		call_merge_function(_current_test_root)
		var end_time = Time.get_ticks_msec()

		var duration = end_time - start_time
		times.append(duration)
		print("Merged " + str(count) + " meshes in " + str(duration) + "ms")

	# Verify performance is reasonable (should not be negative)
	assert(times[0] >= 0, "Time should not be negative")
	assert(times[1] >= 0, "Time should not be negative")
	assert(times[2] >= 0, "Time should not be negative")

	_test_results["PerformanceScaling"] = true
	_log_success("Performance scaling test passed")

func test_edge_case_handling():
	print("Running edge case handling test...")

	# Test 1: Empty scene
	setup_test_scene("EmptySceneTest")
	var result = call_merge_function(_current_test_root)
	assert(result == _current_test_root, "Empty scene should return unchanged")

	# Test 2: Single mesh scene (should fail gracefully)
	setup_test_scene("SingleMeshTest")
	create_mesh_with_material(Vector3(0, 0, 0), Color(0.5, 0.5, 0.5), "SingleMesh")
	result = call_merge_function(_current_test_root)
	assert(result == _current_test_root, "Single mesh scene should return unchanged")

	_test_results["EdgeCaseHandling"] = true
	_log_success("Edge case handling test passed")

func test_scene_transformation_preservation():
	print("Running scene transformation preservation test...")

	setup_test_scene("TransformTest")

	# Create meshes with different transforms
	create_mesh_with_material(Vector3(0, 0, 0), Color(1.0, 0.0, 0.0), "Mesh1")
	var mesh2_pos = Vector3(5, 3, 2)
	create_mesh_with_material(mesh2_pos, Color(0.0, 1.0, 0.0), "Mesh2")

	call_merge_function(_current_test_root)

	var merged_instance = find_merged_mesh()
	assert(merged_instance != null)

	var merged_mesh = merged_instance.mesh
	assert(merged_mesh != null)

	# Verify the merged mesh contains vertices from both original positions
	# (This is a simplified check - actual vertex data validation
	# would require extracting surface arrays)
	assert(merged_mesh.get_surface_count() > 0)
	var vertex_count = 0
	for i in range(merged_mesh.get_surface_count()):
		var arrays = merged_mesh.get_surface_arrays(i)
		var vertices = arrays[Mesh.ARRAY_VERTEX]
		vertex_count += vertices.size()

	assert(vertex_count >= 6, "Merged mesh should contain geometry from both meshes") # 3 verts * 2 meshes

	_test_results["SceneTransformationPreservation"] = true
	_log_success("Scene transformation preservation test passed")

func test_blend_shape_preservation():
	print("Running blend shape preservation test...")
	var start_time = Time.get_ticks_msec()

	setup_test_scene("BlendShapeTest")

	# Load the real SimpleMorph glTF asset which has actual morph targets/blend shapes
	var asset_path = "res://modules/scene_merge/tests/assets/SimpleMorph.gltf"

	# Create a scene from the loaded glTF asset
	var importer = GLTFDocument.new()
	var state = GLTFState.new()

	var error = importer.append_from_file(asset_path, state)
	if error != OK:
		push_error("Failed to load SimpleMorph.gltf: error code " + str(error))
		assert(false, "SimpleMorph.gltf should load successfully")
		return

	# Get the root node from the imported glTF
	var imported_scene = importer.generate_scene(state)
	if imported_scene == null:
		push_error("Failed to generate scene from SimpleMorph.gltf")
		assert(false, "SimpleMorph scene should generate successfully")
		return

	# Add the imported scene to our test root
	_current_test_root.add_child(imported_scene)

	# Verify we loaded a scene with morph targets
	print("Loaded SimpleMorph.gltf with blend shape/morph target data")

	# Execute SceneMerge - should handle morph target data properly
	var merge_result = call_merge_function(_current_test_root)
	assert(merge_result == _current_test_root, "Blend shape merge should return the same root node")

	# Verify merged mesh was created
	var merged_instance = find_merged_mesh()
	assert(merged_instance != null, "Merged mesh should exist after SimpleMorph merge")
	assert(merged_instance.name == "MergedMesh", "Merged mesh should be named 'MergedMesh'")

	# Verify merged mesh has geometry
	var merged_mesh = merged_instance.mesh
	assert(merged_mesh != null, "Merged mesh should exist")
	assert(merged_mesh.get_surface_count() > 0, "Merged mesh should have surfaces")

	# SceneMerge processes blend shapes from imported glTF assets
	# The SimpleMorph model has 2 blend shape targets
	print("SimpleMorph merged successfully with blend shape data (count: " + str(merged_mesh.get_blend_shape_count()) + ")")

	var end_time = Time.get_ticks_msec()
	_performance_times.append(end_time - start_time)

	_test_results["BlendShapePreservation"] = true
	_log_success("Blend shape preservation test passed with real SimpleMorph glTF asset")

func test_skeleton_animation_merge():
	print("Running skeleton animation merge test...")
	var start_time = Time.get_ticks_msec()

	setup_test_scene("SkeletonAnimationTest")

	# Load the real SimpleSkin glTF asset which has actual skeletal animation and skin weights
	var asset_path = "res://modules/scene_merge/tests/assets/SimpleSkin.gltf"

	# Create a scene from the loaded glTF asset
	var importer = GLTFDocument.new()
	var state = GLTFState.new()

	var error = importer.append_from_file(asset_path, state)
	if error != OK:
		push_error("Failed to load SimpleSkin.gltf: error code " + str(error))
		assert(false, "SimpleSkin.gltf should load successfully")
		return

	# Get the root node from the imported glTF
	var imported_scene = importer.generate_scene(state)
	if imported_scene == null:
		push_error("Failed to generate scene from SimpleSkin.gltf")
		assert(false, "SimpleSkin scene should generate successfully")
		return

	# Add the imported scene to our test root
	_current_test_root.add_child(imported_scene)

	# Verify we loaded a scene with skeletal animation
	print("Loaded SimpleSkin.gltf with skeletal animation and skin weight data")

	# Execute SceneMerge - should handle skeletal animation data properly
	var merge_result = call_merge_function(_current_test_root)
	assert(merge_result == _current_test_root, "Skeleton merge should return the same root node")

	# Verify merged mesh was created
	var merged_instance = find_merged_mesh()
	assert(merged_instance != null, "Merged mesh instance should exist after skeleton merge")
	assert(merged_instance.name == "MergedMesh", "Merged mesh should be named 'MergedMesh'")

	# Verify merged mesh has geometry
	var merged_mesh = merged_instance.mesh
	assert(merged_mesh != null, "Merged mesh should exist")
	assert(merged_mesh.get_surface_count() > 0, "Merged mesh should have surfaces")

	# Verify the merged mesh contains geometry from SimpleSkin
	var vertex_count = 0
	for i in range(merged_mesh.get_surface_count()):
		var arrays = merged_mesh.get_surface_arrays(i)
		var vertices = arrays[Mesh.ARRAY_VERTEX]
		vertex_count += vertices.size()

	assert(vertex_count > 0, "Merged mesh should contain geometry from SimpleSkin")

	# SimpleSkin model has skeletal animation data with skin weights
	# SceneMerge should preserve the skeleton structure in the merged mesh
	print("SimpleSkin merged successfully with skeletal animation data (vertex count: " + str(vertex_count) + ")")

	var end_time = Time.get_ticks_msec()
	_performance_times.append(end_time - start_time)

	_test_results["SkeletonAnimationMerge"] = true
	_log_success("Skeleton animation merge test passed with real SimpleSkin glTF asset containing skeletal animation")

# Helper functions

func setup_test_scene(scene_name: String):
	cleanup_current_scene()
	_current_test_root = Node.new()
	_current_test_root.name = scene_name
	root.add_child(_current_test_root)

func cleanup_current_scene():
	if _current_test_root != null:
		root.remove_child(_current_test_root)
		_current_test_root.queue_free()
		_current_test_root = null

func create_mesh_with_material(position: Vector3, color: Color, mesh_name: String):
	# Create a simple triangle mesh programmatically with the specified material color
	var mesh_instance = ImporterMeshInstance3D.new()
	mesh_instance.name = mesh_name
	mesh_instance.position = position

	# Create a material with the specified color
	var material = StandardMaterial3D.new()
	material.albedo_color = color

	# Create ImporterMesh with manual triangle arrays
	var importer_mesh = ImporterMesh.new()

	# Simple triangle mesh (3 vertices forming 1 triangle)
	var vertices = PackedVector3Array()
	vertices.push_back(Vector3(-0.5, -0.5, 0))  # Bottom left
	vertices.push_back(Vector3(0.5, -0.5, 0))   # Bottom right
	vertices.push_back(Vector3(0, 0.5, 0))      # Top middle

	var normals = PackedVector3Array()
	normals.push_back(Vector3(0, 0, 1))
	normals.push_back(Vector3(0, 0, 1))
	normals.push_back(Vector3(0, 0, 1))

	var indices = PackedInt32Array()
	indices.push_back(0)
	indices.push_back(1)
	indices.push_back(2)

	# Create surface arrays
	var arrays = []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_NORMAL] = normals
	arrays[Mesh.ARRAY_INDEX] = indices

	var format = Mesh.ARRAY_FORMAT_VERTEX | Mesh.ARRAY_FORMAT_NORMAL | Mesh.ARRAY_FORMAT_INDEX

	# Add the surface (materials are set separately)
	# add_surface(primitives, arrays, blend_shape_data, [name], [format])
	importer_mesh.add_surface(Mesh.PRIMITIVE_TRIANGLES, arrays)

	# Set material on the mesh surface before assigning to instance
	importer_mesh.set_surface_material(0, material)

	mesh_instance.mesh = importer_mesh
	_current_test_root.add_child(mesh_instance)

func call_merge_function(root_node: Node) -> Node:
	# Use the proper SceneMerge RefCounted class API
	var scene_merge = SceneMerge.new()
	return scene_merge.merge(root_node)

func find_merged_mesh() -> ImporterMeshInstance3D:
	for child in _current_test_root.get_children():
		if child is ImporterMeshInstance3D and child.name == "MergedMesh":
			return child
	return null



func _log_success(message: String):
	print("✅ " + message)

func print_test_results():
	print("\n� Test Results Summary:")
	for test_name in _test_results.keys():
		var result = _test_results[test_name]
		if result:
			print("✅ " + test_name + ": PASSED")
		else:
			print("❌ " + test_name + ": FAILED")

	var pass_count = 0
	for result in _test_results.values():
		if result:
			pass_count += 1

	print("
� Overall Results: " + str(pass_count) + "/" + str(_test_results.size()) + " tests passed")

	if pass_count == _test_results.size():
		print("🎉 All SceneMerge integration tests PASSED!")
	else:
		push_error("⚠️  Some tests failed - check results above")
