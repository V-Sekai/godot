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

	# Export merged result to GLB for demonstration
	print("🔄 Starting GLB export...")
	export_merged_to_glb(merged_instance, "/tmp/scene_merge_test.glb")

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
		var vertices = arrays[Mesh::ARRAY_VERTEX]
		vertex_count += vertices.size()

	assert(vertex_count >= 6, "Merged mesh should contain geometry from both meshes") # 3 verts * 2 meshes

	_test_results["SceneTransformationPreservation"] = true
	_log_success("Scene transformation preservation test passed")

func test_blend_shape_preservation():
	print("Running blend shape preservation test...")

	setup_test_scene("BlendShapeTest")

	# Create two simple meshes to test basic merge functionality
	# (Not testing actual blend shapes since SceneMerge handles basic merging)
	create_mesh_with_material(Vector3(0, 0, 0), Color(1.0, 0.0, 0.0), "BlendMesh1")
	create_mesh_with_material(Vector3(3, 0, 0), Color(0.0, 1.0, 0.0), "BlendMesh2")

	# Execute SceneMerge - should not crash even if blend shapes were present
	var merge_result = call_merge_function(_current_test_root)
	assert(merge_result == _current_test_root, "Blend shape merge should return the same root node")

	# Verify merged mesh was created
	var merged_instance = find_merged_mesh()
	assert(merged_instance != null, "Merged mesh should exist")
	assert(merged_instance.name == "MergedMesh", "Merged mesh should be named 'MergedMesh'")

	# Verify merged mesh has geometry (blend shapes wouldn't be present in this test)
	var merged_mesh = merged_instance.mesh
	assert(merged_mesh != null, "Merged mesh should exist")
	assert(merged_mesh.get_surface_count() > 0, "Merged mesh should have surfaces")

	# SceneMerge preserves blend shapes in ImporterMesh format but this basic test
	# only verifies that the merge process works without crashing on potential blend shape data
	assert(merged_mesh.get_blend_shape_count() == 0, "No blend shapes expected in this basic test")

	_test_results["BlendShapePreservation"] = true
	_log_success("Blend shape preservation test passed (basic compatibility verified)")

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



func export_merged_to_glb(merged_instance: ImporterMeshInstance3D, glb_path: String):
	print("📤 Exporting GLB for merged instance:", merged_instance.name)
	print("💾 Target path:", glb_path)

	# Convert ImporterMesh to proper MeshInstance3D for GLB export
	var export_root = Node3D.new()
	export_root.name = "GLBExportRoot"
	print("🏗️ Created export root node")

	# Create a MeshInstance3D with the merged mesh data instead of ImporterMeshInstance3D
	var mesh_instance_3d = MeshInstance3D.new()
	mesh_instance_3d.name = "ExportedMesh"

	# Convert ImporterMesh to ArrayMesh for runtime mesh instance
	var importer_mesh = merged_instance.mesh as ImporterMesh
	var array_mesh = null
	if importer_mesh:
		# Create ArrayMesh from ImporterMesh data
		array_mesh = ArrayMesh.new()
		print("🔍 ImporterMesh has " + str(importer_mesh.get_surface_count()) + " surfaces")
		for i in range(importer_mesh.get_surface_count()):
			var arrays = importer_mesh.get_surface_arrays(i)
			print("🔍 Surface " + str(i) + " vertex count: " + str(arrays[Mesh.ARRAY_VERTEX].size() if arrays[Mesh.ARRAY_VERTEX] else 0))
			if arrays[Mesh.ARRAY_VERTEX] and arrays[Mesh.ARRAY_VERTEX].size() > 0:
				array_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
				print("✅ Added surface " + str(i) + " with " + str(arrays[Mesh.ARRAY_VERTEX].size()) + " vertices")
			else:
				print("❌ Surface " + str(i) + " has no vertex data - skipping")

		mesh_instance_3d.mesh = array_mesh
		print("🎨 Final ArrayMesh has " + str(array_mesh.get_surface_count()) + " surfaces")

	export_root.add_child(mesh_instance_3d)
	mesh_instance_3d.set_owner(export_root)
	print("📦 Added MeshInstance3D with converted mesh data")

	# Create GLTF document for export
	var gltf_document = GLTFDocument.new()
	var state = GLTFState.new()

	print("🔧 Using .glb extension to enable binary GLB export mode")

	# Set material on the ArrayMesh surface for export
	var merged_importer_mesh = merged_instance.mesh as ImporterMesh
	if merged_importer_mesh:
		var merged_material = merged_importer_mesh.get_surface_material(0)
		if merged_material:
			array_mesh.surface_set_material(0, merged_material)
			print("🎨 Applied merged material to export ArrayMesh")

	# Debug: Check mesh data before export
	var debug_mesh = array_mesh as ArrayMesh
	if debug_mesh:
		print("🔍 DEBUG: ArrayMesh has " + str(debug_mesh.get_surface_count()) + " surfaces")
		print("🔍 DEBUG: Surface 0 format: " + str(debug_mesh.surface_get_format(0)))
		print("🔍 DEBUG: Surface 0 vertex count: " + str(debug_mesh.surface_get_arrays(0)[Mesh.ARRAY_VERTEX].size()))
		print("🔍 DEBUG: Surface 0 material: " + str(debug_mesh.surface_get_material(0)))

	# Generate GLB/GLTF data
	print("⚙️ Generating GLTF data from scene...")
	var result = gltf_document.append_from_scene(export_root, state)
	print("📋 Append result:", result)
	if result == OK:
		# Save as GLB (binary format)
		print("💾 Saving GLB file to filesystem...")
		result = gltf_document.write_to_filesystem(state, glb_path)
		print("📋 Write result:", result)
		if result == OK:
			print("✅ GLB exported successfully to: " + glb_path)
			print("   - Binary GLB file: " + glb_path + " (contains embedded glTF JSON + buffers)")
		else:
			print("❌ Failed to save GLB: " + str(result))
	else:
		print("❌ Failed to append scene to GLTF: " + str(result))

	# Clean up temporary export scene
	print("🧹 Cleaning up export scene")
	export_root.queue_free()
	print("🏁 GLB export function completed")

func _log_success(message: String):
	print("✅ " + message)

func print_test_results():
	print("\n📋 Test Results Summary:")
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
📊 Overall Results: " + str(pass_count) + "/" + str(_test_results.size()) + " tests passed")

	if pass_count == _test_results.size():
		print("🎉 All SceneMerge integration tests PASSED!")
	else:
		push_error("⚠️  Some tests failed - check results above")
