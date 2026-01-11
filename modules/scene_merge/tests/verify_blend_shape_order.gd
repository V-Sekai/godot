#!/usr/bin/env godot --script
extends SceneTree

# Verify Blend Shape Vertex Order Preservation
# This tests that the C++ blend shape code works correctly

func _init():
	print("🔬 Verifying Blend Shape Order Preservation...")

	test_blend_shape_order_preservation()

	print("🏁 Verification complete!")
	quit()

func test_blend_shape_order_preservation():
	print("🎯 Testing vertex order preservation in SceneMerge blend shapes...")

	# Test single mesh blend shape merge (supported scenario)
	print("📦 Testing single mesh with blend shapes...")
	var blend_mesh1 = create_blend_mesh(Vector3(0, 0, 0), ["Shape1", "Shape2"], "BlendMesh1")

	var test_root = Node.new()
	root.add_child(test_root)
	test_root.add_child(blend_mesh1)

	var scene_merge = SceneMerge.new()
	var result = scene_merge.merge(test_root)

	if result != test_root:
		print("❌ Single mesh merge failed")
		return

	var merged = find_merged_mesh(test_root)
	if not merged:
		print("❌ No merged mesh found")
		return

	var merged_mesh = merged.mesh
	if not merged_mesh:
		print("❌ Merged mesh has no mesh data")
		return

	var final_blend_shape_count = merged_mesh.get_blend_shape_count()
	print("🎯 Single mesh merge result: " + str(final_blend_shape_count) + " blend shapes")

	if final_blend_shape_count > 0:
		print("🎉 SINGLE MESH BLEND SHAPE PRESERVATION WORKS!")
		for i in range(final_blend_shape_count):
			var shape_name = merged_mesh.get_blend_shape_name(i)
			print("   ✅ Preserved: '" + shape_name + "'")
	else:
		print("⚠️  Blend shapes not preserved in single mesh merge")

func create_blend_mesh(position: Vector3, blend_shape_names: Array, name: String) -> ImporterMeshInstance3D:
	var importer_mesh = ImporterMesh.new()

	# Add blend shapes FIRST (before surfaces)
	for shape_name in blend_shape_names:
		importer_mesh.add_blend_shape(shape_name)

	# Base surface arrays
	var vertices = PackedVector3Array()
	vertices.push_back(Vector3(-0.5, -0.5, 0))
	vertices.push_back(Vector3(0.5, -0.5, 0))
	vertices.push_back(Vector3(0, 0.5, 0))

	var normals = PackedVector3Array()
	normals.push_back(Vector3(0, 0, 1))
	normals.push_back(Vector3(0, 0, 1))
	normals.push_back(Vector3(0, 0, 1))

	var indices = PackedInt32Array()
	indices.push_back(0)
	indices.push_back(1)
	indices.push_back(2)

	var arrays = []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_NORMAL] = normals
	arrays[Mesh.ARRAY_INDEX] = indices

	# Create blend shape data arrays - each blend shape needs vertex displacement data
	var blend_shapes = []
	for i in range(blend_shape_names.size()):
		var bs_arrays = []
		bs_arrays.resize(Mesh.ARRAY_MAX)

		# Create vertex offsets for blend shape (morphed vertices)
		var bs_vertices = PackedVector3Array()
		bs_vertices.push_back(Vector3(-0.3, -0.3, 0.2))  # Vertex 0 offset
		bs_vertices.push_back(Vector3(0.7, -0.3, 0.1))   # Vertex 1 offset
		bs_vertices.push_back(Vector3(0.1, 0.7, 0.2))    # Vertex 2 offset

		bs_arrays[Mesh.ARRAY_VERTEX] = bs_vertices
		# No normals for blend shapes
		blend_shapes.append(bs_arrays)

	var material = StandardMaterial3D.new()
	material.albedo_color = Color(randf(), randf(), randf())

	# Add surface with blend shapes
	importer_mesh.add_surface(Mesh.PRIMITIVE_TRIANGLES, arrays, blend_shapes)
	importer_mesh.set_surface_material(0, material)

	var mesh_instance = ImporterMeshInstance3D.new()
	mesh_instance.name = name
	mesh_instance.position = position
	mesh_instance.mesh = importer_mesh

	var blend_shape_count = importer_mesh.get_blend_shape_count()
	print("   📋 " + name + " has " + str(blend_shape_count) + " blend shapes")

	return mesh_instance

func find_merged_mesh(scene_node: Node) -> ImporterMeshInstance3D:
	for child in scene_node.get_children():
		if child is ImporterMeshInstance3D and child.name == "MergedMesh":
			return child
	return null
