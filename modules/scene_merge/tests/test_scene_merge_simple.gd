#!/usr/bin/env godot --script
extends SceneTree

# Simple SceneMerge Test - Tests basic functionality
# Focuses on core merge operations without complex GLB/blend shape stuff

func _init():
	print("🧪 Running Simple SceneMerge Test...")
	test_scene_merge()
	print("🏁 Test completed!")

func test_scene_merge():
	print("📥 Creating test scene with meshes...")

	# Create a root node for the scene
	var test_root = Node.new()
	test_root.name = "TestRoot"
	root.add_child(test_root)

	# Create two mesh instances
	var mesh1 = create_simple_mesh(Vector3(0, 0, 0), Color(1.0, 0.0, 0.0), "Mesh1")
	var mesh2 = create_simple_mesh(Vector3(3, 0, 0), Color(0.0, 1.0, 0.0), "Mesh2")

	test_root.add_child(mesh1)
	test_root.add_child(mesh2)

	print("📊 Initial scene has " + str(test_root.get_child_count()) + " children")

	print("🔄 Running SceneMerge...")
	var scene_merge = SceneMerge.new()
	var result = scene_merge.merge(test_root)

	if result == test_root:
		print("✅ Merge completed successfully!")

		# Check the result
		var children = test_root.get_children()
		print("📊 Scene now has " + str(children.size()) + " children")

		for child in children:
			if child is ImporterMeshInstance3D and child.name.begins_with("Merged"):
				var mesh = child.mesh
				print("🎯 Found merged mesh: " + child.name)
				if mesh:
					print("   - Mesh has " + str(mesh.get_surface_count()) + " surfaces")
					print("   - Vertex count: " + str(count_vertices(mesh)))
				break

		print("🎉 SceneMerge test PASSED!")
	else:
		print("❌ Merge failed")

func create_simple_mesh(position: Vector3, color: Color, name: String) -> ImporterMeshInstance3D:
	var mesh_instance = ImporterMeshInstance3D.new()
	mesh_instance.name = name
	mesh_instance.position = position

	var material = StandardMaterial3D.new()
	material.albedo_color = color

	var importer_mesh = ImporterMesh.new()
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

	importer_mesh.add_surface(Mesh.PRIMITIVE_TRIANGLES, arrays)
	importer_mesh.set_surface_material(0, material)

	mesh_instance.mesh = importer_mesh
	return mesh_instance

func count_vertices(mesh: ImporterMesh) -> int:
	var count = 0
	for i in range(mesh.get_surface_count()):
		var arrays = mesh.get_surface_arrays(i)
		var vertices = arrays[Mesh.ARRAY_VERTEX]
		if vertices:
			count += vertices.size()
	return count
