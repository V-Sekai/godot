extends SceneTree

func _init():
	print("Testing GLB export...")
	var gltf_doc_out = GLTFDocument.new()
	var gltf_state_out = GLTFState.new()

	# Create a simple scene with one mesh for testing
	var output_scene = Node3D.new()
	output_scene.name = "Test_Export"

	var mesh_instance = MeshInstance3D.new()
	mesh_instance.name = "Mesh"
	var new_mesh = ArrayMesh.new()
	new_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, [])
	mesh_instance.mesh = new_mesh
	output_scene.add_child(mesh_instance)

	var export_path = "res://test_export.glb"
	var gltf_doc = GLTFDocument.new()
	var gltf_state = GLTFState.new()

	var err = gltf_doc.append_from_scene(output_scene, gltf_state)
	if err != OK:
		print("Failed to append scene to GLTF: ", err)
		return

	err = gltf_doc.write_to_file(export_path, gltf_state)
	if err != OK:
		print("Failed to write GLB file: ", err)
		return

	print("Successfully exported to: ", export_path)
	print("GLB export test completed!")
	quit()
