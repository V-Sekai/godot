extends SceneTree

func _init():
	print("Starting DemBones test...")
	var gltf_path = "res://modules/dem_bones/data_gltf/Bone_Anim.glb"
	print("Loading GLTF from: ", gltf_path)
	var gltf_document = GLTFDocument.new()
	var gltf_state = GLTFState.new()

	var err = gltf_document.append_from_file(gltf_path, gltf_state)
	if err != OK:
		print("Failed to load GLTF: ", err)
		return

	print("Generating scene...")
	var scene_root = gltf_document.generate_scene(gltf_state)
	if scene_root == null:
		print("Failed to generate scene")
		return

	print("Finding AnimationPlayer and ImporterMeshInstance3D...")
	# Find AnimationPlayer and ImporterMeshInstance3D
	var anim_player = null
	var mesh_instance = null

	var nodes_to_check = [scene_root]
	while not nodes_to_check.is_empty():
		var current = nodes_to_check.pop_front()
		if current is AnimationPlayer and anim_player == null:
			anim_player = current
		elif current is ImporterMeshInstance3D and mesh_instance == null:
			mesh_instance = current

		for child in current.get_children():
			nodes_to_check.append(child)

	if anim_player == null:
		print("No AnimationPlayer found")
		return

	if mesh_instance == null:
		print("No ImporterMeshInstance3D found")
		return

	print("Checking mesh blend shapes...")
	var mesh = mesh_instance.mesh
	if mesh == null or mesh.get_blend_shape_count() == 0:
		print("Mesh has no blend shapes")
		return

	print("Found mesh with ", mesh.get_blend_shape_count(), " blend shapes")

	print("Getting animation list...")
	var animation_names = anim_player.get_animation_list()
	if animation_names.is_empty():
		print("No animations found")
		return

	var animation_name = animation_names[0]
	print("Processing animation: ", animation_name)

	print("Creating DemBonesProcessor...")
	var processor = DemBonesProcessor.new()
	print("Starting process_animation...")
	err = processor.process_animation(anim_player, mesh_instance, animation_name)
	if err != OK:
		print("Failed to process animation: ", err)
		return

	print("Processing completed successfully!")
	var rest_vertices = processor.get_rest_vertices()
	var skinning_weights = processor.get_skinning_weights()
	var bone_transforms = processor.get_bone_transforms()
	var bone_count = processor.get_bone_count()

	print("Results:")
	print("  Rest vertices: ", rest_vertices.size())
	print("  Skinning weights: ", skinning_weights.size())
	print("  Bone transforms: ", bone_transforms.size())
	print("  Bone count: ", bone_count)

	print("Creating output scene with skeleton...")

	# Create a new scene root for export
	var output_scene = Node3D.new()
	output_scene.name = "DemBones_Output"

	# Create skeleton
	var skeleton = Skeleton3D.new()
	skeleton.name = "Skeleton"
	output_scene.add_child(skeleton)

	# Add bones to skeleton
	for i in range(bone_count):
		var bone_name = "Bone_" + str(i)
		skeleton.add_bone(bone_name)
		if i > 0:
			skeleton.set_bone_parent(i, 0)  # All bones parented to root for simplicity

	# Set bone poses from transforms
	for i in range(bone_count):
		var transform = bone_transforms[i]
		skeleton.set_bone_pose_position(i, transform.origin)
		skeleton.set_bone_pose_rotation(i, transform.basis.get_rotation_quaternion())
		skeleton.set_bone_pose_scale(i, transform.basis.get_scale())

	# Create new mesh with rest vertices
	var original_mesh = mesh_instance.mesh
	var new_mesh = ArrayMesh.new()

	# Get surface arrays from original mesh (surface 0)
	var arrays = original_mesh.surface_get_arrays(0)
	if arrays.size() > 0:
		# Replace vertex positions with rest vertices
		arrays[Mesh.ARRAY_VERTEX] = rest_vertices

		# Create new surface
		new_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)

		# Copy blend shapes if they exist
		for bs_idx in range(original_mesh.get_blend_shape_count()):
			var bs_arrays = original_mesh.surface_get_blend_shape_arrays(0, bs_idx)
			if bs_arrays.size() > 0:
				new_mesh.add_blend_shape(original_mesh.get_blend_shape_name(bs_idx))
				new_mesh.surface_set_blend_shape_arrays(0, bs_idx, bs_arrays)

	# Create mesh instance
	var mesh_instance_out = MeshInstance3D.new()
	mesh_instance_out.name = "Mesh"
	mesh_instance_out.mesh = new_mesh
	output_scene.add_child(mesh_instance_out)

	# Create skin and set up skinning
	var weights = PackedFloat32Array()
	var bone_indices = PackedInt32Array()

	if skinning_weights.size() > 0:
		var skin = Skin.new()
		skin.set_bind_count(bone_count)

		# Set bone names and transforms
		for i in range(bone_count):
			var bone_name = "Bone_" + str(i)
			skin.set_bind_name(i, bone_name)
			skin.set_bind_pose(i, bone_transforms[i])

		# Set skinning weights
		for vertex_weights in skinning_weights:
			for weight_data in vertex_weights:
				bone_indices.append(weight_data.bone_index)
				weights.append(weight_data.weight)

		mesh_instance_out.skin = skin
		new_mesh.surface_set_skin(0, skin)

	print("Setting up skinning data...")
	# Set the skinning data on the mesh surface
	var skin_arrays = []
	skin_arrays.resize(Mesh.ARRAY_MAX)
	skin_arrays[Mesh.ARRAY_BONES] = bone_indices
	skin_arrays[Mesh.ARRAY_WEIGHTS] = weights

	# Update the surface with skinning data
	new_mesh.surface_update_region(0, 0, new_mesh.surface_get_array_len(0))
	new_mesh.surface_set_arrays(0, arrays, skin_arrays)

	print("Exporting to GLB...")
	# Export to GLB
	var gltf_doc_out = GLTFDocument.new()
	var gltf_state_out = GLTFState.new()

	var export_path = "res://dem_bones_output.glb"
	err = gltf_doc_out.append_from_scene(output_scene, gltf_state_out)
	if err != OK:
		print("Failed to append scene to GLTF: ", err)
	else:
		err = gltf_doc_out.write_to_file(export_path, gltf_state_out)
		if err != OK:
			print("Failed to write GLB file: ", err)
		else:
			print("Successfully exported to: ", export_path)

	# Clean up
	print("Cleaning up...")
	scene_root.queue_free()
	output_scene.queue_free()
	print("Test completed successfully!")
