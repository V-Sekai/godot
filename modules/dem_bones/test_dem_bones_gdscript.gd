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
		quit(1)
		return

	print("Generating scene...")
	var scene_root = gltf_document.generate_scene(gltf_state)
	if scene_root == null:
		print("Failed to generate scene")
		quit(1)
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
		quit(1)
		return

	if mesh_instance == null:
		print("No ImporterMeshInstance3D found")
		quit(1)
		return

	print("Checking mesh blend shapes...")
	var mesh = mesh_instance.mesh
	if mesh == null or mesh.get_blend_shape_count() == 0:
		print("Mesh has no blend shapes")
		quit(1)
		return

	print("Found mesh with ", mesh.get_blend_shape_count(), " blend shapes")

	print("Getting animation list...")
	var animation_names = anim_player.get_animation_list()
	if animation_names.is_empty():
		print("No animations found")
		quit(1)
		return

	var animation_name = animation_names[0]
	print("Processing animation: ", animation_name)

	print("Creating DemBonesProcessor...")
	var processor = DemBonesProcessor.new()
	print("Starting process_animation...")
	err = processor.process_animation(anim_player, mesh_instance, animation_name)
	if err != OK:
		print("Failed to process animation: ", err)
		quit(1)
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

	# Clean up
	print("Cleaning up...")
	scene_root.queue_free()
	print("Test completed successfully!")
	quit(0)