extends SceneTree

func _init():
	print("Starting DemBones test...")

	# Load geometry file (Bone_Geom.glb)
	print("Loading geometry GLTF...")
	var geom_gltf_path = "res://modules/dem_bones/data/Bone_Geom.glb"
	var geom_gltf_doc = GLTFDocument.new()
	var geom_gltf_state = GLTFState.new()
	var err = geom_gltf_doc.append_from_file(geom_gltf_path, geom_gltf_state)
	if err != OK:
		print("Failed to load geometry GLTF: ", err)
		return
	var geom_scene = geom_gltf_doc.generate_scene(geom_gltf_state)
	if geom_scene == null:
		print("Failed to generate geometry scene")
		return

	# Load animation file (Bone_Anim.glb) to get blend shapes
	print("Loading animation GLTF...")
	var anim_gltf_path = "res://modules/dem_bones/data/Bone_Anim.glb"
	var anim_gltf_doc = GLTFDocument.new()
	var anim_gltf_state = GLTFState.new()
	err = anim_gltf_doc.append_from_file(anim_gltf_path, anim_gltf_state)
	if err != OK:
		print("Failed to load animation GLTF: ", err)
		print("Trying experimental blend shape file...")
		anim_gltf_path = "res://modules/dem_bones/data/Bone_Anim_Blender5.glb"
		err = anim_gltf_doc.append_from_file(anim_gltf_path, anim_gltf_state)
		if err != OK:
			print("Failed to load experimental animation GLTF: ", err)
			return

	var anim_scene = anim_gltf_doc.generate_scene(anim_gltf_state)
	if anim_scene == null:
		print("Failed to generate animation scene - using geom_scene as root")
		anim_scene = geom_scene

	# Use geometry scene as the main scene
	var scene_root = geom_scene
	print("Using merged scene...")

	print("Finding ImporterMeshInstance3D with blend shapes...")
	var mesh_instance = null

	# Find ImporterMeshInstance3D
	var nodes_to_check = [scene_root]
	while not nodes_to_check.is_empty():
		var current = nodes_to_check.pop_front()
		if current is ImporterMeshInstance3D and mesh_instance == null:
			mesh_instance = current

		for child in current.get_children():
			nodes_to_check.append(child)

	if mesh_instance == null:
		print("No ImporterMeshInstance3D found")
		return

	print("Checking mesh blend shapes...")
	var mesh = mesh_instance.mesh
	if mesh == null or mesh.get_blend_shape_count() == 0:
		print("Mesh has no blend shapes - checking geom scene instead...")

		# Try geometry scene instead
		nodes_to_check = [geom_scene]
		while not nodes_to_check.is_empty():
			var current = nodes_to_check.pop_front()
			if current is ImporterMeshInstance3D and mesh_instance == null:
				mesh_instance = current

			for child in current.get_children():
				nodes_to_check.append(child)

		if mesh_instance == null:
			print("Still no ImporterMeshInstance3D found")
			return

		mesh = mesh_instance.mesh
		if mesh == null or mesh.get_blend_shape_count() == 0:
			print("No blend shapes found in any scene")
			return

	print("Found mesh with ", mesh.get_blend_shape_count(), " blend shapes")

	# Create synthetic AnimationPlayer with blend shape animations
	print("Creating synthetic AnimationPlayer with blend shape animations...")
	var anim_player = AnimationPlayer.new()
	anim_player.name = "BlendShapeAnimPlayer"
	mesh_instance.add_child(anim_player)

	var animation = Animation.new()
	animation.length = 10.0  # 10 seconds

	# Create tracks for each blend shape
	var blend_shape_count = mesh.get_blend_shape_count()
	print("Creating " + str(blend_shape_count) + " blend shape animation tracks...")

	for i in range(blend_shape_count):
		var track_path = ".:blend_shapes/" + mesh.get_blend_shape_name(i)
		var track_idx = animation.add_track(Animation.TYPE_BLEND_SHAPE)
		animation.track_set_path(track_idx, track_path)

		# Create keyframes - each blend shape activates briefly in sequence
		var start_time = float(i) * (animation.length / float(blend_shape_count))
		var peak_time = start_time + 0.5
		var end_time = start_time + 1.0

		# Start: 0
		animation.track_insert_key(track_idx, start_time, 0.0)

		# Peak: 1.0
		animation.track_insert_key(track_idx, peak_time, 1.0)

		# End: 0
		animation.track_insert_key(track_idx, end_time, 0.0)

		print("  Created track for: " + mesh.get_blend_shape_name(i))

	anim_player.add_animation("blend_shapes", animation)

	var animation_names = anim_player.get_animation_list()
	if animation_names.is_empty():
		print("Failed to create animation")
		return

	var animation_name = animation_names[0]
	print("Processing animation: ", animation_name)

	print("Creating DemBonesProcessor...")
	var processor = DemBonesProcessor.new()
	print("Starting process_animation...")
	err = processor.process_animation(anim_player, mesh_instance, animation_name, -1)
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

	# Set bone poses from transforms (use first frame as rest pose)
	if bone_transforms.size() > 0:
		for i in range(bone_count):
			var transform = bone_transforms[0][i] as Transform3D  # Access first frame, then bone index
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
			skin.set_bind_pose(i, bone_transforms[0][i] as Transform3D)

		# Set skinning weights - proper multi-bone skinning (up to 8 bones per vertex)
		const MAX_BONES_PER_VERTEX = 8
		const WEIGHT_THRESHOLD = 0.001

		# For each vertex, collect bone influences
		var vertex_bone_influences = []
		vertex_bone_influences.resize(rest_vertices.size())

		for vertex_idx in range(rest_vertices.size()):
			var influences = []

			# Collect all bone influences for this vertex above threshold
			for bone_idx in range(bone_count):
				var bone_weights = skinning_weights[bone_idx] as Array
				if vertex_idx < bone_weights.size():
					var bone_weight = bone_weights[vertex_idx] as float
					if bone_weight > WEIGHT_THRESHOLD:
						influences.append({"bone_idx": bone_idx, "weight": bone_weight})

			# Sort by weight descending
			influences.sort_custom(func(a, b): return a["weight"] > b["weight"])

			# Take up to MAX_BONES_PER_VERTEX bones
			if influences.size() > MAX_BONES_PER_VERTEX:
				influences.resize(MAX_BONES_PER_VERTEX)

			# Normalize weights
			var total_weight = 0.0
			for influence in influences:
				total_weight += influence["weight"]
			if total_weight > 0.0:
				for influence in influences:
					influence["weight"] /= total_weight

			vertex_bone_influences[vertex_idx] = influences

		# Build bone_indices and weights arrays (Godot format)
		var max_influences = 0
		for influences in vertex_bone_influences:
			max_influences = max(max_influences, influences.size())

		bone_indices = PackedInt32Array()
		weights = PackedFloat32Array()

		# Each vertex gets exactly max_influences bones (padding with 0, 0.0 for unused slots)
		for vertex_influences in vertex_bone_influences:
			var influences_count = vertex_influences.size()
			for i in range(max_influences):
				if i < influences_count:
					bone_indices.append(vertex_influences[i]["bone_idx"])
					weights.append(vertex_influences[i]["weight"])
				else:
					bone_indices.append(0)  # Padding
					weights.append(0.0)    # Padding

		print("Skinning setup: " + str(rest_vertices.size()) + " vertices with up to " + str(min(MAX_BONES_PER_VERTEX, max_influences)) + " bones per vertex")

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
