extends SceneTree

func _init():
	print("=== Inspecting GLTF files in modules/dem_bones/data_gltf/ ===\n")

	var gltf_document = GLTFDocument.new()
	var gltf_state = GLTFState.new()

	# Inspect Bone_Anim.glb
	print("--- Bone_Anim.glb (vertex cache/animation) ---")
	var anim_err = gltf_document.append_from_file("res://modules/dem_bones/data_gltf/Bone_Anim.glb", gltf_state)
	if anim_err == OK:
		var anim_scene = gltf_document.generate_scene(gltf_state)
		inspect_scene(anim_scene, "Bone_Anim")
		anim_scene.free()
	else:
		print("Failed to load Bone_Anim.glb: ", anim_err)

	print("")

	# Reset state for next file
	gltf_state = GLTFState.new()

	# Inspect Bone_Geom.glb
	print("--- Bone_Geom.glb (geometry) ---")
	var geom_err = gltf_document.append_from_file("res://modules/dem_bones/data_gltf/Bone_Geom.glb", gltf_state)
	if geom_err == OK:
		var geom_scene = gltf_document.generate_scene(gltf_state)
		inspect_scene(geom_scene, "Bone_Geom")
		geom_scene.free()
	else:
		print("Failed to load Bone_Geom.glb: ", geom_err)

	print("\n=== Analysis Complete ===")

func inspect_scene(scene: Node, name: String):
	var meshes = []
	var anim_players = []
	var skeletons = []

	# Find all relevant nodes
	find_nodes_by_type(scene, meshes, anim_players, skeletons)

	print("Scene root: ", scene.name)
	print("Total child nodes: ", count_all_children(scene))

	if meshes.size() > 0:
		print("Meshes found: ", meshes.size())
		for i in meshes.size():
			var mesh_instance = meshes[i]
			var mesh = mesh_instance.mesh
			print("  Mesh ", i, ": ", mesh.resource_name if mesh else "null")
			if mesh:
				print("    Blend shapes: ", mesh.get_blend_shape_count())
				print("    Surfaces: ", mesh.get_surface_count())
				if mesh.get_surface_count() > 0:
					# Handle both Mesh and ImporterMesh
					var vertex_count = 0
					if mesh is Mesh:
						var surface_arrays = mesh.surface_get_arrays(0)
						if surface_arrays.size() > 0:
							vertex_count = surface_arrays[0].size() # Array[Mesh.ARRAY_VERTEX]
					elif mesh is ImporterMesh:
						# ImporterMesh needs to get surface arrays differently
						var surface_arrays = mesh.get_surface_arrays(0)
						if surface_arrays.size() > 0:
							vertex_count = surface_arrays[0].size()
					print("    Vertices: ", vertex_count)
	else:
		print("No meshes found")

	if anim_players.size() > 0:
		print("AnimationPlayers found: ", anim_players.size())
		for i in anim_players.size():
			var ap = anim_players[i]
			print("  AnimationPlayer ", i, ": ", ap.name)
			var animation_names = ap.get_animation_list()
			print("    Animations: ", animation_names.size())
			if animation_names.size() > 0:
				for anim_name in animation_names:
					var anim = ap.get_animation(anim_name)
					print("      Animation '", anim_name, "': ", anim.get_length(), "s, ", anim.get_track_count(), " tracks")
					# Show track names
					for t in min(anim.get_track_count(), 5):  # Show first 5 tracks
						var track_path = String(anim.track_get_path(t))
						print("        Track ", t, ": ", track_path)
					if anim.get_track_count() > 5:
						print("        ... and ", anim.get_track_count() - 5, " more tracks")
	else:
		print("No AnimationPlayers found")

	if skeletons.size() > 0:
		print("Skeletons found: ", skeletons.size())
		for i in skeletons.size():
			var skeleton = skeletons[i]
			print("  Skeleton ", i, ": ", skeleton.name)
			print("    Bones: ", skeleton.get_bone_count())
			# List first few bone names
			for b in min(skeleton.get_bone_count(), 5):
				print("      Bone ", b, ": ", skeleton.get_bone_name(b))
			if skeleton.get_bone_count() > 5:
				print("      ... and ", skeleton.get_bone_count() - 5, " more bones")
	else:
		print("No skeletons found")

func find_nodes_by_type(node: Node, meshes: Array, anim_players: Array, skeletons: Array):
	if node is MeshInstance3D or node is ImporterMeshInstance3D:
		meshes.append(node)
	if node is AnimationPlayer:
		anim_players.append(node)
	if node is Skeleton3D:
		skeletons.append(node)

	for child in node.get_children():
		find_nodes_by_type(child, meshes, anim_players, skeletons)

func count_all_children(node: Node) -> int:
	var count = node.get_child_count()
	for child in node.get_children():
		count += count_all_children(child)
	return count
