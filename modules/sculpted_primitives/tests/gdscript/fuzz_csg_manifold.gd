extends SceneTree

# Ad hoc fuzzing for non-manifold CSG meshes
func _init():
	print("[FUZZ] Starting CSG mesh fuzzing...")
	var mesh = ArrayMesh.new()
	# Example: Add degenerate triangle
	var arrays = []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = [Vector3(0,0,0), Vector3(0,0,0), Vector3(1,0,0)]
	arrays[Mesh.ARRAY_INDEX] = [0,1,2]
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	var result = CSGShape3D.validate_manifold_mesh(mesh)
	print("Degenerate triangle manifold status: ", result)
	# TODO: Add more random/edge-case geometry
	quit()
