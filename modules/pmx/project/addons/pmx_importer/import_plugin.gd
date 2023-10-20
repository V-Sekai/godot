tool
extends EditorImportPlugin

var shader: Shader = preload("toon.shader")

enum Presets { DEFAULT }

func get_importer_name():
	return "PMX"

func get_visible_name():
	return "PMX Scene"

func get_recognized_extensions():
	return ["pmx"]

func get_save_extension():
	return "scn"

func get_resource_type():
	return "PackedScene"
	
func get_preset_count():
	return Presets.size()
	
func get_preset_name(preset):
	match preset:
		Presets.DEFAULT:
			return "Default"	
		_:
			return "Unknown"

func get_import_options(preset):
	match preset:
		_:
			return []
			
func make_material(m: Dictionary, textures: Array, source_dir: String, texture_cache: Dictionary) -> ShaderMaterial:
	var material := ShaderMaterial.new()
	material.resource_name = m.name_universal if m.name_universal else m.name_local
	material.shader = shader
	material.set_shader_param("albedo", m.diffuse)
	if m.texture >= 0:
		var tex = load_texture(textures[m.texture], source_dir, texture_cache)
		if tex:
			material.set_shader_param("albedo_texture", tex)
		else:
			print("Failed to load texture %s" % [textures[m.texture]])

	material.set_shader_param("specular_shininess", m.specularity)
	material.set_shader_param("rim_width", m.edge_scale)
	material.set_shader_param("rim_color", m.edge_color)
	return material
	
func load_texture(name: String, source_dir: String, cache: Dictionary):
	if cache.has(name):
		return cache[name]

	var filename = "%s/%s" % [source_dir, name.replace("\\", "/")]
	var tex = load(filename)
	cache[name] = tex
	return tex
	
func make_skeleton(bones: Array) -> Skeleton:
	var skel := Skeleton.new()
	for bone in bones:
		print("Bone %s (%s)" % [bone.name_local, bone.name_universal])
		var bone_idx = skel.get_bone_count()
		skel.add_bone("bone%d" % [bone_idx])
		if bone.parent >= 0:
			skel.set_bone_parent(bone_idx, bone.parent)
			
		var t := Transform.IDENTITY
		t.origin = bone.position
		skel.set_bone_rest(bone_idx, t)

	skel.localize_rests()
	return skel
	
func make_mesh(pmx, source_dir: String) -> ArrayMesh:
	var mesh := ArrayMesh.new()
	var arrays := []
	arrays.resize(ArrayMesh.ARRAY_MAX)
	print('Got ', pmx.positions.size(), " positions")
	arrays[ArrayMesh.ARRAY_VERTEX] = pmx.positions
	arrays[ArrayMesh.ARRAY_NORMAL] = pmx.normals
	arrays[ArrayMesh.ARRAY_TEX_UV] = pmx.uvs
	arrays[ArrayMesh.ARRAY_BONES] = pmx.bone_indices
	arrays[ArrayMesh.ARRAY_WEIGHTS] = pmx.bone_weights
	var triangles: PoolIntArray = pmx.triangles
	var texture_cache := {}
	for m in pmx.materials:
		var name = m.name_universal if m.name_universal else m.name_local
		print("Material: ", name)
		arrays[ArrayMesh.ARRAY_INDEX] = m.indices
		var surf_idx = mesh.get_surface_count() 
		mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
		mesh.surface_set_name(surf_idx, name)
		var material := make_material(m, pmx.textures, source_dir, texture_cache)
		mesh.surface_set_material(surf_idx, material)
		
	return mesh

func import(source_file: String, save_path: String, options, r_platform_variants, r_gen_files):
	var pmx = preload("PMX.gdns").new()
	var source_dir := source_file.rsplit("/", true, 1)[0]
	var file := File.new();
	var err = file.open(source_file, File.READ)
	if err != OK:
		return err
	var data := file.get_buffer(file.get_len())
	var ret = pmx.parse(data)
	var scene := Node.new()
	scene.name = pmx.model_name_universal if pmx.model_name_universal else pmx.model_name_local
	var skeleton := make_skeleton(pmx.bones)
	scene.add_child(skeleton)
	skeleton.set_owner(scene)
	var mesh_instance := MeshInstance.new()
	mesh_instance.mesh = make_mesh(pmx, source_dir)
	skeleton.add_child(mesh_instance)
	mesh_instance.set_owner(scene)
	var packed_scene := PackedScene.new()
	packed_scene.pack(scene)
	ret = ResourceSaver.save("%s.%s" % [save_path, get_save_extension()], packed_scene, ResourceSaver.FLAG_COMPRESS)
	scene.free()
	return ret
