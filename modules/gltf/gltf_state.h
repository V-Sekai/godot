/**************************************************************************/
/*  gltf_state.h                                                          */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef GLTF_STATE_H
#define GLTF_STATE_H

#include "core/error/error_macros.h"
#include "core/object/class_db.h"
#include "core/object/object_id.h"
#include "core/templates/hash_map.h"
#include "core/templates/template_convert.h"
#include "extensions/gltf_light.h"
#include "modules/gltf/gltf_defines.h"
#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/resources/material.h"
#include "scene/resources/texture.h"
#include "structures/gltf_accessor.h"
#include "structures/gltf_animation.h"
#include "structures/gltf_buffer_view.h"
#include "structures/gltf_camera.h"
#include "structures/gltf_mesh.h"
#include "structures/gltf_node.h"
#include "structures/gltf_skeleton.h"
#include "structures/gltf_skin.h"
#include "structures/gltf_texture.h"
#include "structures/gltf_texture_sampler.h"

class GLTFState : public Resource {
	GDCLASS(GLTFState, Resource);
	friend class GLTFDocument;

	String base_path;
	String filename;
	Dictionary json;
	int major_version = 0;
	int minor_version = 0;
	String copyright;
	Vector<uint8_t> glb_data;

	bool use_named_skin_binds = false;
	bool use_khr_texture_transform = false;
	bool discard_meshes_and_materials = false;
	bool force_generate_tangents = false;
	bool create_animations = true;
	bool force_disable_compression = false;

	int handle_binary_image = HANDLE_BINARY_EXTRACT_TEXTURES;

	Vector<Ref<GLTFNode>> nodes;
	Vector<Vector<uint8_t>> buffers;
	Vector<Ref<GLTFBufferView>> buffer_views;
	Vector<Ref<GLTFAccessor>> accessors;

	Vector<Ref<GLTFMesh>> meshes; // Meshes are loaded directly, no reason not to.

	Vector<AnimationPlayer *> animation_players;
	HashMap<Ref<Material>, GLTFMaterialIndex> material_cache;
	Vector<Ref<Material>> materials;

	String scene_name;
	Vector<int> root_nodes;
	Vector<Ref<GLTFTexture>> textures;
	Vector<Ref<GLTFTextureSampler>> texture_samplers;
	Ref<GLTFTextureSampler> default_texture_sampler;
	Vector<Ref<Texture2D>> images;
	Vector<String> extensions_used;
	Vector<String> extensions_required;
	Vector<Ref<Image>> source_images;

	Vector<Ref<GLTFSkin>> skins;
	Vector<Ref<GLTFCamera>> cameras;
	Vector<Ref<GLTFLight>> lights;
	HashSet<String> unique_names;
	HashSet<String> unique_animation_names;

	Vector<Ref<GLTFSkeleton>> skeletons;
	Vector<Ref<GLTFAnimation>> animations;
	HashMap<GLTFNodeIndex, Node *> scene_nodes;
	HashMap<GLTFNodeIndex, ImporterMeshInstance3D *> scene_mesh_instances;

	HashMap<ObjectID, GLTFSkeletonIndex> skeleton3d_to_gltf_skeleton;
	HashMap<ObjectID, HashMap<ObjectID, GLTFSkinIndex>> skin_and_skeleton3d_to_gltf_skin;
	Dictionary additional_data;

protected:
	static void _bind_methods();

public:
	void add_used_extension(const String &p_extension, bool p_required = false);
	GLTFBufferViewIndex append_data_to_buffers(const Vector<uint8_t> &p_data, const bool p_deduplication);

	enum GLTFHandleBinary {
		HANDLE_BINARY_DISCARD_TEXTURES = 0,
		HANDLE_BINARY_EXTRACT_TEXTURES,
		HANDLE_BINARY_EMBED_AS_BASISU,
		HANDLE_BINARY_EMBED_AS_UNCOMPRESSED, // If this value changes from 3, ResourceImporterScene::pre_import must be changed as well.
	};
	int32_t get_handle_binary_image() {
		return handle_binary_image;
	}
	void set_handle_binary_image(int32_t p_handle_binary_image) {
		handle_binary_image = p_handle_binary_image;
	}

	Dictionary get_json();
	void set_json(Dictionary p_json);

	int get_major_version();
	void set_major_version(int p_major_version);

	int get_minor_version();
	void set_minor_version(int p_minor_version);

	String get_copyright() const;
	void set_copyright(const String &p_copyright);

	Vector<uint8_t> get_glb_data();
	void set_glb_data(Vector<uint8_t> p_glb_data);

	bool get_use_named_skin_binds();
	void set_use_named_skin_binds(bool p_use_named_skin_binds);

	bool get_discard_textures();
	void set_discard_textures(bool p_discard_textures);

	bool get_embed_as_basisu();
	void set_embed_as_basisu(bool p_embed_as_basisu);

	bool get_extract_textures();
	void set_extract_textures(bool p_extract_textures);

	bool get_discard_meshes_and_materials();
	void set_discard_meshes_and_materials(bool p_discard_meshes_and_materials);

	TypedArray<GLTFNode> get_nodes();
	void set_nodes(TypedArray<GLTFNode> p_nodes);
	void set_node_index(GLTFNodeIndex p_index, Ref<GLTFNode> p_node) {
		ERR_FAIL_INDEX(p_index, nodes.size());
		nodes.write[p_index] = p_node;
	}
	Vector<Ref<GLTFNode>> &get_nodes_reference() {
		return nodes;
	}

	TypedArray<PackedByteArray> get_buffers();
	void set_buffers(TypedArray<PackedByteArray> p_buffers);

	TypedArray<GLTFBufferView> get_buffer_views();
	void set_buffer_views(TypedArray<GLTFBufferView> p_buffer_views);

	TypedArray<GLTFAccessor> get_accessors();
	void set_accessors(TypedArray<GLTFAccessor> p_accessors);

	TypedArray<GLTFMesh> get_meshes();
	void set_meshes(TypedArray<GLTFMesh> p_meshes);

	TypedArray<Material> get_materials();
	void set_materials(TypedArray<Material> p_materials);
	void add_material(Ref<Material> p_material) {
		materials.push_back(p_material);
	}

	String get_scene_name();
	void set_scene_name(String p_scene_name);

	String get_base_path();
	void set_base_path(String p_base_path);

	String get_filename() const;
	void set_filename(const String &p_filename);

	PackedInt32Array get_root_nodes();
	void set_root_nodes(PackedInt32Array p_root_nodes);

	TypedArray<GLTFTexture> get_textures();
	void set_textures(TypedArray<GLTFTexture> p_textures);
	void add_texture(Ref<GLTFTexture> p_texture) {
		textures.push_back(p_texture);
	}

	TypedArray<GLTFTextureSampler> get_texture_samplers();
	void set_texture_samplers(TypedArray<GLTFTextureSampler> p_texture_samplers);

	TypedArray<Texture2D> get_images();
	void set_images(TypedArray<Texture2D> p_images);
	void set_image_index(GLTFTextureIndex p_index, Ref<Texture2D> p_texture) {
		images.write[p_index] = p_texture;
	}
	void add_image(Ref<Image> p_image) {
		images.push_back(p_image);
	}

	TypedArray<Image> get_source_images() {
		return to_array(source_images);
	}
	void set_source_images(TypedArray<Image> p_images) {
		source_images.clear();
		source_images.resize(p_images.size());
		for (int32_t image_i = 0; image_i < p_images.size(); image_i++) {
			source_images.write[image_i] = p_images[image_i];
		}
	}
	void set_source_image_index(GLTFImageIndex p_index, Ref<Image> p_texture) {
		images.write[p_index] = p_texture;
	}
	void add_source_image(Ref<Image> p_image) {
		source_images.push_back(p_image);
	}

	TypedArray<GLTFSkin> get_skins();
	void set_skins(TypedArray<GLTFSkin> p_skins);
	void add_skin(Ref<GLTFSkin> p_skin) {
		skins.push_back(p_skin);
	}

	Vector<Ref<GLTFSkin>> &get_skins_reference() {
		return skins;
	}

	TypedArray<GLTFCamera> get_cameras();
	void set_cameras(TypedArray<GLTFCamera> p_cameras);
	void add_camera(Ref<GLTFCamera> p_camera) {
		cameras.push_back(p_camera);
	}

	TypedArray<GLTFLight> get_lights();
	void set_lights(TypedArray<GLTFLight> p_lights);
	void add_light(Ref<GLTFLight> p_light) {
		lights.push_back(p_light);
	}

	TypedArray<String> get_unique_names();
	void set_unique_names(TypedArray<String> p_unique_names);

	HashSet<String> &get_unique_names_set() {
		return unique_names;
	}

	TypedArray<String> get_unique_animation_names();
	void set_unique_animation_names(TypedArray<String> p_unique_names);

	TypedArray<GLTFSkeleton> get_skeletons();
	void set_skeletons(TypedArray<GLTFSkeleton> p_skeletons);
	void set_skeleton_index(GLTFSkeletonIndex p_index, TypedArray<GLTFSkeleton> p_skeleton) {
		ERR_FAIL_INDEX(p_index, skeletons.size());
		skeletons.write[p_index] = p_skeleton;
	}
	Vector<Ref<GLTFSkeleton>>& get_skeletons_reference() {
		return skeletons;
	}

	bool get_create_animations();
	void set_create_animations(bool p_create_animations);

	TypedArray<GLTFAnimation> get_animations();
	void set_animations(TypedArray<GLTFAnimation> p_animations);
	void add_animation(Ref<GLTFAnimation> p_animation) {
		animations.push_back(p_animation);
	}

	Node *get_scene_node(GLTFNodeIndex idx);
	GLTFNodeIndex get_node_index(Node *p_node);

	int get_animation_players_count(int idx);

	AnimationPlayer *get_animation_player(int idx);

	Variant get_additional_data(const StringName &p_extension_name);
	void set_additional_data(const StringName &p_extension_name, Variant p_additional_data);

	void set_scene_mesh_instance_index(GLTFNodeIndex p_index, ObjectID p_object_id) {
		if (!scene_mesh_instances.has(p_index)) {
			return;
		}
		ImporterMeshInstance3D *mesh_instance = Object::cast_to<ImporterMeshInstance3D>(ObjectDB::get_instance(p_object_id));
		scene_mesh_instances[p_index] = mesh_instance;
	}

	HashMap<GLTFNodeIndex, Node *> &get_scene_nodes_reference() {
		return scene_nodes;
	}
};

#endif // GLTF_STATE_H
