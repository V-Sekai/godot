/**************************************************************************/
/*  fbx_state.cpp                                                         */
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

#include "fbx_state.h"

#include "core/templates/template_convert.h"

void FBXState::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_used_extension", "extension_name", "required"), &FBXState::add_used_extension);
	ClassDB::bind_method(D_METHOD("get_major_version"), &FBXState::get_major_version);
	ClassDB::bind_method(D_METHOD("set_major_version", "major_version"), &FBXState::set_major_version);
	ClassDB::bind_method(D_METHOD("get_minor_version"), &FBXState::get_minor_version);
	ClassDB::bind_method(D_METHOD("set_minor_version", "minor_version"), &FBXState::set_minor_version);
	ClassDB::bind_method(D_METHOD("get_use_named_skin_binds"), &FBXState::get_use_named_skin_binds);
	ClassDB::bind_method(D_METHOD("set_use_named_skin_binds", "use_named_skin_binds"), &FBXState::set_use_named_skin_binds);
	ClassDB::bind_method(D_METHOD("get_nodes"), &FBXState::get_nodes);
	ClassDB::bind_method(D_METHOD("set_nodes", "nodes"), &FBXState::set_nodes);
	ClassDB::bind_method(D_METHOD("get_buffers"), &FBXState::get_buffers);
	ClassDB::bind_method(D_METHOD("set_buffers", "buffers"), &FBXState::set_buffers);
	ClassDB::bind_method(D_METHOD("get_meshes"), &FBXState::get_meshes);
	ClassDB::bind_method(D_METHOD("set_meshes", "meshes"), &FBXState::set_meshes);
	ClassDB::bind_method(D_METHOD("get_animation_players_count", "idx"), &FBXState::get_animation_players_count);
	ClassDB::bind_method(D_METHOD("get_animation_player", "idx"), &FBXState::get_animation_player);
	ClassDB::bind_method(D_METHOD("get_materials"), &FBXState::get_materials);
	ClassDB::bind_method(D_METHOD("set_materials", "materials"), &FBXState::set_materials);
	ClassDB::bind_method(D_METHOD("get_scene_name"), &FBXState::get_scene_name);
	ClassDB::bind_method(D_METHOD("set_scene_name", "scene_name"), &FBXState::set_scene_name);
	ClassDB::bind_method(D_METHOD("get_base_path"), &FBXState::get_base_path);
	ClassDB::bind_method(D_METHOD("set_base_path", "base_path"), &FBXState::set_base_path);
	ClassDB::bind_method(D_METHOD("get_filename"), &FBXState::get_filename);
	ClassDB::bind_method(D_METHOD("set_filename", "filename"), &FBXState::set_filename);
	ClassDB::bind_method(D_METHOD("get_root_nodes"), &FBXState::get_root_nodes);
	ClassDB::bind_method(D_METHOD("set_root_nodes", "root_nodes"), &FBXState::set_root_nodes);
	ClassDB::bind_method(D_METHOD("get_textures"), &FBXState::get_textures);
	ClassDB::bind_method(D_METHOD("set_textures", "textures"), &FBXState::set_textures);
	ClassDB::bind_method(D_METHOD("get_images"), &FBXState::get_images);
	ClassDB::bind_method(D_METHOD("set_images", "images"), &FBXState::set_images);
	ClassDB::bind_method(D_METHOD("get_skins"), &FBXState::get_skins);
	ClassDB::bind_method(D_METHOD("set_skins", "skins"), &FBXState::set_skins);
	ClassDB::bind_method(D_METHOD("get_cameras"), &FBXState::get_cameras);
	ClassDB::bind_method(D_METHOD("set_cameras", "cameras"), &FBXState::set_cameras);
	ClassDB::bind_method(D_METHOD("get_unique_names"), &FBXState::get_unique_names);
	ClassDB::bind_method(D_METHOD("set_unique_names", "unique_names"), &FBXState::set_unique_names);
	ClassDB::bind_method(D_METHOD("get_unique_animation_names"), &FBXState::get_unique_animation_names);
	ClassDB::bind_method(D_METHOD("set_unique_animation_names", "unique_animation_names"), &FBXState::set_unique_animation_names);
	ClassDB::bind_method(D_METHOD("get_skeletons"), &FBXState::get_skeletons);
	ClassDB::bind_method(D_METHOD("set_skeletons", "skeletons"), &FBXState::set_skeletons);
	ClassDB::bind_method(D_METHOD("get_create_animations"), &FBXState::get_create_animations);
	ClassDB::bind_method(D_METHOD("set_create_animations", "create_animations"), &FBXState::set_create_animations);
	ClassDB::bind_method(D_METHOD("get_animations"), &FBXState::get_animations);
	ClassDB::bind_method(D_METHOD("set_animations", "animations"), &FBXState::set_animations);
	ClassDB::bind_method(D_METHOD("get_scene_node", "idx"), &FBXState::get_scene_node);
	ClassDB::bind_method(D_METHOD("get_node_index", "scene_node"), &FBXState::get_node_index);
	ClassDB::bind_method(D_METHOD("get_additional_data", "extension_name"), &FBXState::get_additional_data);
	ClassDB::bind_method(D_METHOD("set_additional_data", "extension_name", "additional_data"), &FBXState::set_additional_data);
	ClassDB::bind_method(D_METHOD("get_handle_binary_image"), &FBXState::get_handle_binary_image);
	ClassDB::bind_method(D_METHOD("set_handle_binary_image", "method"), &FBXState::set_handle_binary_image);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "major_version"), "set_major_version", "get_major_version"); // int
	ADD_PROPERTY(PropertyInfo(Variant::INT, "minor_version"), "set_minor_version", "get_minor_version"); // int
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_named_skin_binds"), "set_use_named_skin_binds", "get_use_named_skin_binds"); // bool
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "nodes", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_INTERNAL | PROPERTY_USAGE_EDITOR), "set_nodes", "get_nodes"); // Vector<Ref<GLTFNode>>
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "buffers"), "set_buffers", "get_buffers"); // Vector<Vector<uint8_t>
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "meshes", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_INTERNAL | PROPERTY_USAGE_EDITOR), "set_meshes", "get_meshes"); // Vector<Ref<FBXMesh>>
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "materials", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_INTERNAL | PROPERTY_USAGE_EDITOR), "set_materials", "get_materials"); // Vector<Ref<Material>
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "scene_name"), "set_scene_name", "get_scene_name"); // String
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "base_path"), "set_base_path", "get_base_path"); // String
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "filename"), "set_filename", "get_filename"); // String
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "root_nodes"), "set_root_nodes", "get_root_nodes"); // Vector<int>
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "textures", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_INTERNAL | PROPERTY_USAGE_EDITOR), "set_textures", "get_textures"); // Vector<Ref<FBXTexture>>
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "images", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_INTERNAL | PROPERTY_USAGE_EDITOR), "set_images", "get_images"); // Vector<Ref<Texture>
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "skins", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_INTERNAL | PROPERTY_USAGE_EDITOR), "set_skins", "get_skins"); // Vector<Ref<GLTFSkin>>
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "cameras", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_INTERNAL | PROPERTY_USAGE_EDITOR), "set_cameras", "get_cameras"); // Vector<Ref<FBXCamera>>
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "unique_names", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_INTERNAL | PROPERTY_USAGE_EDITOR), "set_unique_names", "get_unique_names"); // Set<String>
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "unique_animation_names", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_INTERNAL | PROPERTY_USAGE_EDITOR), "set_unique_animation_names", "get_unique_animation_names"); // Set<String>
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "skeletons", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_INTERNAL | PROPERTY_USAGE_EDITOR), "set_skeletons", "get_skeletons"); // Vector<Ref<GLTFSkeleton>>
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "create_animations"), "set_create_animations", "get_create_animations"); // bool
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "animations", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_INTERNAL | PROPERTY_USAGE_EDITOR), "set_animations", "get_animations"); // Vector<Ref<FBXAnimation>>
	ADD_PROPERTY(PropertyInfo(Variant::INT, "handle_binary_image", PROPERTY_HINT_ENUM, "Discard All Textures,Extract Textures,Embed As Basis Universal,Embed as Uncompressed", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_INTERNAL | PROPERTY_USAGE_EDITOR), "set_handle_binary_image", "get_handle_binary_image"); // enum

	BIND_CONSTANT(HANDLE_BINARY_DISCARD_TEXTURES);
	BIND_CONSTANT(HANDLE_BINARY_EXTRACT_TEXTURES);
	BIND_CONSTANT(HANDLE_BINARY_EMBED_AS_BASISU);
	BIND_CONSTANT(HANDLE_BINARY_EMBED_AS_UNCOMPRESSED);
}

void FBXState::add_used_extension(const String &p_extension_name, bool p_required) {
	if (!extensions_used.has(p_extension_name)) {
		extensions_used.push_back(p_extension_name);
	}
	if (p_required) {
		if (!extensions_required.has(p_extension_name)) {
			extensions_required.push_back(p_extension_name);
		}
	}
}

int FBXState::get_major_version() {
	return major_version;
}

void FBXState::set_major_version(int p_major_version) {
	major_version = p_major_version;
}

int FBXState::get_minor_version() {
	return minor_version;
}

void FBXState::set_minor_version(int p_minor_version) {
	minor_version = p_minor_version;
}

bool FBXState::get_use_named_skin_binds() {
	return use_named_skin_binds;
}

void FBXState::set_use_named_skin_binds(bool p_use_named_skin_binds) {
	use_named_skin_binds = p_use_named_skin_binds;
}

TypedArray<GLTFNode> FBXState::get_nodes() {
	return to_array(nodes);
}

void FBXState::set_nodes(TypedArray<GLTFNode> p_nodes) {
	set_from_array(nodes, p_nodes);
}

TypedArray<PackedByteArray> FBXState::get_buffers() {
	return to_array(buffers);
}

void FBXState::set_buffers(TypedArray<PackedByteArray> p_buffers) {
	set_from_array(buffers, p_buffers);
}

TypedArray<FBXMesh> FBXState::get_meshes() {
	return to_array(meshes);
}

void FBXState::set_meshes(TypedArray<FBXMesh> p_meshes) {
	set_from_array(meshes, p_meshes);
}

TypedArray<Material> FBXState::get_materials() {
	return to_array(materials);
}

void FBXState::set_materials(TypedArray<Material> p_materials) {
	set_from_array(materials, p_materials);
}

String FBXState::get_scene_name() {
	return scene_name;
}

void FBXState::set_scene_name(String p_scene_name) {
	scene_name = p_scene_name;
}

PackedInt32Array FBXState::get_root_nodes() {
	return root_nodes;
}

void FBXState::set_root_nodes(PackedInt32Array p_root_nodes) {
	root_nodes = p_root_nodes;
}

TypedArray<FBXTexture> FBXState::get_textures() {
	return to_array(textures);
}

void FBXState::set_textures(TypedArray<FBXTexture> p_textures) {
	set_from_array(textures, p_textures);
}

TypedArray<Texture2D> FBXState::get_images() {
	return to_array(images);
}

void FBXState::set_images(TypedArray<Texture2D> p_images) {
	set_from_array(images, p_images);
}

TypedArray<GLTFSkin> FBXState::get_skins() {
	return to_array(skins);
}

void FBXState::set_skins(TypedArray<GLTFSkin> p_skins) {
	set_from_array(skins, p_skins);
}

TypedArray<FBXCamera> FBXState::get_cameras() {
	return to_array(cameras);
}

void FBXState::set_cameras(TypedArray<FBXCamera> p_cameras) {
	set_from_array(cameras, p_cameras);
}

TypedArray<String> FBXState::get_unique_names() {
	return to_array(unique_names);
}

void FBXState::set_unique_names(TypedArray<String> p_unique_names) {
	set_from_array(unique_names, p_unique_names);
}

TypedArray<String> FBXState::get_unique_animation_names() {
	return to_array(unique_animation_names);
}

void FBXState::set_unique_animation_names(TypedArray<String> p_unique_animation_names) {
	set_from_array(unique_animation_names, p_unique_animation_names);
}

TypedArray<GLTFSkeleton> FBXState::get_skeletons() {
	return to_array(skeletons);
}

void FBXState::set_skeletons(TypedArray<GLTFSkeleton> p_skeletons) {
	set_from_array(skeletons, p_skeletons);
}

bool FBXState::get_create_animations() {
	return create_animations;
}

void FBXState::set_create_animations(bool p_create_animations) {
	create_animations = p_create_animations;
}

TypedArray<FBXAnimation> FBXState::get_animations() {
	return to_array(animations);
}

void FBXState::set_animations(TypedArray<FBXAnimation> p_animations) {
	set_from_array(animations, p_animations);
}

Node *FBXState::get_scene_node(GLTFNodeIndex idx) {
	if (!scene_nodes.has(idx)) {
		return nullptr;
	}
	return scene_nodes[idx];
}

GLTFNodeIndex FBXState::get_node_index(Node *p_node) {
	for (KeyValue<GLTFNodeIndex, Node *> x : scene_nodes) {
		if (x.value == p_node) {
			return x.key;
		}
	}
	return -1;
}

int FBXState::get_animation_players_count(int idx) {
	return animation_players.size();
}

AnimationPlayer *FBXState::get_animation_player(int idx) {
	ERR_FAIL_INDEX_V(idx, animation_players.size(), nullptr);
	return animation_players[idx];
}

void FBXState::set_discard_meshes_and_materials(bool p_discard_meshes_and_materials) {
	discard_meshes_and_materials = p_discard_meshes_and_materials;
}

bool FBXState::get_discard_meshes_and_materials() {
	return discard_meshes_and_materials;
}

String FBXState::get_base_path() {
	return base_path;
}

void FBXState::set_base_path(String p_base_path) {
	base_path = p_base_path;
}

String FBXState::get_filename() const {
	return filename;
}

void FBXState::set_filename(const String &p_filename) {
	filename = p_filename;
}

Variant FBXState::get_additional_data(const StringName &p_extension_name) {
	return additional_data[p_extension_name];
}

void FBXState::set_additional_data(const StringName &p_extension_name, Variant p_additional_data) {
	additional_data[p_extension_name] = p_additional_data;
}

bool FBXState::get_allow_geometry_helper_nodes() {
	return allow_geometry_helper_nodes;
}

void FBXState::set_allow_geometry_helper_nodes(bool p_allow_geometry_helper_nodes) {
	allow_geometry_helper_nodes = p_allow_geometry_helper_nodes;
}
