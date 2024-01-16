/**************************************************************************/
/*  fbx_state.h                                                           */
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

#ifndef FBX_STATE_H
#define FBX_STATE_H

#include "modules/fbx/fbx_defines.h"
#include "modules/gltf/gltf_defines.h"
#include "modules/gltf/gltf_state.h"
#include "modules/gltf/structures/gltf_animation.h"
#include "modules/gltf/structures/gltf_mesh.h"
#include "modules/gltf/structures/gltf_node.h"
#include "modules/gltf/structures/gltf_skeleton.h"
#include "modules/gltf/structures/gltf_skin.h"
#include "modules/gltf/structures/gltf_texture.h"
#include "scene/3d/importer_mesh_instance_3d.h"
#include "structures/fbx_camera.h"
#include "structures/fbx_light.h"

#include "thirdparty/ufbx/ufbx.h"

class FBXState : public GLTFState {
	GDCLASS(FBXState, GLTFState);
	friend class FBXDocument;
	friend class SkinTool;
	friend class GLTFSkin;

	// Smart pointer that holds the loaded scene.
	ufbx_unique_ptr<ufbx_scene> scene;

	String base_path;
	String filename;
	int major_version = 0;
	int minor_version = 0;

	bool use_named_skin_binds = false;
	bool use_khr_texture_transform = false;
	bool discard_meshes_and_materials = false;
	bool allow_geometry_helper_nodes = false;
	bool create_animations = true;

	int handle_binary_image = HANDLE_BINARY_EXTRACT_TEXTURES;

	Vector<Ref<GLTFNode>> nodes;
	Vector<Vector<uint8_t>> buffers;

	Vector<Ref<GLTFMesh>> meshes; // Meshes are loaded directly, no reason not to.

	Vector<AnimationPlayer *> animation_players;
	HashMap<Ref<Material>, FBXMaterialIndex> material_cache;
	Vector<Ref<Material>> materials;

	String scene_name;
	Vector<int> root_nodes;
	Vector<Ref<GLTFTexture>> textures;
	Vector<Ref<Texture2D>> images;
	Vector<String> extensions_used;
	Vector<String> extensions_required;
	Vector<Ref<Image>> source_images;

	HashMap<uint64_t, Image::AlphaMode> alpha_mode_cache;
	HashMap<Pair<uint64_t, uint64_t>, GLTFTextureIndex, PairHash<uint64_t, uint64_t>> albedo_transparency_textures;

	Vector<Ref<GLTFSkin>> skins;
	Vector<GLTFSkinIndex> skin_indices;
	Vector<Ref<FBXCamera>> cameras;
	Vector<Ref<FBXLight>> lights;
	HashSet<String> unique_names;
	HashSet<String> unique_animation_names;

	Vector<Ref<GLTFSkeleton>> skeletons;
	Vector<Ref<GLTFAnimation>> animations;
	HashMap<GLTFNodeIndex, Node *> scene_nodes;
	HashMap<GLTFNodeIndex, ImporterMeshInstance3D *> scene_mesh_instances;

	HashMap<ObjectID, GLTFSkeletonIndex> skeleton3d_to_fbx_skeleton;
	HashMap<ObjectID, HashMap<ObjectID, GLTFSkinIndex>> skin_and_skeleton3d_to_fbx_skin;
	Dictionary additional_data;

protected:
	static void _bind_methods();

public:
	bool get_allow_geometry_helper_nodes();
	void set_allow_geometry_helper_nodes(bool p_allow_geometry_helper_nodes);
};

#endif // FBX_STATE_H
