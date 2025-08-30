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

#pragma once

#include "modules/gltf/gltf_defines.h"
#include "modules/gltf/gltf_state.h"
#include "modules/gltf/structures/gltf_skeleton.h"
#include "modules/gltf/structures/gltf_skin.h"
#include "modules/gltf/structures/gltf_texture.h"

#include "../../thirdparty/ufbx/ufbx_export.h"
#include <ufbx.h>

class FBXState : public GLTFState {
	GDCLASS(FBXState, GLTFState);
	friend class FBXDocument;
	friend class SkinTool;
	friend class GLTFSkin;

public:
	enum CoordinateSystem {
		COORDINATE_SYSTEM_Y_UP,
		COORDINATE_SYSTEM_Z_UP
	};

private:
	// Smart pointer that holds the loaded scene.
	ufbx_unique_ptr<ufbx_scene> scene;
	// Export scene for FBX export functionality
	ufbx_export_scene *export_scene = nullptr;
	bool allow_geometry_helper_nodes = false;

	// Original scene root for animation export
	Node *original_scene_root = nullptr;

	// Export options for FBX export
	struct ufbx_export_opts export_opts = {};

	HashMap<uint64_t, Image::AlphaMode> alpha_mode_cache;
	HashMap<Pair<uint64_t, uint64_t>, GLTFTextureIndex> albedo_transparency_textures;

	Vector<GLTFSkinIndex> skin_indices;
	Vector<GLTFSkinIndex> original_skin_indices;
	HashMap<ObjectID, GLTFSkeletonIndex> skeleton3d_to_fbx_skeleton;
	HashMap<ObjectID, HashMap<ObjectID, GLTFSkinIndex>> skin_and_skeleton3d_to_fbx_skin;
	HashSet<String> unique_mesh_names; // Not in GLTFState because GLTFState prefixes mesh names with the scene name (or _)

protected:
	static void _bind_methods();

public:
	bool get_allow_geometry_helper_nodes();
	void set_allow_geometry_helper_nodes(bool p_allow_geometry_helper_nodes);

	// Export options API
	void set_export_ascii_format(bool p_ascii_format);
	bool get_export_ascii_format() const;
	void set_export_embed_textures(bool p_embed_textures);
	bool get_export_embed_textures() const;
	void set_export_animations(bool p_export_animations);
	bool get_export_animations() const;
	void set_export_materials(bool p_export_materials);
	bool get_export_materials() const;
	void set_export_fbx_version(int p_fbx_version);
	int get_export_fbx_version() const;
	void set_export_coordinate_system(CoordinateSystem p_coordinate_system);
	CoordinateSystem get_export_coordinate_system() const;
};

VARIANT_ENUM_CAST(FBXState::CoordinateSystem);
