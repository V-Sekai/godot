/**************************************************************************/
/*  skin_tool.h                                                           */
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

#include "core/io/resource.h"
#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/templates/rb_set.h"

#include "core/math/disjoint_set.h"

#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"
#include "modules/fbx/fbx_defines.h"
#include "modules/gltf/gltf_defines.h"
#include "modules/gltf/structures/gltf_node.h"
#include "modules/gltf/structures/gltf_skeleton.h"
#include "modules/gltf/structures/gltf_skin.h"
#include "scene/main/node.h"
#include "scene/resources/skin.h"

using SkinNodeIndex = int;
using SkinSkeletonIndex = int;

class SkinTool : public Resource {
	GDCLASS(SkinTool, Resource);

	static String _sanitize_bone_name(const String &p_name);
	static String _gen_unique_bone_name(HashSet<String> unique_names, const String &p_name);
	static SkinNodeIndex _find_highest_node(
			TypedArray<GLTFNode> &nodes, const Vector<SkinNodeIndex> &p_subset);
	static bool _capture_nodes_in_skin(const TypedArray<GLTFNode> &nodes, Ref<GLTFSkin> p_skin, const SkinNodeIndex p_node_index);
	static void _capture_nodes_for_multirooted_skin(TypedArray<GLTFNode> &r_nodes, Ref<GLTFSkin> p_skin);
	static void _recurse_children(
			TypedArray<GLTFNode> &nodes,
			const SkinNodeIndex p_node_index,
			RBSet<GLTFNodeIndex> &p_all_skin_nodes,
			HashSet<GLTFNodeIndex> &p_child_visited_set);
	static Error _reparent_non_joint_skeleton_subtrees(
			TypedArray<GLTFNode> &nodes,
			Ref<GLTFSkeleton> p_skeleton,
			const Vector<SkinNodeIndex> &p_non_joints);
	static Error _determine_skeleton_roots(
			TypedArray<GLTFNode> &nodes,
			TypedArray<GLTFSkeleton> &skeletons,
			const SkinSkeletonIndex p_skel_i);
	static Error _map_skin_joints_indices_to_skeleton_bone_indices(
			TypedArray<GLTFSkin> &skins,
			TypedArray<GLTFSkeleton> &skeletons,
			TypedArray<GLTFNode> &nodes);
	static bool _skins_are_same(const Ref<Skin> p_skin_a, const Ref<Skin> p_skin_b);
	static void _remove_duplicate_skins(TypedArray<GLTFSkin> &r_skins);

public:
	static Error _expand_skin(TypedArray<GLTFNode> &r_nodes, Ref<GLTFSkin> p_skin);
	static Error _verify_skin(TypedArray<GLTFNode> &r_nodes, Ref<GLTFSkin> p_skin);
	static Error asset_parse_skins(
			const Vector<SkinNodeIndex> &input_skin_indices,
			const TypedArray<Dictionary> &input_skins,
			const TypedArray<Dictionary> &input_nodes,
			Vector<SkinNodeIndex> &output_skin_indices,
			TypedArray<Dictionary> &output_skins,
			HashMap<GLTFNodeIndex, bool> &joint_mapping);
	static Error _determine_skeletons(
			TypedArray<GLTFSkin> &skins,
			TypedArray<GLTFNode> &nodes,
			TypedArray<GLTFSkeleton> &skeletons);
	static Error _create_skeletons(
			HashSet<String> &unique_names,
			TypedArray<GLTFSkin> &skins,
			TypedArray<GLTFNode> &nodes,
			HashMap<ObjectID, GLTFSkeletonIndex> &skeleton3d_to_fbx_skeleton,
			TypedArray<GLTFSkeleton> &skeletons,
			HashMap<GLTFNodeIndex, Node *> &scene_nodes);
	static Error _create_skins(TypedArray<GLTFSkin> &skins, TypedArray<GLTFNode> &nodes, bool use_named_skin_binds, HashSet<String> &unique_names);
};