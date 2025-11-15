/**************************************************************************/
/*  usd_state.h                                                          */
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

// TinyUSDZ headers
#include "tinyusdz.hh"

namespace tinyusdz {
	class Stage;
}

class USDState : public GLTFState {
	GDCLASS(USDState, GLTFState);
	friend class USDDocument;
	friend class SkinTool;
	friend class GLTFSkin;

	// TinyUSDZ stage reference
	tinyusdz::Stage stage;

	Vector<GLTFSkinIndex> skin_indices;
	Vector<GLTFSkinIndex> original_skin_indices;
	HashMap<ObjectID, GLTFSkeletonIndex> skeleton3d_to_usd_skeleton;
	HashMap<ObjectID, HashMap<ObjectID, GLTFSkinIndex>> skin_and_skeleton3d_to_usd_skin;
	HashSet<String> unique_mesh_names; // Not in GLTFState because GLTFState prefixes mesh names with the scene name (or _)
	
	// Performance optimization: cache prim path to node index mapping
	HashMap<String, GLTFNodeIndex> prim_path_to_node_index;

protected:
	static void _bind_methods();

public:
	const tinyusdz::Stage &get_stage() const { return stage; }
	tinyusdz::Stage &get_stage() { return stage; }
	void set_stage(const tinyusdz::Stage &p_stage) { stage = p_stage; }
};

