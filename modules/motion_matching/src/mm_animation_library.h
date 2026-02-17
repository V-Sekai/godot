/**************************************************************************/
/*  mm_animation_library.h                                                */
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

#include "algo/kd_tree.h"
#include "common.h"
#include "features/mm_feature.h"
#include "mm_query.h"

#include "core/object/ref_counted.h"
#include "core/variant/typed_array.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/animation/animation_mixer.h"
#include "scene/resources/animation_library.h"

#ifdef TOOLS_ENABLED
#include "editor/scene/3d/node_3d_editor_gizmos.h"
#endif

#include <memory>

class MMFeature;
class MMCharacter;

class MMAnimationLibrary : public AnimationLibrary {
	GDCLASS(MMAnimationLibrary, AnimationLibrary)

public:
	void bake_data(const MMCharacter *p_character, const AnimationMixer *p_player, const Skeleton3D *p_skeleton);
	MMQueryOutput query(const MMQueryInput &p_query_input);
	int64_t get_dim_count() const;
	int64_t get_animation_pose_count(String p_animation_name) const;

#ifdef TOOLS_ENABLED
	void display_data(const Ref<EditorNode3DGizmo> &p_gizmo, const Transform3D &p_transform, String p_animation_name, int32_t p_pose_index) const;
#endif

	int64_t compute_features_hash() const;

	bool needs_baking() const;

	GETSET(TypedArray<MMFeature>, features)
	GETSET(float, sampling_rate, 1.f)
	GETSET(bool, include_cost_results, false);

	// Database data
	GETSET(PackedFloat32Array, motion_data)
	GETSET(PackedInt32Array, db_anim_index)
	GETSET(PackedFloat32Array, db_time_index)
	GETSET(PackedInt32Array, db_pose_offset)
	GETSET(int64_t, schema_hash)

	// KD Tree data
	GETSET(PackedInt32Array, node_indices);

protected:
	static void _bind_methods();

private:
	void _normalize_data(PackedFloat32Array &p_data, size_t p_dim_count) const;
	float _compute_feature_costs(int p_pose_index, const PackedFloat32Array &p_query, Dictionary *p_feature_costs) const;

	MMQueryOutput _search_naive(const PackedFloat32Array &p_query) const;
	MMQueryOutput _search_kd_tree(const PackedFloat32Array &p_query);

	std::unique_ptr<KDTree> _kd_tree;
};
