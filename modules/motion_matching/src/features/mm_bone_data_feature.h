/**************************************************************************/
/*  mm_bone_data_feature.h                                                */
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

#include "common.h"
#include "features/mm_feature.h"

class MMBoneDataFeature : public MMFeature {
	GDCLASS(MMBoneDataFeature, MMFeature)

public:
	virtual void setup_skeleton(const MMCharacter *p_character, const AnimationMixer *p_player, const Skeleton3D *p_skeleton) override;

	virtual void setup_for_animation(Ref<Animation> animation) override;

	virtual int64_t get_dimension_count() const override;

	virtual PackedFloat32Array bake_animation_pose(Ref<Animation> p_animation, double time) const override;

	virtual PackedFloat32Array evaluate_runtime_data(const MMQueryInput &p_query_input) const override;

#ifdef TOOLS_ENABLED
	virtual void display_data(const Ref<EditorNode3DGizmo> &p_gizmo, const Transform3D &p_transform, const float *p_data) const override;
#endif

	GETSET(PackedStringArray, bone_names, PackedStringArray());

protected:
	static void _bind_methods();

private:
	BoneState _sample_bone_state(Ref<Animation> p_animation, double p_time, const String &p_bone_path) const;
	StringName _skeleton_path;
	const Skeleton3D *_skeleton;
	int32_t _root_bone_index;
};
