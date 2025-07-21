/**************************************************************************/
/*  ewbik_document.h                                                      */
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
#include "scene/3d/skeleton_3d.h"

class EWBIK3D;
class IKEffectorTemplate3D;
class IKBone3D;
class IKBoneSegment3D;
class IKNode3D;

// SOA (Struct of Arrays) structures for performance optimization
struct BoneSOA {
	Vector<Transform3D> local_transforms;
	Vector<Transform3D> global_transforms;
	Vector<int32_t> parent_indices;
	Vector<String> names;
	Vector<float> lengths;
	Vector<bool> is_pinned;
	Vector<int32_t> bone_ids;
};

struct IKChainSOA {
	Vector<int32_t> bone_indices;
	Vector<float> weights;
	Vector<Vector3> target_positions;
	Vector<bool> is_active;
};

class EWBIKState : public RefCounted {
	GDCLASS(EWBIKState, RefCounted);

private:
	// Shadow copy of skeleton pose before IK simulation
	Vector<Transform3D> original_bone_poses;

	// Complete abstracted skeleton data (shadow bones)
	BoneSOA shadow_bones;
	Vector<IKChainSOA> shadow_chains;

	// IK configuration data
	Vector<Ref<IKEffectorTemplate3D>> effector_templates;

	// Simulation parameters
	int iterations_per_frame = 10;
	float default_damp = 0.1f;
	int stabilization_passes = 1;

protected:
	static void _bind_methods();

public:
	void set_original_bone_poses(const Vector<Transform3D> &p_poses);
	Vector<Transform3D> get_original_bone_poses() const;

	void set_effector_templates(const Vector<Ref<IKEffectorTemplate3D>> &p_templates);
	Vector<Ref<IKEffectorTemplate3D>> get_effector_templates() const;

	void set_iterations_per_frame(int p_iterations);
	int get_iterations_per_frame() const;

	void set_default_damp(float p_damp);
	float get_default_damp() const;

	void set_stabilization_passes(int p_passes);
	int get_stabilization_passes() const;

	void set_original_bone_poses_array(const Array &p_poses);
	Array get_original_bone_poses_array() const;

	void set_effector_templates_array(const Array &p_templates);
	Array get_effector_templates_array() const;

	// Shadow bone abstraction methods
	void set_shadow_bones(const BoneSOA &p_bones);
	const BoneSOA &get_shadow_bones() const;

	void set_shadow_chains(const Vector<IKChainSOA> &p_chains);
	const Vector<IKChainSOA> &get_shadow_chains() const;

	// Methods to build shadow bones from skeleton
	Error build_shadow_bones_from_skeleton(Skeleton3D *p_skeleton);
	Error apply_shadow_bones_to_skeleton(Skeleton3D *p_skeleton) const;
};

class EWBIKDocument : public Resource {
	GDCLASS(EWBIKDocument, Resource);

protected:
	static void _bind_methods();

public:
	// All methods are static and operate on state passed as parameters
	static Error append_from_scene(Node *p_node, Ref<EWBIKState> p_state, uint32_t p_flags = 0);
	static Node *generate_scene(Ref<EWBIKState> p_state, bool p_apply_simulation = true);

	// Simulation control - all state passed as parameters
	static Error simulate_ik(Ref<EWBIKState> p_state, Skeleton3D *p_skeleton);
	static Error simulate_ik(Ref<EWBIKState> p_state); // Fully abstracted version
	static Error apply_simulation_results(Ref<EWBIKState> p_state, Skeleton3D *p_skeleton);

private:
	// IK algorithm methods - all take state as parameters
	static void _bone_list_changed(Ref<EWBIKState> p_state, Skeleton3D *p_skeleton, const Vector<Ref<IKEffectorTemplate3D>> &p_pins,
			Vector<Ref<IKBone3D>> &r_bone_list, Vector<Ref<IKBoneSegment3D>> &r_segmented_skeletons,
			Ref<IKNode3D> &r_ik_origin, BoneSOA &r_bone_data,
			Vector<IKChainSOA> &r_ik_chains);
	static void _update_ik_bones_transform(const Vector<Ref<IKBone3D>> &p_bone_list);
	static void _update_skeleton_bones_transform(const Vector<Ref<IKBone3D>> &p_bone_list, Skeleton3D *p_skeleton);

	// Helper methods for scene processing - all take state as parameters
	static Error _extract_ik_from_scene(Node *p_node, Ref<EWBIKState> p_state);
	static Node *_create_ik_scene_node(Ref<EWBIKState> p_state);
	static Error _save_shadow_copy(Skeleton3D *p_skeleton, Ref<EWBIKState> p_state);
	static Error _restore_shadow_copy(Skeleton3D *p_skeleton, Ref<EWBIKState> p_state);
	static Skeleton3D *_find_skeleton_in_scene(Node *p_node);
};
