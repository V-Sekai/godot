/**************************************************************************/
/*  ewbik_document.cpp                                                    */
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

#include "ewbik_document.h"

#include "core/error/error_macros.h"
#include "core/math/math_defs.h"
#include "core/object/class_db.h"
#include "ewbik_3d.h"
#include "ik_bone_3d.h"
#include "ik_bone_segment_3d.h"
#include "ik_effector_3d.h"
#include "ik_effector_template_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/main/node.h"

// EWBIKState implementation

void EWBIKState::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_original_bone_poses", "poses"), &EWBIKState::set_original_bone_poses_array);
	ClassDB::bind_method(D_METHOD("get_original_bone_poses"), &EWBIKState::get_original_bone_poses_array);

	ClassDB::bind_method(D_METHOD("set_effector_templates", "templates"), &EWBIKState::set_effector_templates_array);
	ClassDB::bind_method(D_METHOD("get_effector_templates"), &EWBIKState::get_effector_templates_array);

	ClassDB::bind_method(D_METHOD("set_iterations_per_frame", "iterations"), &EWBIKState::set_iterations_per_frame);
	ClassDB::bind_method(D_METHOD("get_iterations_per_frame"), &EWBIKState::get_iterations_per_frame);

	ClassDB::bind_method(D_METHOD("set_default_damp", "damp"), &EWBIKState::set_default_damp);
	ClassDB::bind_method(D_METHOD("get_default_damp"), &EWBIKState::get_default_damp);

	ClassDB::bind_method(D_METHOD("set_stabilization_passes", "passes"), &EWBIKState::set_stabilization_passes);
	ClassDB::bind_method(D_METHOD("get_stabilization_passes"), &EWBIKState::get_stabilization_passes);
}

void EWBIKState::set_original_bone_poses(const Vector<Transform3D> &p_poses) {
	original_bone_poses = p_poses;
}

Vector<Transform3D> EWBIKState::get_original_bone_poses() const {
	return original_bone_poses;
}

void EWBIKState::set_effector_templates(const Vector<Ref<IKEffectorTemplate3D>> &p_templates) {
	effector_templates = p_templates;
}

Vector<Ref<IKEffectorTemplate3D>> EWBIKState::get_effector_templates() const {
	return effector_templates;
}

void EWBIKState::set_iterations_per_frame(int p_iterations) {
	iterations_per_frame = p_iterations;
}

int EWBIKState::get_iterations_per_frame() const {
	return iterations_per_frame;
}

void EWBIKState::set_default_damp(float p_damp) {
	default_damp = p_damp;
}

float EWBIKState::get_default_damp() const {
	return default_damp;
}

void EWBIKState::set_stabilization_passes(int p_passes) {
	stabilization_passes = p_passes;
}

int EWBIKState::get_stabilization_passes() const {
	return stabilization_passes;
}

void EWBIKState::set_original_bone_poses_array(const Array &p_poses) {
	original_bone_poses.clear();
	for (int i = 0; i < p_poses.size(); ++i) {
		Transform3D transform = p_poses[i];
		original_bone_poses.push_back(transform);
	}
}

Array EWBIKState::get_original_bone_poses_array() const {
	Array result;
	for (const Transform3D &transform : original_bone_poses) {
		result.push_back(transform);
	}
	return result;
}

void EWBIKState::set_effector_templates_array(const Array &p_templates) {
	effector_templates.clear();
	for (int i = 0; i < p_templates.size(); ++i) {
		Ref<IKEffectorTemplate3D> template_ref = p_templates[i];
		if (template_ref.is_valid()) {
			effector_templates.push_back(template_ref);
		}
	}
}

Array EWBIKState::get_effector_templates_array() const {
	Array result;
	for (const Ref<IKEffectorTemplate3D> &template_ref : effector_templates) {
		result.push_back(template_ref);
	}
	return result;
}

// Shadow bone abstraction methods
void EWBIKState::set_shadow_bones(const BoneSOA &p_bones) {
	shadow_bones = p_bones;
}

const BoneSOA &EWBIKState::get_shadow_bones() const {
	return shadow_bones;
}

void EWBIKState::set_shadow_chains(const Vector<IKChainSOA> &p_chains) {
	shadow_chains = p_chains;
}

const Vector<IKChainSOA> &EWBIKState::get_shadow_chains() const {
	return shadow_chains;
}

Error EWBIKState::build_shadow_bones_from_skeleton(Skeleton3D *p_skeleton) {
	ERR_FAIL_COND_V(p_skeleton == nullptr, ERR_INVALID_PARAMETER);

	int bone_count = p_skeleton->get_bone_count();
	if (bone_count == 0) {
		return ERR_INVALID_DATA;
	}

	// Clear existing shadow data
	shadow_bones.local_transforms.clear();
	shadow_bones.global_transforms.clear();
	shadow_bones.parent_indices.clear();
	shadow_bones.names.clear();
	shadow_bones.lengths.clear();
	shadow_bones.is_pinned.clear();
	shadow_bones.bone_ids.clear();

	// Resize arrays
	shadow_bones.local_transforms.resize(bone_count);
	shadow_bones.global_transforms.resize(bone_count);
	shadow_bones.parent_indices.resize(bone_count);
	shadow_bones.names.resize(bone_count);
	shadow_bones.lengths.resize(bone_count);
	shadow_bones.is_pinned.resize(bone_count);
	shadow_bones.bone_ids.resize(bone_count);

	// Populate bone data
	for (int i = 0; i < bone_count; ++i) {
		shadow_bones.bone_ids.write[i] = i;
		shadow_bones.names.write[i] = p_skeleton->get_bone_name(i);
		shadow_bones.parent_indices.write[i] = p_skeleton->get_bone_parent(i);
		shadow_bones.local_transforms.write[i] = p_skeleton->get_bone_pose(i);
		shadow_bones.global_transforms.write[i] = p_skeleton->get_bone_global_pose(i);
		shadow_bones.is_pinned.write[i] = false; // Will be set based on effectors

		// Calculate bone length
		float bone_length = 1.0f; // Default length
		const PackedInt32Array &children = p_skeleton->get_bone_children(i);
		if (!children.is_empty()) {
			Vector3 bone_pos = p_skeleton->get_bone_global_pose(i).origin;
			Vector3 child_pos = p_skeleton->get_bone_global_pose(children[0]).origin;
			bone_length = bone_pos.distance_to(child_pos);
		}
		shadow_bones.lengths.write[i] = bone_length;
	}

	// Mark pinned bones based on effector templates
	for (const Ref<IKEffectorTemplate3D> &effector : effector_templates) {
		if (effector.is_valid()) {
			String bone_name = effector->get_name();
			for (int i = 0; i < bone_count; ++i) {
				if (shadow_bones.names[i] == bone_name) {
					shadow_bones.is_pinned.write[i] = true;
					break;
				}
			}
		}
	}

	// Initialize IK chains from effectors
	shadow_chains.clear();
	for (const Ref<IKEffectorTemplate3D> &effector : effector_templates) {
		if (effector.is_valid()) {
			IKChainSOA chain;
			chain.is_active.push_back(true);
			chain.weights.push_back(effector->get_weight());
			chain.target_positions.push_back(Vector3()); // Will be updated during simulation
			// Note: bone_indices would need to be populated based on chain analysis
			shadow_chains.push_back(chain);
		}
	}

	return OK;
}

Error EWBIKState::apply_shadow_bones_to_skeleton(Skeleton3D *p_skeleton) const {
	ERR_FAIL_COND_V(p_skeleton == nullptr, ERR_INVALID_PARAMETER);

	int shadow_bone_count = shadow_bones.bone_ids.size();
	int skeleton_bone_count = p_skeleton->get_bone_count();

	// Apply transforms to skeleton (assuming matching bone count/order)
	for (int i = 0; i < MIN(shadow_bone_count, skeleton_bone_count); ++i) {
		p_skeleton->set_bone_pose(i, shadow_bones.local_transforms[i]);
	}

	return OK;
}

// EWBIKDocument implementation

static Error _simulate_ik_state_only(Ref<EWBIKState> p_state) {
	return EWBIKDocument::simulate_ik(p_state);
}

void EWBIKDocument::_bind_methods() {
	ClassDB::bind_static_method("EWBIKDocument", D_METHOD("append_from_scene", "node", "state", "flags"), &EWBIKDocument::append_from_scene, DEFVAL(0));
	ClassDB::bind_static_method("EWBIKDocument", D_METHOD("generate_scene", "state", "apply_simulation"), &EWBIKDocument::generate_scene, DEFVAL(true));

	ClassDB::bind_static_method("EWBIKDocument", D_METHOD("simulate_ik", "state", "skeleton"), static_cast<Error (*)(Ref<EWBIKState>, Skeleton3D *)>(&EWBIKDocument::simulate_ik));
	ClassDB::bind_static_method("EWBIKDocument", D_METHOD("simulate_ik_state_only", "state"), &_simulate_ik_state_only);
	ClassDB::bind_static_method("EWBIKDocument", D_METHOD("apply_simulation_results", "state", "skeleton"), &EWBIKDocument::apply_simulation_results);
}

Error EWBIKDocument::append_from_scene(Node *p_node, Ref<EWBIKState> p_state, uint32_t p_flags) {
	ERR_FAIL_COND_V(p_node == nullptr, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_state.is_null(), ERR_INVALID_PARAMETER);

	// Extract IK data from the scene
	Error err = _extract_ik_from_scene(p_node, p_state);
	if (err != OK) {
		return err;
	}

	// Save shadow copy of original bone poses
	Skeleton3D *skeleton = _find_skeleton_in_scene(p_node);
	if (skeleton) {
		err = _save_shadow_copy(skeleton, p_state);
		if (err != OK) {
			return err;
		}
	}

	return OK;
}

Node *EWBIKDocument::generate_scene(Ref<EWBIKState> p_state, bool p_apply_simulation) {
	ERR_FAIL_COND_V(p_state.is_null(), nullptr);

	// Create the IK scene node
	Node *scene_root = _create_ik_scene_node(p_state);

	if (p_apply_simulation && scene_root) {
		Skeleton3D *skeleton = _find_skeleton_in_scene(scene_root);
		if (skeleton) {
			// Simulate IK and apply results
			Error err = simulate_ik(p_state, skeleton);
			if (err == OK) {
				apply_simulation_results(p_state, skeleton);
			}
		}
	}

	return scene_root;
}

Error EWBIKDocument::simulate_ik(Ref<EWBIKState> p_state, Skeleton3D *p_skeleton) {
	ERR_FAIL_COND_V(p_state.is_null(), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_skeleton == nullptr, ERR_INVALID_PARAMETER);

	// Create temporary working structures
	Vector<Ref<IKBone3D>> bone_list;
	Vector<Ref<IKBoneSegment3D>> segmented_skeletons;
	Ref<IKNode3D> ik_origin;
	BoneSOA bone_data;
	Vector<IKChainSOA> ik_chains;

	// Set up IK algorithm state from the state parameters
	float default_damp = p_state->get_default_damp();

	// Set up bone structures for IK solving
	const Vector<Ref<IKEffectorTemplate3D>> &pins = p_state->get_effector_templates();
	_bone_list_changed(p_state, p_skeleton, pins, bone_list, segmented_skeletons, ik_origin, bone_data, ik_chains);

	// Update IK bones with current skeleton transforms
	_update_ik_bones_transform(bone_list);

	// Run IK iterations
	int iterations = p_state->get_iterations_per_frame();
	for (int i = 0; i < iterations; i++) {
		for (Ref<IKBoneSegment3D> segmented_skeleton : segmented_skeletons) {
			if (segmented_skeleton.is_null()) {
				continue;
			}
			segmented_skeleton->segment_solver(default_damp, i, iterations);
		}
	}

	// Update skeleton bones with solved poses
	_update_skeleton_bones_transform(bone_list, p_skeleton);

	return OK;
}

Error EWBIKDocument::simulate_ik(Ref<EWBIKState> p_state) {
	ERR_FAIL_COND_V(p_state.is_null(), ERR_INVALID_PARAMETER);

	// Check if shadow bones are available
	const BoneSOA &shadow_bones = p_state->get_shadow_bones();
	if (shadow_bones.bone_ids.is_empty()) {
		return ERR_INVALID_DATA;
	}

	// For now, this is a simplified implementation that works on the shadow bones
	// In a full implementation, this would run the actual IK algorithm on the SOA data

	// Placeholder: Just ensure shadow bones are properly initialized
	// Real implementation would:
	// 1. Update global transforms from local transforms
	// 2. Run IK iterations on the SOA data structures
	// 3. Update local transforms with solved poses

	// For demonstration, we'll just mark that simulation ran successfully
	// The actual IK algorithm would modify the shadow_bones.local_transforms

	return OK;
}

Error EWBIKDocument::apply_simulation_results(Ref<EWBIKState> p_state, Skeleton3D *p_skeleton) {
	ERR_FAIL_COND_V(p_state.is_null(), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_skeleton == nullptr, ERR_INVALID_PARAMETER);

	// Create a temporary IK solver to get bone information
	EWBIK3D *ik_solver = memnew(EWBIK3D());

	// Configure solver with state parameters
	ik_solver->set_iterations_per_frame(p_state->get_iterations_per_frame());
	ik_solver->set_default_damp(p_state->get_default_damp());
	ik_solver->set_stabilization_passes(p_state->get_stabilization_passes());

	// Configure effector templates
	const Vector<Ref<IKEffectorTemplate3D>> &templates = p_state->get_effector_templates();
	ik_solver->set_pin_count(templates.size());

	for (int i = 0; i < templates.size(); ++i) {
		const Ref<IKEffectorTemplate3D> &template_ref = templates[i];
		if (template_ref.is_valid()) {
			ik_solver->set_pin_bone_name(i, template_ref->get_name());
			ik_solver->set_pin_target_node_path(i, template_ref->get_target_node());
			ik_solver->set_pin_weight(i, template_ref->get_weight());
			ik_solver->set_pin_direction_priorities(i, template_ref->get_direction_priorities());
			ik_solver->set_pin_motion_propagation_factor(i, template_ref->get_motion_propagation_factor());
		}
	}

	// Note: In a real implementation, we would run the IK simulation here
	// and then apply the results. For now, this is a placeholder.

	// Clean up temporary solver
	memdelete(ik_solver);

	return OK;
}

Error EWBIKDocument::_extract_ik_from_scene(Node *p_node, Ref<EWBIKState> p_state) {
	// Find EWBIK3D nodes in the scene
	TypedArray<Node> ik_nodes = p_node->find_children("*", "EWBIK3D", true, false);

	if (ik_nodes.size() == 0) {
		return ERR_DOES_NOT_EXIST;
	}

	// Use the first IK node found (or merge multiple if needed)
	EWBIK3D *ik_solver = Object::cast_to<EWBIK3D>(ik_nodes[0]);
	if (!ik_solver) {
		return ERR_INVALID_DATA;
	}

	// Extract solver configuration
	p_state->set_iterations_per_frame(ik_solver->get_iterations_per_frame());
	p_state->set_default_damp(ik_solver->get_default_damp());
	p_state->set_stabilization_passes(ik_solver->get_stabilization_passes());

	// Extract effector templates by reconstructing from public API
	Vector<Ref<IKEffectorTemplate3D>> templates;
	int pin_count = ik_solver->get_pin_count();
	for (int i = 0; i < pin_count; ++i) {
		Ref<IKEffectorTemplate3D> template_ref = Ref<IKEffectorTemplate3D>(memnew(IKEffectorTemplate3D()));
		template_ref->set_name(ik_solver->get_pin_bone_name(i));
		template_ref->set_target_node(ik_solver->get_pin_target_node_path(i));
		template_ref->set_weight(ik_solver->get_pin_weight(i));
		template_ref->set_direction_priorities(ik_solver->get_pin_direction_priorities(i));
		template_ref->set_motion_propagation_factor(ik_solver->get_pin_motion_propagation_factor(i));
		templates.push_back(template_ref);
	}
	p_state->set_effector_templates(templates);

	return OK;
}

Node *EWBIKDocument::_create_ik_scene_node(Ref<EWBIKState> p_state) {
	// Create a new IK solver instance
	EWBIK3D *ik_solver = memnew(EWBIK3D());

	// Configure with state parameters
	ik_solver->set_iterations_per_frame(p_state->get_iterations_per_frame());
	ik_solver->set_default_damp(p_state->get_default_damp());
	ik_solver->set_stabilization_passes(p_state->get_stabilization_passes());

	// Copy effector configuration from templates
	const Vector<Ref<IKEffectorTemplate3D>> &templates = p_state->get_effector_templates();
	ik_solver->set_pin_count(templates.size());

	for (int i = 0; i < templates.size(); ++i) {
		const Ref<IKEffectorTemplate3D> &template_ref = templates[i];
		if (template_ref.is_valid()) {
			ik_solver->set_pin_bone_name(i, template_ref->get_name());
			ik_solver->set_pin_target_node_path(i, template_ref->get_target_node());
			ik_solver->set_pin_weight(i, template_ref->get_weight());
			ik_solver->set_pin_direction_priorities(i, template_ref->get_direction_priorities());
			ik_solver->set_pin_motion_propagation_factor(i, template_ref->get_motion_propagation_factor());
		}
	}

	return ik_solver;
}

Error EWBIKDocument::_save_shadow_copy(Skeleton3D *p_skeleton, Ref<EWBIKState> p_state) {
	ERR_FAIL_COND_V(p_skeleton == nullptr, ERR_INVALID_PARAMETER);

	Vector<Transform3D> original_poses;
	int current_bone_count = p_skeleton->get_bone_count();

	for (int i = 0; i < current_bone_count; ++i) {
		Transform3D pose = p_skeleton->get_bone_pose(i);
		original_poses.push_back(pose);
	}

	p_state->set_original_bone_poses(original_poses);
	return OK;
}

Error EWBIKDocument::_restore_shadow_copy(Skeleton3D *p_skeleton, Ref<EWBIKState> p_state) {
	ERR_FAIL_COND_V(p_skeleton == nullptr, ERR_INVALID_PARAMETER);

	const Vector<Transform3D> &original_poses = p_state->get_original_bone_poses();
	int shadow_bone_count = p_skeleton->get_bone_count();
	int pose_count = original_poses.size();

	// Restore original poses
	for (int i = 0; i < MIN(shadow_bone_count, pose_count); ++i) {
		p_skeleton->set_bone_pose(i, original_poses[i]);
	}

	return OK;
}

Skeleton3D *EWBIKDocument::_find_skeleton_in_scene(Node *p_node) {
	// Find skeleton in the scene hierarchy
	TypedArray<Node> skeletons = p_node->find_children("*", "Skeleton3D", true, false);
	if (skeletons.size() > 0) {
		return Object::cast_to<Skeleton3D>(skeletons[0]);
	}
	return nullptr;
}

void EWBIKDocument::_bone_list_changed(Ref<EWBIKState> p_state, Skeleton3D *p_skeleton, const Vector<Ref<IKEffectorTemplate3D>> &p_pins,
		Vector<Ref<IKBone3D>> &r_bone_list, Vector<Ref<IKBoneSegment3D>> &r_segmented_skeletons,
		Ref<IKNode3D> &r_ik_origin, BoneSOA &r_bone_data,
		Vector<IKChainSOA> &r_ik_chains) {
	ERR_FAIL_COND(p_skeleton == nullptr);

	// Clear existing structures
	r_bone_list.clear();
	r_segmented_skeletons.clear();

	// Clear SOA data
	r_bone_data.local_transforms.clear();
	r_bone_data.global_transforms.clear();
	r_bone_data.parent_indices.clear();
	r_bone_data.names.clear();
	r_bone_data.lengths.clear();
	r_bone_data.is_pinned.clear();
	r_bone_data.bone_ids.clear();

	r_ik_chains.clear();

	// Create ik_origin once outside the loop
	if (r_ik_origin.is_null()) {
		r_ik_origin.instantiate();
	}

	Vector<int32_t> roots = p_skeleton->get_parentless_bones();
	if (roots.is_empty()) {
		return;
	}

	int stabilize_passes = p_state->get_stabilization_passes();

	for (BoneId root_bone_index : roots) {
		String parentless_bone = p_skeleton->get_bone_name(root_bone_index);
		// Create a non-const copy of pins for the constructor
		Vector<Ref<IKEffectorTemplate3D>> pins_copy = p_pins;
		Ref<IKBoneSegment3D> segmented_skeleton = Ref<IKBoneSegment3D>(memnew(IKBoneSegment3D(p_skeleton, parentless_bone, pins_copy, nullptr, nullptr, root_bone_index, -1, stabilize_passes)));
		segmented_skeleton->get_root()->get_ik_transform()->set_parent(r_ik_origin);
		segmented_skeleton->generate_default_segments(pins_copy, root_bone_index, -1, nullptr);
		Vector<Ref<IKBone3D>> new_bone_list;
		segmented_skeleton->create_bone_list(new_bone_list, true);
		r_bone_list.append_array(new_bone_list);
		Vector<Vector<double>> weight_array;
		segmented_skeleton->update_pinned_list(weight_array);
		segmented_skeleton->recursive_create_headings_arrays_for(segmented_skeleton);
		r_segmented_skeletons.push_back(segmented_skeleton);
	}

	// Populate SOA bone data
	int32_t bone_count = r_bone_list.size();
	r_bone_data.local_transforms.resize(bone_count);
	r_bone_data.global_transforms.resize(bone_count);
	r_bone_data.parent_indices.resize(bone_count);
	r_bone_data.names.resize(bone_count);
	r_bone_data.lengths.resize(bone_count);
	r_bone_data.is_pinned.resize(bone_count);
	r_bone_data.bone_ids.resize(bone_count);

	for (int32_t i = 0; i < bone_count; ++i) {
		Ref<IKBone3D> bone = r_bone_list[i];
		if (bone.is_valid()) {
			r_bone_data.bone_ids.write[i] = bone->get_bone_id();
			r_bone_data.names.write[i] = bone->get_name();
			r_bone_data.is_pinned.write[i] = bone->is_pinned();

			// Calculate bone length from skeleton
			float bone_length = 1.0f; // Default length
			if (bone->get_bone_id() >= 0 && p_skeleton) {
				const PackedInt32Array &children = p_skeleton->get_bone_children(bone->get_bone_id());
				if (!children.is_empty()) {
					Vector3 bone_pos = p_skeleton->get_bone_global_pose(bone->get_bone_id()).origin;
					Vector3 child_pos = p_skeleton->get_bone_global_pose(children[0]).origin;
					bone_length = bone_pos.distance_to(child_pos);
				}
			}
			r_bone_data.lengths.write[i] = bone_length;

			// Get transforms from skeleton
			if (bone->get_bone_id() >= 0 && p_skeleton) {
				r_bone_data.local_transforms.write[i] = p_skeleton->get_bone_pose(bone->get_bone_id());
			}
		}
	}

	// Update default bone direction transforms
	for (Ref<IKBone3D> &ik_bone_3d : r_bone_list) {
		ik_bone_3d->update_default_bone_direction_transform(p_skeleton);
	}

	// Set up IK chains from effector templates
	for (const Ref<IKEffectorTemplate3D> &pin : p_pins) {
		if (pin.is_valid()) {
			IKChainSOA chain;
			chain.is_active.push_back(true);
			chain.weights.push_back(pin->get_weight());
			chain.target_positions.push_back(Vector3()); // Will be updated during simulation
			r_ik_chains.push_back(chain);
		}
	}
}

void EWBIKDocument::_update_ik_bones_transform(const Vector<Ref<IKBone3D>> &p_bone_list) {
	for (int32_t bone_i = p_bone_list.size(); bone_i-- > 0;) {
		Ref<IKBone3D> bone = p_bone_list[bone_i];
		if (bone.is_null()) {
			continue;
		}
		// Note: In the original EWBIK3D, this calls bone->set_initial_pose(get_skeleton())
		// For headless operation, we assume the skeleton poses are already set
		if (bone->is_pinned()) {
			bone->get_pin()->update_target_global_transform(nullptr, nullptr);
		}
	}
}

void EWBIKDocument::_update_skeleton_bones_transform(const Vector<Ref<IKBone3D>> &p_bone_list, Skeleton3D *p_skeleton) {
	ERR_FAIL_NULL(p_skeleton);

	for (int32_t bone_i = p_bone_list.size(); bone_i-- > 0;) {
		Ref<IKBone3D> bone = p_bone_list[bone_i];
		if (bone.is_null()) {
			continue;
		}
		if (bone->get_bone_id() == -1) {
			continue;
		}
		bone->set_skeleton_bone_pose(p_skeleton);
	}
}
