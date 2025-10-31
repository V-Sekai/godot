/**************************************************************************/
/*  ik_bone_segment_3d.cpp                                                */
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

#include "ik_bone_segment_3d.h"

#include "core/string/string_builder.h"
#include "ewbik_3d.h"
#include "ik_effector_3d.h"
#include "scene/3d/skeleton_3d.h"

Ref<IKBone3D> IKBoneSegment3D::get_root() const {
	return root;
}

Ref<IKBone3D> IKBoneSegment3D::get_tip() const {
	return tip;
}

bool IKBoneSegment3D::is_pinned() const {
	ERR_FAIL_COND_V(tip.is_null(), false);
	return tip->is_pinned();
}

Vector<Ref<IKBoneSegment3D>> IKBoneSegment3D::get_child_segments() const {
	return child_segments;
}

void IKBoneSegment3D::create_bone_list(Vector<Ref<IKBone3D>> &p_list, bool p_recursive) const {
	if (p_recursive) {
		for (int32_t child_i = 0; child_i < child_segments.size(); child_i++) {
			child_segments[child_i]->create_bone_list(p_list, p_recursive);
		}
	}
	Ref<IKBone3D> current_bone = tip;
	Vector<Ref<IKBone3D>> list;
	while (current_bone.is_valid()) {
		list.push_back(current_bone);
		if (current_bone == root) {
			break;
		}
		current_bone = current_bone->get_parent();
	}
	p_list.append_array(list);
}

void IKBoneSegment3D::update_pinned_list(Vector<Vector<double>> &r_weights) {
	// Segments are now independent - collect only this segment's effectors
	effector_list.clear();
	if (is_pinned()) {
		effector_list.push_back(tip->get_pin());
	}
	// Note: No longer collecting from child_segments since segments are independent
}

void IKBoneSegment3D::_update_optimal_rotation(Ref<IKBone3D> p_for_bone, double p_damp, bool p_translate, int32_t current_iteration, int32_t total_iterations) {
	ERR_FAIL_COND(p_for_bone.is_null());
	_update_target_headings(p_for_bone, &heading_weights, &target_headings);
	_update_tip_headings(p_for_bone, &tip_headings);
	_set_optimal_rotation(p_for_bone, &tip_headings, &target_headings, &heading_weights, p_damp, p_translate);
}

Quaternion IKBoneSegment3D::clamp_to_cos_half_angle(Quaternion p_quat, double p_cos_half_angle) {
	if (p_quat.w < 0.0) {
		p_quat = p_quat * -1;
	}
	double previous_coefficient = (1.0 - (p_quat.w * p_quat.w));
	if (p_cos_half_angle <= p_quat.w || previous_coefficient == 0.0) {
		return p_quat;
	} else {
		double composite_coefficient = Math::sqrt((1.0 - (p_cos_half_angle * p_cos_half_angle)) / previous_coefficient);
		p_quat.w = p_cos_half_angle;
		p_quat.x *= composite_coefficient;
		p_quat.y *= composite_coefficient;
		p_quat.z *= composite_coefficient;
	}
	return p_quat;
}

float IKBoneSegment3D::_get_manual_msd(const PackedVector3Array &r_htip, const PackedVector3Array &r_htarget, const Vector<double> &p_weights) {
	float manual_RMSD = 0.0f;
	float w_sum = 0.0f;
	for (int i = 0; i < r_htarget.size(); i++) {
		float x_d = r_htarget[i].x - r_htip[i].x;
		float y_d = r_htarget[i].y - r_htip[i].y;
		float z_d = r_htarget[i].z - r_htip[i].z;
		float mag_sq = p_weights[i] * (x_d * x_d + y_d * y_d + z_d * z_d);
		manual_RMSD += mag_sq;
		w_sum += p_weights[i];
	}
	manual_RMSD /= w_sum * w_sum;
	return manual_RMSD;
}

void IKBoneSegment3D::_set_optimal_rotation(Ref<IKBone3D> p_for_bone, PackedVector3Array *r_htip, PackedVector3Array *r_htarget, Vector<double> *r_weights, float p_dampening, bool p_translate, double current_iteration, double total_iterations) {
	ERR_FAIL_COND(p_for_bone.is_null());
	ERR_FAIL_NULL(r_htip);
	ERR_FAIL_NULL(r_htarget);
	ERR_FAIL_NULL(r_weights);

	_update_target_headings(p_for_bone, &heading_weights, &target_headings);
	Transform3D prev_transform = p_for_bone->get_pose();
	bool got_closer = true;
	double bone_damp = p_for_bone->get_cos_half_dampen();
	int i = 0;
	do {
		_update_tip_headings(p_for_bone, &tip_headings);
		Array superpose_result = QuaternionCharacteristicPolynomial::weighted_superpose(*r_htip, *r_htarget, *r_weights, p_translate, evec_prec);
		Quaternion rotation = superpose_result[0];
		Vector3 translation = superpose_result[1];
		double dampening = (p_dampening != -1.0) ? p_dampening : bone_damp;
		rotation = clamp_to_cos_half_angle(rotation, cos(dampening / 2.0));
		if (current_iteration == 0) {
			current_iteration = 0.0001;
		}
		rotation = rotation.slerp(p_for_bone->get_global_pose().basis.get_rotation_quaternion(), static_cast<double>(total_iterations) / current_iteration);
		p_for_bone->get_ik_transform()->rotate_local_with_global(rotation);
		Transform3D result = Transform3D(p_for_bone->get_global_pose().basis, p_for_bone->get_global_pose().origin + translation);
		p_for_bone->set_global_pose(result);
		if (default_stabilizing_pass_count > 0) {
			_update_tip_headings(p_for_bone, &tip_headings_uniform);
			double current_msd = _get_manual_msd(tip_headings_uniform, target_headings, heading_weights);
			if (current_msd <= previous_deviation * 1.0001) {
				previous_deviation = current_msd;
				got_closer = true;
				break;
			} else {
				got_closer = false;
				p_for_bone->set_pose(prev_transform);
			}
		}
		i++;
	} while (i < default_stabilizing_pass_count && !got_closer);

	if (root == p_for_bone) {
		previous_deviation = INFINITY;
	}
}

void IKBoneSegment3D::_update_target_headings(Ref<IKBone3D> p_for_bone, Vector<double> *r_weights, PackedVector3Array *r_target_headings) {
	ERR_FAIL_COND(p_for_bone.is_null());
	ERR_FAIL_NULL(r_weights);
	ERR_FAIL_NULL(r_target_headings);
	int32_t last_index = 0;
	for (int32_t effector_i = 0; effector_i < effector_list.size(); effector_i++) {
		Ref<IKEffector3D> effector = effector_list[effector_i];
		if (effector.is_null()) {
			continue;
		}
		last_index = effector->update_effector_target_headings(r_target_headings, last_index, p_for_bone, &heading_weights);
	}
}

void IKBoneSegment3D::_update_tip_headings(Ref<IKBone3D> p_for_bone, PackedVector3Array *r_heading_tip) {
	ERR_FAIL_NULL(r_heading_tip);
	ERR_FAIL_COND(p_for_bone.is_null());
	int32_t last_index = 0;
	for (int32_t effector_i = 0; effector_i < effector_list.size(); effector_i++) {
		Ref<IKEffector3D> effector = effector_list[effector_i];
		if (effector.is_null()) {
			continue;
		}
		last_index = effector->update_effector_tip_headings(r_heading_tip, last_index, p_for_bone);
	}
}

void IKBoneSegment3D::segment_solver(float p_default_damp, int32_t p_current_iteration, int32_t p_total_iteration) {
	// Segments are now independent - no recursive solving of child segments
	bool is_translate = parent_segment.is_null();
	if (is_translate) {
		_qcp_solver(Math::PI, is_translate, p_current_iteration, p_total_iteration);
		return;
	}
	_qcp_solver(p_default_damp, is_translate, p_current_iteration, p_total_iteration);
}

void IKBoneSegment3D::_qcp_solver(float p_default_damp, bool p_translate, int32_t p_current_iteration, int32_t p_total_iterations) {
	// Use Eron's decomposition algorithm to determine optimal solve order
	Vector<Ref<IKBone3D>> solve_order;
	apply_erons_decomposition(solve_order);

	// Solve bones in the determined order
	for (Ref<IKBone3D> current_bone : solve_order) {
		float damp = p_default_damp;
		bool is_non_default_damp = p_default_damp < damp;
		if (is_non_default_damp) {
			damp = p_default_damp;
		}
		_update_optimal_rotation(current_bone, damp, p_translate, p_current_iteration, p_total_iterations);
	}
}

void IKBoneSegment3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_pinned"), &IKBoneSegment3D::is_pinned);
	ClassDB::bind_method(D_METHOD("get_ik_bone", "bone"), &IKBoneSegment3D::get_ik_bone);
}

IKBoneSegment3D::IKBoneSegment3D(Skeleton3D *p_skeleton, StringName p_root_bone_name, Vector<Ref<IKEffectorTemplate3D>> &p_pins, EWBIK3D *p_many_bone_ik, const Ref<IKBoneSegment3D> &p_parent,
		BoneId p_root, BoneId p_tip, int32_t p_stabilizing_pass_count) {
	root = p_root;
	tip = p_tip;
	skeleton = p_skeleton;
	root = Ref<IKBone3D>(memnew(IKBone3D(p_root_bone_name, p_skeleton, p_parent, p_pins, Math::PI, p_many_bone_ik)));
	if (p_parent.is_valid()) {
		root_segment = p_parent->root_segment;
	} else {
		root_segment = Ref<IKBoneSegment3D>(this);
	}
	root_segment->bone_map[root->get_bone_id()] = root;
	if (p_parent.is_valid()) {
		parent_segment = p_parent;
		root->set_parent(p_parent->get_tip());
	}
	default_stabilizing_pass_count = p_stabilizing_pass_count;
}

void IKBoneSegment3D::_enable_pinned_descendants() {
	pinned_descendants = true;
}

bool IKBoneSegment3D::_has_pinned_descendants() {
	return pinned_descendants;
}

Ref<IKBone3D> IKBoneSegment3D::get_ik_bone(BoneId p_bone) const {
	if (!bone_map.has(p_bone)) {
		return Ref<IKBone3D>();
	}
	return bone_map[p_bone];
}

void IKBoneSegment3D::create_headings_arrays() {
	Vector<Vector<double>> penalty_array;
	Vector<Ref<IKBone3D>> new_pinned_bones;
	recursive_create_penalty_array(this, penalty_array, new_pinned_bones, 1.0);
	pinned_bones.resize(new_pinned_bones.size());
	int32_t total_headings = 0;
	for (const Vector<double> &current_penalty_array : penalty_array) {
		total_headings += current_penalty_array.size();
	}
	for (int32_t bone_i = 0; bone_i < new_pinned_bones.size(); bone_i++) {
		pinned_bones.write[bone_i] = new_pinned_bones[bone_i];
	}
	target_headings.resize(total_headings);
	tip_headings.resize(total_headings);
	tip_headings_uniform.resize(total_headings);
	heading_weights.resize(total_headings);
	int currentHeading = 0;
	for (const Vector<double> &current_penalty_array : penalty_array) {
		for (double ad : current_penalty_array) {
			heading_weights.write[currentHeading] = ad;
			target_headings.write[currentHeading] = Vector3();
			tip_headings.write[currentHeading] = Vector3();
			tip_headings_uniform.write[currentHeading] = Vector3();
			currentHeading++;
		}
	}
}

void IKBoneSegment3D::recursive_create_penalty_array(Ref<IKBoneSegment3D> p_bone_segment, Vector<Vector<double>> &r_penalty_array, Vector<Ref<IKBone3D>> &r_pinned_bones, double p_falloff) {
	if (p_falloff <= 0.0) {
		return;
	}

	double current_falloff = 1.0;

	if (p_bone_segment->is_pinned()) {
		Ref<IKBone3D> current_tip = p_bone_segment->get_tip();
		Ref<IKEffector3D> pin = current_tip->get_pin();
		double weight = pin->get_weight();
		Vector<double> inner_weight_array;
		inner_weight_array.push_back(weight * p_falloff);

		double max_pin_weight = MAX(MAX(pin->get_direction_priorities().x, pin->get_direction_priorities().y), pin->get_direction_priorities().z);
		max_pin_weight = max_pin_weight == 0.0 ? 1.0 : max_pin_weight;

		for (int i = 0; i < 3; ++i) {
			double priority = pin->get_direction_priorities()[i];
			if (priority > 0.0) {
				double sub_target_weight = weight * (priority / max_pin_weight) * p_falloff;
				inner_weight_array.push_back(sub_target_weight);
				inner_weight_array.push_back(sub_target_weight);
			}
		}

		r_penalty_array.push_back(inner_weight_array);
		r_pinned_bones.push_back(current_tip);
		current_falloff = pin->get_motion_propagation_factor();
	}

	for (Ref<IKBoneSegment3D> s : p_bone_segment->get_child_segments()) {
		recursive_create_penalty_array(s, r_penalty_array, r_pinned_bones, p_falloff * current_falloff);
	}
}

void IKBoneSegment3D::recursive_create_headings_arrays_for(Ref<IKBoneSegment3D> p_bone_segment) {
	p_bone_segment->create_headings_arrays();
	for (Ref<IKBoneSegment3D> segments : p_bone_segment->get_child_segments()) {
		recursive_create_headings_arrays_for(segments);
	}
}

void IKBoneSegment3D::generate_default_segments(Vector<Ref<IKEffectorTemplate3D>> &p_pins, BoneId p_root_bone, BoneId p_tip_bone, EWBIK3D *p_many_bone_ik) {
	Ref<IKBone3D> current_tip = root;
	Vector<BoneId> children;

	while (!_is_parent_of_tip(current_tip, p_tip_bone)) {
		children = skeleton->get_bone_children(current_tip->get_bone_id());

		if (children.is_empty() || _has_multiple_children_or_pinned(children, current_tip)) {
			_process_children(children, current_tip, p_pins, p_root_bone, p_tip_bone, p_many_bone_ik);
			break;
		} else {
			Vector<BoneId>::Iterator bone_id_iterator = children.begin();
			current_tip = _create_next_bone(*bone_id_iterator, current_tip, p_pins, p_many_bone_ik);
		}
	}

	_finalize_segment(current_tip);
}

bool IKBoneSegment3D::_is_parent_of_tip(Ref<IKBone3D> p_current_tip, BoneId p_tip_bone) {
	return skeleton->get_bone_parent(p_current_tip->get_bone_id()) >= p_tip_bone && p_tip_bone != -1;
}

bool IKBoneSegment3D::_has_multiple_children_or_pinned(Vector<BoneId> &r_children, Ref<IKBone3D> p_current_tip) {
	return r_children.size() > 1 || p_current_tip->is_pinned();
}

void IKBoneSegment3D::_process_children(Vector<BoneId> &r_children, Ref<IKBone3D> p_current_tip, Vector<Ref<IKEffectorTemplate3D>> &r_pins, BoneId p_root_bone, BoneId p_tip_bone, EWBIK3D *p_many_bone_ik) {
	tip = p_current_tip;
	Ref<IKBoneSegment3D> parent(this);

	for (int32_t child_i = 0; child_i < r_children.size(); child_i++) {
		BoneId child_bone = r_children[child_i];
		String child_name = skeleton->get_bone_name(child_bone);
		Ref<IKBoneSegment3D> child_segment = _create_child_segment(child_name, r_pins, p_root_bone, p_tip_bone, p_many_bone_ik, parent);

		child_segment->generate_default_segments(r_pins, p_root_bone, p_tip_bone, p_many_bone_ik);

		if (child_segment->_has_pinned_descendants()) {
			_enable_pinned_descendants();
			// Add independent segments directly to EWBIK3D instead of as child segments
			p_many_bone_ik->add_segment(child_segment);
		}
	}
}

Ref<IKBoneSegment3D> IKBoneSegment3D::_create_child_segment(String &p_child_name, Vector<Ref<IKEffectorTemplate3D>> &p_pins, BoneId p_root_bone, BoneId p_tip_bone, EWBIK3D *p_many_bone_ik, Ref<IKBoneSegment3D> &p_parent) {
	return Ref<IKBoneSegment3D>(memnew(IKBoneSegment3D(skeleton, p_child_name, p_pins, p_many_bone_ik, p_parent, p_root_bone, p_tip_bone)));
}

Ref<IKBone3D> IKBoneSegment3D::_create_next_bone(BoneId p_bone_id, Ref<IKBone3D> p_current_tip, Vector<Ref<IKEffectorTemplate3D>> &p_pins, EWBIK3D *p_many_bone_ik) {
	String bone_name = skeleton->get_bone_name(p_bone_id);
	Ref<IKBone3D> next_bone = Ref<IKBone3D>(memnew(IKBone3D(bone_name, skeleton, p_current_tip, p_pins, p_many_bone_ik->get_default_damp(), p_many_bone_ik)));
	root_segment->bone_map[p_bone_id] = next_bone;

	return next_bone;
}

void IKBoneSegment3D::_finalize_segment(Ref<IKBone3D> p_current_tip) {
	tip = p_current_tip;

	if (tip->is_pinned()) {
		_enable_pinned_descendants();
	}

	StringBuilder name_builder;
	name_builder.append("IKBoneSegment");
	name_builder.append(root->get_name());
	name_builder.append("Root");
	name_builder.append(tip->get_name());
	name_builder.append("Tip");

	String ik_bone_name = name_builder.as_string();
	set_name(ik_bone_name);
	bones.clear();
	create_bone_list(bones, false);
}

// Eron's decomposition algorithm implementation
void IKBoneSegment3D::apply_erons_decomposition(Vector<Ref<IKBone3D>> &r_solve_order) {
	r_solve_order.clear();

	// Phase 1: Traverse from each effector to build bone sequences
	Vector<EffectorTraversal> traversals;
	for (Ref<IKEffector3D> effector : effector_list) {
		if (effector.is_valid()) {
			traverse_from_effector(effector, traversals);
		}
	}

	// Phase 2: Consolidate into effector groups
	Vector<EffectorGroup> groups;
	consolidate_effector_groups(traversals, groups);

	// Phase 3: Create solve order
	create_solve_order(groups, r_solve_order);
}

void IKBoneSegment3D::traverse_from_effector(Ref<IKEffector3D> p_effector, Vector<EffectorTraversal> &r_traversals) {
	if (p_effector.is_null()) {
		return;
	}

	Ref<IKBone3D> current_bone = p_effector->get_ik_bone_3d();
	if (current_bone.is_null()) {
		return;
	}

	EffectorTraversal traversal;
	traversal.source_effector = p_effector;
	traversal.current_weight = 1.0f;

	// Traverse up from effector to root
	while (current_bone.is_valid()) {
		// Check if this bone is pinned by another effector
		if (is_bone_pinned(current_bone, p_effector)) {
			Ref<IKEffector3D> other_effector = current_bone->get_pin();
			if (other_effector != p_effector) {
				// Multiply current_weight by (1 - effector_opacity)
				float opacity = get_effector_opacity(other_effector);
				traversal.current_weight *= (1.0f - opacity);

				// If weight becomes 0, stop traversing
				if (traversal.current_weight <= 0.0f) {
					break;
				}
			}
		}

		// Add bone to encountered list
		traversal.bones_encountered.push_back(current_bone);

		// Move to parent bone
		current_bone = current_bone->get_parent();
	}

	r_traversals.push_back(traversal);
}

void IKBoneSegment3D::consolidate_effector_groups(const Vector<EffectorTraversal> &p_traversals, Vector<EffectorGroup> &r_groups) {
	// For each traversal, find or create effector groups with identical bone sequences
	for (const EffectorTraversal &traversal : p_traversals) {
		bool found_group = false;

		// Look for existing group with same bone sequence
		for (EffectorGroup &group : r_groups) {
			if (group.bone_sequence.size() == traversal.bones_encountered.size()) {
				bool sequences_match = true;
				for (int i = 0; i < group.bone_sequence.size(); ++i) {
					if (group.bone_sequence[i] != traversal.bones_encountered[i]) {
						sequences_match = false;
						break;
					}
				}

				if (sequences_match) {
					// Add effector to existing group
					group.effectors.push_back(traversal.source_effector);
					found_group = true;
					break;
				}
			}
		}

		if (!found_group) {
			// Create new group
			EffectorGroup new_group;
			new_group.bone_sequence = traversal.bones_encountered;
			new_group.effectors.push_back(traversal.source_effector);

			// Calculate root distance (distance from skeleton root)
			if (!new_group.bone_sequence.is_empty()) {
				Ref<IKBone3D> rootmost_bone = new_group.bone_sequence[new_group.bone_sequence.size() - 1];
				int distance = 0;
				Ref<IKBone3D> current = rootmost_bone;
				while (current.is_valid() && current->get_parent().is_valid()) {
					distance++;
					current = current->get_parent();
				}
				new_group.root_distance = distance;
			}

			r_groups.push_back(new_group);
		}
	}
}

void IKBoneSegment3D::create_solve_order(const Vector<EffectorGroup> &p_groups, Vector<Ref<IKBone3D>> &r_solve_order) {
	// Sort groups by root distance (reverse sorted - farthest from root first)
	Vector<EffectorGroup> sorted_groups = p_groups;

	// Manual bubble sort since Vector doesn't have sort_custom
	for (int i = 0; i < sorted_groups.size(); ++i) {
		for (int j = 0; j < sorted_groups.size() - 1 - i; ++j) {
			if (sorted_groups[j].root_distance < sorted_groups[j + 1].root_distance) {
				// Swap
				EffectorGroup temp = sorted_groups[j];
				sorted_groups.write[j] = sorted_groups[j + 1];
				sorted_groups.write[j + 1] = temp;
			}
		}
	}

	// Traverse groups and append bones to solve order
	for (const EffectorGroup &group : sorted_groups) {
		for (Ref<IKBone3D> bone : group.bone_sequence) {
			// Avoid duplicates
			if (!r_solve_order.has(bone)) {
				r_solve_order.push_back(bone);
			}
		}
	}
}

bool IKBoneSegment3D::is_bone_pinned(Ref<IKBone3D> p_bone, Ref<IKEffector3D> p_exclude_effector) const {
	if (p_bone.is_null()) {
		return false;
	}

	Ref<IKEffector3D> pin = p_bone->get_pin();
	return pin.is_valid() && pin != p_exclude_effector;
}

float IKBoneSegment3D::get_effector_opacity(Ref<IKEffector3D> p_effector) const {
	if (p_effector.is_null()) {
		return 0.0f;
	}
	// For now, use weight as opacity. This could be extended to use other properties
	return p_effector->get_weight();
}
