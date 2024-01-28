/**************************************************************************/
/*  bone_attachment_3d.cpp                                                */
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

#include "bone_attachment_3d.h"
#include "bone_attachment_3d.compat.inc"

void BoneAttachment3D::_validate_property(PropertyInfo &p_property) const {
	SkeletonModifier3D::_validate_property(p_property);

	if (p_property.name == "bone_name") {
		// Because it is a constant function, we cannot use the _get_skeleton_3d function.
		const Skeleton3D *skeleton = get_skeleton();
		if (skeleton) {
			String names;
			for (int i = 0; i < skeleton->get_bone_count(); i++) {
				if (i > 0) {
					names += ",";
				}
				names += skeleton->get_bone_name(i);
			}

			p_property.hint = PROPERTY_HINT_ENUM;
			p_property.hint_string = names;
		} else {
			p_property.hint = PROPERTY_HINT_NONE;
			p_property.hint_string = "";
		}
	}
}

PackedStringArray BoneAttachment3D::get_configuration_warnings() const {
	PackedStringArray warnings = SkeletonModifier3D::get_configuration_warnings();
	if (bone_idx == -1) {
		warnings.push_back(RTR("BoneAttachment3D node is not bound to any bones! Please select a bone to attach this node."));
	}
	return warnings;
}

ObjectID BoneAttachment3D::_update_skeleton_path_extend() {
	BoneAttachment3D *parent_attachment = Object::cast_to<BoneAttachment3D>(get_parent());
	if (parent_attachment) {
		Skeleton3D *sk = parent_attachment->get_skeleton();
		if (sk) {
			return sk->get_instance_id();
		}
	}
	return ObjectID();
}

void BoneAttachment3D::_skeleton_changed(Skeleton3D *p_old, Skeleton3D *p_new) {
	bone_idx = -1;
	_update_bone();
}

void BoneAttachment3D::_update_bone() {
	if (!is_inside_tree()) {
		return;
	}
	if (bone_idx <= -1) {
		Skeleton3D *sk = get_skeleton();
		if (sk) {
			bone_idx = sk->find_bone(bone_name);
		}
	}
}

void BoneAttachment3D::_override_pose() {
	if (!is_inside_tree()) {
		return;
	}

	Skeleton3D *sk = get_skeleton();
	if (!sk) {
		return;
	}

	ERR_FAIL_NULL_MSG(sk, "Cannot override pose: Skeleton not found!");
	ERR_FAIL_INDEX_MSG(bone_idx, sk->get_bone_count(), "Cannot override pose: Bone index is out of range!");

	Transform3D our_trans = sk->get_global_transform().affine_inverse() * get_global_transform();
	sk->set_bone_global_pose(bone_idx, our_trans);
}

void BoneAttachment3D::_retrieve_pose() {
	if (!is_inside_tree()) {
		return;
	}

	Skeleton3D *sk = get_skeleton();
	if (!sk) {
		return;
	}

	ERR_FAIL_NULL_MSG(sk, "Cannot retrieve pose: Skeleton not found!");
	ERR_FAIL_INDEX_MSG(bone_idx, sk->get_bone_count(), "Cannot override pose: Bone index is out of range!");

	set_global_transform(sk->get_global_transform() * sk->get_bone_global_pose(bone_idx));
}

void BoneAttachment3D::_process_modification(double p_delta) {
	if (override_pose) {
		_override_pose();
	} else {
		_retrieve_pose();
	}
}

void BoneAttachment3D::set_bone_name(const String &p_name) {
	bone_name = p_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_bone_idx(sk->find_bone(bone_name));
	}
}

String BoneAttachment3D::get_bone_name() const {
	return bone_name;
}

void BoneAttachment3D::set_bone_idx(const int &p_idx) {
	bone_idx = p_idx;

	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (bone_idx <= -1 || bone_idx >= sk->get_bone_count()) {
			WARN_PRINT("Bone index out of range! Cannot connect BoneAttachment to node!");
			bone_idx = -1;
		} else {
			bone_name = sk->get_bone_name(bone_idx);
		}
	}
	_update_bone();

	notify_property_list_changed();
}

int BoneAttachment3D::get_bone_idx() const {
	return bone_idx;
}

void BoneAttachment3D::set_override_pose(bool p_override) {
	override_pose = p_override;
	set_notify_transform(override_pose);
	notify_property_list_changed();
}

bool BoneAttachment3D::get_override_pose() const {
	return override_pose;
}

void BoneAttachment3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_PARENTED: {
			_update_bone();
		} break;
	}
}

#ifndef DISABLE_DEPRECATED
void BoneAttachment3D::on_bone_pose_update(int p_bone_index) {
	if (bone_idx == p_bone_index) {
		_retrieve_pose();
	}
}
#endif // DISABLE_DEPRECATED

#ifdef TOOLS_ENABLED
void BoneAttachment3D::notify_skeleton_bones_renamed(Node *p_base_scene, Skeleton3D *p_skeleton, Dictionary p_rename_map) {
	const Skeleton3D *skeleton = get_skeleton();
	if (skeleton && skeleton == p_skeleton) {
		StringName bn = p_rename_map[bone_name];
		if (bn) {
			set_bone_name(bn);
		}
	}
}
#endif // TOOLS_ENABLED

BoneAttachment3D::BoneAttachment3D() {
}

void BoneAttachment3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_bone_name", "bone_name"), &BoneAttachment3D::set_bone_name);
	ClassDB::bind_method(D_METHOD("get_bone_name"), &BoneAttachment3D::get_bone_name);

	ClassDB::bind_method(D_METHOD("set_bone_idx", "bone_idx"), &BoneAttachment3D::set_bone_idx);
	ClassDB::bind_method(D_METHOD("get_bone_idx"), &BoneAttachment3D::get_bone_idx);

	ClassDB::bind_method(D_METHOD("set_override_pose", "override_pose"), &BoneAttachment3D::set_override_pose);
	ClassDB::bind_method(D_METHOD("get_override_pose"), &BoneAttachment3D::get_override_pose);

#ifndef DISABLE_DEPRECATED
	ClassDB::bind_method(D_METHOD("on_bone_pose_update", "bone_index"), &BoneAttachment3D::on_bone_pose_update);
#endif // DISABLE_DEPRECATED

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "bone_name"), "set_bone_name", "get_bone_name");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bone_idx"), "set_bone_idx", "get_bone_idx");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "override_pose"), "set_override_pose", "get_override_pose");
}
