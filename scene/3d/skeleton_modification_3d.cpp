/*************************************************************************/
/*  skeleton_modification_3d.cpp                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "skeleton_modification_3d.h"
#include "scene/3d/skeleton_3d.h"

void SkeletonModification3D::_validate_property(PropertyInfo &property) const {
	if (is_bone_property(property.name)) {
		// Because it is a constant function, we cannot use the _get_skeleton_3d function.
		const Skeleton3D *skel = get_skeleton();

		if (skel) {
			if (bone_name_list.is_empty()) {
				for (int i = 0; i < skel->get_bone_count(); i++) {
					if (i > 0) {
						bone_name_list += ",";
					}
					bone_name_list += skel->get_bone_name(i);
				}
			}

			property.hint = PROPERTY_HINT_ENUM;
			property.hint_string = bone_name_list;
		}
		else {
			property.hint = PROPERTY_HINT_NONE;
			property.hint_string = "";
		}
	}

	Node::_validate_property(property);
}

void SkeletonModification3D::set_enabled(bool p_enabled) {
	enabled = p_enabled;
}

bool SkeletonModification3D::get_enabled() const {
	return enabled;
}

void SkeletonModification3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_skeleton_path", "path"), &SkeletonModification3D::set_skeleton_path);
	ClassDB::bind_method(D_METHOD("get_skeleton_path"), &SkeletonModification3D::get_skeleton_path);
	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &SkeletonModification3D::set_enabled);
	ClassDB::bind_method(D_METHOD("get_enabled"), &SkeletonModification3D::get_enabled);
	ClassDB::bind_method(D_METHOD("execute", "delta"), &SkeletonModification3D::execute);

	ClassDB::bind_method(D_METHOD("resolve_bone", "target_bone_name"), &SkeletonModification3D::resolve_bone);
	ClassDB::bind_method(D_METHOD("resolve_target", "target_node_path", "target_bone_name"), &SkeletonModification3D::resolve_target);
	ClassDB::bind_method(D_METHOD("get_target_transform", "resolved_target"), &SkeletonModification3D::get_target_transform);
	ClassDB::bind_method(D_METHOD("get_target_quaternion", "resolved_target"), &SkeletonModification3D::get_target_quaternion);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "get_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "skeleton_path"), "set_skeleton_path", "get_skeleton_path");
}

NodePath SkeletonModification3D::get_skeleton_path() const {
	return skeleton_path;
}

void SkeletonModification3D::set_skeleton_path(NodePath p_path) {
	if (p_path.is_empty()) {
		p_path = NodePath("..");
	}
	skeleton_path = p_path;
	skeleton_change_queued = true;
	cached_skeleton = Variant();
	bone_name_list.clear();
	update_configuration_warnings();
}

Skeleton3D *SkeletonModification3D::get_skeleton() const {
	Skeleton3D *skeleton_node = cast_to<Skeleton3D>(cached_skeleton);
	if (skeleton_node == nullptr) {
		skeleton_node = cast_to<Skeleton3D>(get_node_or_null(skeleton_path));
		cached_skeleton = skeleton_node;
	}
	return skeleton_node;
}

void SkeletonModification3D::set_bone_property_suffixes(PackedStringArray suffixes) {
	bone_property_suffixes = suffixes;
}


void SkeletonModification3D::_notification(int p_what) {
	switch (p_what) {
	case NOTIFICATION_ENTER_TREE: {
		set_process_internal(get_enabled());
		set_physics_process_internal(false);
		cached_skeleton = Variant();
		if (Engine::get_singleton()->is_editor_hint()) {
			call_deferred(SNAME("update_configuration_warnings"));
		}
	} break;
	case NOTIFICATION_READY: {
		Skeleton3D *skel = get_skeleton();
		if (skel) {
			skeleton_changed(skel);
		}
	} break;
	case NOTIFICATION_INTERNAL_PROCESS: {
		ERR_FAIL_COND(!enabled);
		execute(get_process_delta_time());
	} break;
	}
}

void SkeletonModification3D::skeleton_changed(Skeleton3D *skeleton) {
	bone_name_list.clear();
	cached_skeleton_version = skeleton->get_version();
	skeleton_change_queued = false;
	GDVIRTUAL_CALL(_skeleton_changed, skeleton);
}

int SkeletonModification3D::resolve_bone(const String &target_bone_name) const {
	Skeleton3D *skel = get_skeleton();
	if (skel) {
		return skel->find_bone(target_bone_name);
	}
	return -1;
}

bool SkeletonModification3D::_cache_bone(int &bone_cache, const String &target_bone_name) const {
	if (bone_cache == UNCACHED_BONE_IDX) {
		bone_cache = resolve_bone(target_bone_name);
	}
	return bone_cache >= 0;
}

Variant SkeletonModification3D::resolve_target(const NodePath &target_node_path, const String &target_bone_name) const {
	if (target_node_path.is_empty()) {
		Skeleton3D *skel = get_skeleton();
		if (skel) {
			int found_bone = skel->find_bone(target_bone_name);
			if (found_bone >= 0) {
				return Variant(found_bone);
			}
		}
	} else {
		Node *resolved_node = get_node(target_node_path);
		if (cast_to<Node3D>(resolved_node)) {
			return Variant(resolved_node);
		}
	}
	return Variant(false);
}

bool SkeletonModification3D::_cache_target(Variant &cache, const NodePath &target_node_path, const String &target_bone_name) const {
	if (cache.get_type() == Variant::NIL) {
		cache = resolve_target(target_node_path, target_bone_name);
	}
	return cache.get_type() == Variant::OBJECT || cache.get_type() == Variant::INT;
}

Transform3D SkeletonModification3D::get_target_transform(Variant resolved_target) const {
	Skeleton3D *skel = get_skeleton();
	if (resolved_target.get_type() == Variant::OBJECT) {
		Node3D *resolved_node3d = cast_to<Node3D>((Object*)resolved_target);
		return skel->get_global_transform().affine_inverse() * resolved_node3d->get_global_transform();
	} else if (resolved_target.get_type() == Variant::INT) {
		int resolved_bone = (int)resolved_target;
		ERR_FAIL_COND_V(resolved_bone < 0, Transform3D());
		Transform3D xform = skel->get_bone_pose(resolved_bone);
		resolved_bone = skel->get_bone_parent(resolved_bone);
		while (resolved_bone >= 0) {
			xform = skel->get_bone_pose(resolved_bone) * xform;
			resolved_bone = skel->get_bone_parent(resolved_bone);
		}
		return xform;
	}
	ERR_FAIL_V_MSG(Transform3D(), "Looking up transform of unresolved target.");
}

Quaternion SkeletonModification3D::get_target_quaternion(Variant resolved_target) const {
	Skeleton3D *skel = get_skeleton();
	if (resolved_target.get_type() == Variant::OBJECT) {
		Node3D *resolved_node3d = cast_to<Node3D>((Object*)resolved_target);
		return skel->get_global_transform().basis.get_rotation_quaternion().inverse() * resolved_node3d->get_global_transform().basis.get_rotation_quaternion();
	}
	else if (resolved_target.get_type() == Variant::INT) {
		int resolved_bone = (int)resolved_target;
		ERR_FAIL_COND_V(resolved_bone < 0, Quaternion());
		Quaternion quat = skel->get_bone_pose_rotation(resolved_bone);
		resolved_bone = skel->get_bone_parent(resolved_bone);
		while (resolved_bone >= 0) {
			quat = skel->get_bone_pose_rotation(resolved_bone) * quat;
			resolved_bone = skel->get_bone_parent(resolved_bone);
		}
		return quat;
	}
	ERR_FAIL_V_MSG(Quaternion(), "Looking up quaternion of unresolved target.");
}

void SkeletonModification3D::execute(real_t delta) {
	Skeleton3D *skel = get_skeleton();
	bool skeleton_did_change = false;
	if (skel != nullptr) {
		if (skel->get_version() != cached_skeleton_version || skeleton_change_queued) {
			skeleton_changed(skel);
		}
	}
	GDVIRTUAL_CALL(_execute, delta);
}

bool SkeletonModification3D::is_bone_property(String property_name) const {
	for (const String &elem : bone_property_suffixes) {
		if (elem.is_empty() || elem[0] != '/') {
			if (property_name == elem) {
				return true;
			}
		} else {
			if (property_name.ends_with(elem)) {
				return true;
			}
		}
	}
	return false;
}

TypedArray<String> SkeletonModification3D::get_configuration_warnings() const {
	TypedArray<String> ret = Node::get_configuration_warnings();

	if (!get_skeleton()) {
		ret.push_back("Modification skeleton_path must point to a Skeleton3D node.");
	}

	return ret;
}
