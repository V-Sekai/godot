/**************************************************************************/
/*  skeleton_modifier_3d.cpp                                              */
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

#include "skeleton_modifier_3d.h"

void SkeletonModifier3D::_validate_property(PropertyInfo &p_property) const {
	if (animation_mixer_id.is_valid() && p_property.name == "callback_mode_process") {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
}

PackedStringArray SkeletonModifier3D::get_configuration_warnings() const {
	PackedStringArray warnings = Node3D::get_configuration_warnings();
	if (skeleton_id.is_null()) {
		warnings.push_back(RTR("Skeleton3D node not set! SkeletonModifier3D must be child of Skeleton3D or set a path to an external skeleton."));
	}
	return warnings;
}

void SkeletonModifier3D::rebind(bool p_force) {
	_update_skeleton(p_force);
	_update_animation_mixer(p_force);
	_rebind(p_force);
}

void SkeletonModifier3D::_rebind(bool p_force) {
	//
}

/* Skeleton3D */

Skeleton3D *SkeletonModifier3D::get_skeleton() const {
	return Object::cast_to<Skeleton3D>(ObjectDB::get_instance(skeleton_id));
}

void SkeletonModifier3D::set_external_skeleton(const NodePath &p_path) {
	if (external_skeleton == p_path) {
		return;
	}
	external_skeleton = p_path;
	_update_skeleton(true);
	notify_property_list_changed();
}

NodePath SkeletonModifier3D::get_external_skeleton() const {
	return external_skeleton;
}

void SkeletonModifier3D::_update_skeleton_path() {
	skeleton_id = ObjectID();

	if (!external_skeleton.is_empty()) {
		Skeleton3D *sk = Object::cast_to<Skeleton3D>(get_node_or_null(external_skeleton));
		ERR_FAIL_NULL_MSG(sk, "Cannot update skeleton cache: Node cannot be found!");
		skeleton_id = sk->get_instance_id();
		return;
	}

	// Make sure parent is a Skeleton3D.
	Skeleton3D *sk = Object::cast_to<Skeleton3D>(get_parent());
	if (sk) {
		skeleton_id = sk->get_instance_id();
		return;
	}

	// Make sure if it can get a Skeleton3D with another way.
	ObjectID candidated = _update_skeleton_path_extend();
	if (candidated.is_null()) {
		ERR_FAIL_NULL_MSG(sk, "Cannot update skeleton cache: Node cannot be found!");
	}

	skeleton_id = candidated;
}

ObjectID SkeletonModifier3D::_update_skeleton_path_extend() {
	return ObjectID();
}

void SkeletonModifier3D::_update_skeleton(bool p_path_changed) {
	if (!is_inside_tree()) {
		return;
	}
	Skeleton3D *old_sk = get_skeleton();
	if (!p_path_changed && !external_skeleton.is_empty() && old_sk) {
		return; // Probably Skeleton is not changed.
	}
	_update_skeleton_path();
	Skeleton3D *new_sk = get_skeleton();
	if (old_sk != new_sk) {
		_skeleton_changed(old_sk, new_sk);
	}
	update_configuration_warnings();
}

void SkeletonModifier3D::_skeleton_changed(Skeleton3D *p_old, Skeleton3D *p_new) {
	//
}

/* AnimationMixer */

AnimationMixer *SkeletonModifier3D::get_animation_mixer() const {
	return Object::cast_to<AnimationMixer>(ObjectDB::get_instance(animation_mixer_id));
}

void SkeletonModifier3D::set_target_animation_mixer(const NodePath &p_path) {
	if (target_animation_mixer == p_path) {
		return;
	}

	target_animation_mixer = p_path;
	_update_animation_mixer(true);

	_process_changed();
	notify_property_list_changed();
}

NodePath SkeletonModifier3D::get_target_animation_mixer() const {
	return target_animation_mixer;
}

void SkeletonModifier3D::_update_animation_mixer_path() {
	animation_mixer_id = ObjectID();

	if (target_animation_mixer.is_empty()) {
		return;
	}

	AnimationMixer *am = Object::cast_to<AnimationMixer>(get_node_or_null(target_animation_mixer));
	if (am) {
		animation_mixer_id = am->get_instance_id();
	}
}

void SkeletonModifier3D::_update_animation_mixer(bool p_path_changed) {
	if (!is_inside_tree()) {
		return;
	}
	AnimationMixer *old_am = get_animation_mixer();
	if (!p_path_changed && !target_animation_mixer.is_empty() && old_am) {
		return; // Probably AnimationMixer is not changed.
	}
	_update_animation_mixer_path();
	AnimationMixer *new_am = get_animation_mixer();
	if (old_am != new_am) {
		_animation_mixer_changed(old_am, new_am);
	}
}

void SkeletonModifier3D::_animation_mixer_changed(AnimationMixer *p_old, AnimationMixer *p_new) {
	if (p_old && p_old->is_connected(SNAME("mixer_updated"), callable_mp(this, &SkeletonModifier3D::_process_changed))) {
		p_old->disconnect(SNAME("mixer_updated"), callable_mp(this, &SkeletonModifier3D::_process_changed));
	}
	if (p_new && !p_new->is_connected(SNAME("mixer_updated"), callable_mp(this, &SkeletonModifier3D::_process_changed))) {
		p_new->connect(SNAME("mixer_updated"), callable_mp(this, &SkeletonModifier3D::_process_changed));
	}
}

/* Process */

void SkeletonModifier3D::set_active(bool p_active) {
	if (active == p_active) {
		return;
	}
	active = p_active;
	_process_changed();
	_set_active(active);
}

bool SkeletonModifier3D::is_active() const {
	return active;
}

void SkeletonModifier3D::_set_active(bool p_active) {
	//
}

void SkeletonModifier3D::set_callback_mode_process(AnimationMixer::AnimationCallbackModeProcess p_mode) {
	if (callback_mode_process == p_mode) {
		return;
	}
	callback_mode_process = p_mode;
	_process_changed();
}

AnimationMixer::AnimationCallbackModeProcess SkeletonModifier3D::get_callback_mode_process() const {
	return callback_mode_process;
}

void SkeletonModifier3D::process() {
	if (!is_inside_tree()) {
		return;
	}
	AnimationMixer *mixer = get_animation_mixer();
	if (mixer) {
		mixer->add_post_process(this);
	} else {
		if (callback_mode_process == AnimationMixer::ANIMATION_CALLBACK_MODE_PROCESS_IDLE) {
			process_modification(get_process_delta_time());
		} else if (callback_mode_process == AnimationMixer::ANIMATION_CALLBACK_MODE_PROCESS_PHYSICS) {
			process_modification(get_physics_process_delta_time());
		}
	}
}

void SkeletonModifier3D::_process_changed() {
	if (active) {
		AnimationMixer *mixer = get_animation_mixer();
		if (mixer) {
			callback_mode_process = mixer->get_callback_mode_process();
		}
		if (callback_mode_process == AnimationMixer::ANIMATION_CALLBACK_MODE_PROCESS_IDLE) {
			set_process_internal(true);
			set_physics_process_internal(false);
		} else if (callback_mode_process == AnimationMixer::ANIMATION_CALLBACK_MODE_PROCESS_PHYSICS) {
			set_process_internal(false);
			set_physics_process_internal(true);
		} else {
			set_process_internal(false);
			set_physics_process_internal(false);
		}
	} else {
		set_process_internal(false);
		set_physics_process_internal(false);
	}
}

void SkeletonModifier3D::advance(double p_time) {
	process_modification(p_time);
}

void SkeletonModifier3D::process_modification(double p_delta) {
	GDVIRTUAL_CALL(_process_modification, p_delta);
	_process_modification(p_delta);
	emit_signal(SNAME("modification_processed"));
}

void SkeletonModifier3D::_process_modification(double p_delta) {
	//
}

void SkeletonModifier3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_process_changed();
		}
			[[fallthrough]];
		case NOTIFICATION_PARENTED: {
			rebind();
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			if (active && callback_mode_process == AnimationMixer::ANIMATION_CALLBACK_MODE_PROCESS_IDLE) {
				process();
			}
		} break;
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (active && callback_mode_process == AnimationMixer::ANIMATION_CALLBACK_MODE_PROCESS_PHYSICS) {
				process();
			}
		} break;
	}
}

void SkeletonModifier3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_active", "active"), &SkeletonModifier3D::set_active);
	ClassDB::bind_method(D_METHOD("is_active"), &SkeletonModifier3D::is_active);

	ClassDB::bind_method(D_METHOD("set_external_skeleton", "path"), &SkeletonModifier3D::set_external_skeleton);
	ClassDB::bind_method(D_METHOD("get_external_skeleton"), &SkeletonModifier3D::get_external_skeleton);

	ClassDB::bind_method(D_METHOD("set_callback_mode_process", "mode"), &SkeletonModifier3D::set_callback_mode_process);
	ClassDB::bind_method(D_METHOD("get_callback_mode_process"), &SkeletonModifier3D::get_callback_mode_process);

	ClassDB::bind_method(D_METHOD("set_target_animation_mixer", "path"), &SkeletonModifier3D::set_target_animation_mixer);
	ClassDB::bind_method(D_METHOD("get_target_animation_mixer"), &SkeletonModifier3D::get_target_animation_mixer);

	ClassDB::bind_method(D_METHOD("advance", "delta"), &SkeletonModifier3D::advance);

	ClassDB::bind_method(D_METHOD("_process_modification", "delta"), &SkeletonModifier3D::_process_modification);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "active"), "set_active", "is_active");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "callback_mode_process", PROPERTY_HINT_ENUM, "Physics,Idle,Manual"), "set_callback_mode_process", "get_callback_mode_process");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_animation_mixer", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "AnimationMixer"), "set_target_animation_mixer", "get_target_animation_mixer");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "external_skeleton", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Skeleton3D"), "set_external_skeleton", "get_external_skeleton");

	ADD_SIGNAL(MethodInfo("modification_processed"));
	GDVIRTUAL_BIND(_process_modification, "delta");
}

SkeletonModifier3D::SkeletonModifier3D() {
}

#ifdef TOOLS_ENABLED
void SkeletonModifier3D::notify_rebind_required() {
	rebind(true);
}
#endif // TOOLS_ENABLED
