/**************************************************************************/
/*  skeleton_modifier_3d.h                                                */
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

#ifndef SKELETON_MODIFIER_3D_H
#define SKELETON_MODIFIER_3D_H

#include "scene/3d/node_3d.h"

#include "scene/3d/skeleton_3d.h"
#include "scene/animation/animation_mixer.h"

class SkeletonModifier3D : public Node3D {
	GDCLASS(SkeletonModifier3D, Node3D);

	void rebind(bool p_path_changed = false);
	void process_modification(double p_delta);

protected:
	bool active = true;
	AnimationMixer::AnimationCallbackModeProcess callback_mode_process = AnimationMixer::ANIMATION_CALLBACK_MODE_PROCESS_IDLE;
	NodePath external_skeleton;
	NodePath target_animation_mixer;

	// Cache them for the performance reason since finding node with NodePath is slow.
	ObjectID skeleton_id;
	ObjectID animation_mixer_id;

	void _update_skeleton(bool p_path_changed = false);
	void _update_animation_mixer(bool p_path_changed = false);

	void _update_skeleton_path();
	virtual ObjectID _update_skeleton_path_extend();
	void _update_animation_mixer_path();

	virtual void _skeleton_changed(Skeleton3D *p_old, Skeleton3D *p_new);
	virtual void _animation_mixer_changed(AnimationMixer *p_old, AnimationMixer *p_new);
	virtual void _rebind(bool p_force = false);

	void _validate_property(PropertyInfo &p_property) const;
	void _notification(int p_what);
	static void _bind_methods();

	virtual void _set_active(bool p_active);

	virtual void _process_modification(double p_delta);
	GDVIRTUAL1(_process_modification, double);

	void _update_process_mode();

public:
	virtual PackedStringArray get_configuration_warnings() const override;

	void advance(double p_time);

	void set_active(bool p_active);
	bool is_active() const;

	Skeleton3D *get_skeleton() const;
	void set_external_skeleton(const NodePath &p_path);
	NodePath get_external_skeleton() const;

	AnimationMixer *get_animation_mixer() const;
	void set_target_animation_mixer(const NodePath &p_path);
	NodePath get_target_animation_mixer() const;

	void set_callback_mode_process(AnimationMixer::AnimationCallbackModeProcess p_mode);
	AnimationMixer::AnimationCallbackModeProcess get_callback_mode_process() const;

#ifdef TOOLS_ENABLED
	virtual void notify_rebind_required();
#endif

	SkeletonModifier3D();
};

#endif // SKELETON_MODIFIER_3D_H
