/**************************************************************************/
/*  bone_attachment_3d.h                                                  */
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

#ifndef BONE_ATTACHMENT_3D_H
#define BONE_ATTACHMENT_3D_H

#include "scene/3d/skeleton_modifier_3d.h"
#ifdef TOOLS_ENABLED
#include "scene/resources/bone_map.h"
#endif // TOOLS_ENABLED

class BoneAttachment3D : public SkeletonModifier3D {
	GDCLASS(BoneAttachment3D, SkeletonModifier3D);

	bool bound = false;
	String bone_name;
	int bone_idx = -1;

	bool override_pose = false;
	void _override_pose();
	void _retrieve_pose();

protected:
	void _validate_property(PropertyInfo &p_property) const;
	void _notification(int p_what);

	virtual void _process_modification(double p_delta) override;

	static void _bind_methods();

	virtual ObjectID _update_skeleton_path_extend() override;

	virtual void _skeleton_changed(Skeleton3D *p_old, Skeleton3D *p_new) override;

	void _update_bone();

#ifndef DISABLE_DEPRECATED
	void _set_use_external_skeleton_bind_compat_87888(bool use_external_skeleton);
	bool _get_use_external_skeleton_bind_compat_87888() const;
	static void _bind_compatibility_methods();
#endif // DISABLE_DEPRECATED

public:
#ifdef TOOLS_ENABLED
	virtual void notify_skeleton_bones_renamed(Node *p_base_scene, Skeleton3D *p_skeleton, Dictionary p_rename_map);
#endif // TOOLS_ENABLED

	virtual PackedStringArray get_configuration_warnings() const override;

	void set_bone_name(const String &p_name);
	String get_bone_name() const;

	void set_bone_idx(const int &p_idx);
	int get_bone_idx() const;

	void set_override_pose(bool p_override);
	bool get_override_pose() const;

#ifndef DISABLE_DEPRECATED
	virtual void on_bone_pose_update(int p_bone_index);
#endif // DISABLE_DEPRECATED

	BoneAttachment3D();
};

#endif // BONE_ATTACHMENT_3D_H
