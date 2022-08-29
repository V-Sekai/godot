/*************************************************************************/
/*  skeleton_modification_3d.h                                           */
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

#ifndef SKELETON_MODIFICATION_3D_H
#define SKELETON_MODIFICATION_3D_H

#include "core/string/node_path.h"
#include "scene/3d/skeleton_3d.h"

class SkeletonModification3D : public Node {
	GDCLASS(SkeletonModification3D, Node);

private:
	static void _bind_methods();

	bool enabled = true;
	bool skeleton_change_queued = true;
	PackedStringArray bone_property_suffixes;
	mutable Variant cached_skeleton;
	mutable String bone_name_list;
	uint64_t cached_skeleton_version = 0;
	NodePath skeleton_path = NodePath("..");

protected:
	bool _cache_bone(int &bone_cache, const String &target_bone_name) const;
	bool _cache_target(Variant &cache, const NodePath &target_node_path, const String &target_bone_name) const;

public:
	enum { UNCACHED_BONE_IDX = -2 };

	void set_enabled(bool p_enabled);
	bool get_enabled() const;

	NodePath get_skeleton_path() const;
	void set_skeleton_path(NodePath p_path);
	Skeleton3D *get_skeleton() const;
	void set_bone_property_suffixes(PackedStringArray suffixes);

	void _validate_property(PropertyInfo &property) const override;
	void _notification(int32_t p_what);

	virtual void skeleton_changed(Skeleton3D *skeleton);
	GDVIRTUAL1(_skeleton_changed, Skeleton3D*);
	virtual void execute(real_t delta);
	GDVIRTUAL1(_execute, real_t);
	virtual bool is_bone_property(String property_name) const;
	TypedArray<String> get_configuration_warnings() const override;

	int resolve_bone(const String &target_bone_name) const;
	Variant resolve_target(const NodePath &target_node_path, const String &target_bone_name) const;
	Transform3D get_target_transform(Variant resolved_target) const;
	Quaternion get_target_quaternion(Variant resolved_target) const;

	SkeletonModification3D() {}
};

#endif // SKELETON_MODIFICATION_3D_H
