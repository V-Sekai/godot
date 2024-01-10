/**************************************************************************/
/*  fbx_skeleton.h                                                        */
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

#ifndef FBX_SKELETON_H
#define FBX_SKELETON_H

#include "../fbx_defines.h"

#include "core/io/resource.h"

class FBXSkeleton : public Resource {
	GDCLASS(FBXSkeleton, Resource);
	friend class FBXDocument;
	friend class SkinTool;

private:
	// The *synthesized* skeletons joints
	Vector<FBXNodeIndex> joints;

	// The roots of the skeleton. If there are multiple, each root must have the
	// same parent (ie roots are siblings)
	Vector<FBXNodeIndex> roots;

	// The created Skeleton3D for the scene
	Skeleton3D *godot_skeleton = nullptr;

	// Set of unique bone names for the skeleton
	HashSet<String> unique_names;

	HashMap<int32_t, FBXNodeIndex> godot_bone_node;

	Vector<BoneAttachment3D *> bone_attachments;

protected:
	static void _bind_methods();

public:
	Vector<FBXNodeIndex> get_joints();
	void set_joints(Vector<FBXNodeIndex> p_joints);

	Vector<FBXNodeIndex> get_roots();
	void set_roots(Vector<FBXNodeIndex> p_roots);

	Skeleton3D *get_godot_skeleton();

	TypedArray<String> get_unique_names();
	void set_unique_names(TypedArray<String> p_unique_names);

	Dictionary get_godot_bone_node();
	void set_godot_bone_node(Dictionary p_indict);

	BoneAttachment3D *get_bone_attachment(int idx);

	int32_t get_bone_attachment_count();
};

#endif // FBX_SKELETON_H
