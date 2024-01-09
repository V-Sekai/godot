/**************************************************************************/
/*  fbx_skeleton.cpp                                                      */
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

#include "fbx_skeleton.h"

#include "../fbx_template_convert.h"

#include "scene/3d/bone_attachment_3d.h"

void FBXSkeleton::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_joints"), &FBXSkeleton::get_joints);
	ClassDB::bind_method(D_METHOD("set_joints", "joints"), &FBXSkeleton::set_joints);
	ClassDB::bind_method(D_METHOD("get_roots"), &FBXSkeleton::get_roots);
	ClassDB::bind_method(D_METHOD("set_roots", "roots"), &FBXSkeleton::set_roots);
	ClassDB::bind_method(D_METHOD("get_godot_skeleton"), &FBXSkeleton::get_godot_skeleton);
	ClassDB::bind_method(D_METHOD("get_unique_names"), &FBXSkeleton::get_unique_names);
	ClassDB::bind_method(D_METHOD("set_unique_names", "unique_names"), &FBXSkeleton::set_unique_names);
	ClassDB::bind_method(D_METHOD("get_godot_bone_node"), &FBXSkeleton::get_godot_bone_node);
	ClassDB::bind_method(D_METHOD("set_godot_bone_node", "godot_bone_node"), &FBXSkeleton::set_godot_bone_node);
	ClassDB::bind_method(D_METHOD("get_bone_attachment_count"), &FBXSkeleton::get_bone_attachment_count);
	ClassDB::bind_method(D_METHOD("get_bone_attachment", "idx"), &FBXSkeleton::get_bone_attachment);

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "joints"), "set_joints", "get_joints"); // Vector<GLTFNodeIndex>
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "roots"), "set_roots", "get_roots"); // Vector<GLTFNodeIndex>
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "unique_names", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_INTERNAL | PROPERTY_USAGE_EDITOR), "set_unique_names", "get_unique_names"); // Set<String>
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "godot_bone_node", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_INTERNAL | PROPERTY_USAGE_EDITOR), "set_godot_bone_node", "get_godot_bone_node"); // RBMap<int32_t,
}

Vector<GLTFNodeIndex> FBXSkeleton::get_joints() {
	return joints;
}

void FBXSkeleton::set_joints(Vector<GLTFNodeIndex> p_joints) {
	joints = p_joints;
}

Vector<GLTFNodeIndex> FBXSkeleton::get_roots() {
	return roots;
}

void FBXSkeleton::set_roots(Vector<GLTFNodeIndex> p_roots) {
	roots = p_roots;
}

Skeleton3D *FBXSkeleton::get_godot_skeleton() {
	return godot_skeleton;
}

TypedArray<String> FBXSkeleton::get_unique_names() {
	return FBXTemplateConvert::to_array(unique_names);
}

void FBXSkeleton::set_unique_names(TypedArray<String> p_unique_names) {
	FBXTemplateConvert::set_from_array(unique_names, p_unique_names);
}

Dictionary FBXSkeleton::get_godot_bone_node() {
	return FBXTemplateConvert::to_dict(godot_bone_node);
}

void FBXSkeleton::set_godot_bone_node(Dictionary p_indict) {
	FBXTemplateConvert::set_from_dict(godot_bone_node, p_indict);
}

BoneAttachment3D *FBXSkeleton::get_bone_attachment(int idx) {
	ERR_FAIL_INDEX_V(idx, bone_attachments.size(), nullptr);
	return bone_attachments[idx];
}

int32_t FBXSkeleton::get_bone_attachment_count() {
	return bone_attachments.size();
}
