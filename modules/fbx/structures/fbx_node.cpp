/**************************************************************************/
/*  fbx_node.cpp                                                          */
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

#include "fbx_node.h"

void FBXNode::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_parent"), &FBXNode::get_parent);
	ClassDB::bind_method(D_METHOD("set_parent", "parent"), &FBXNode::set_parent);
	ClassDB::bind_method(D_METHOD("get_height"), &FBXNode::get_height);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &FBXNode::set_height);
	ClassDB::bind_method(D_METHOD("get_xform"), &FBXNode::get_xform);
	ClassDB::bind_method(D_METHOD("set_xform", "xform"), &FBXNode::set_xform);
	ClassDB::bind_method(D_METHOD("get_mesh"), &FBXNode::get_mesh);
	ClassDB::bind_method(D_METHOD("set_mesh", "mesh"), &FBXNode::set_mesh);
	ClassDB::bind_method(D_METHOD("get_camera"), &FBXNode::get_camera);
	ClassDB::bind_method(D_METHOD("set_camera", "camera"), &FBXNode::set_camera);
	ClassDB::bind_method(D_METHOD("get_skin"), &FBXNode::get_skin);
	ClassDB::bind_method(D_METHOD("set_skin", "skin"), &FBXNode::set_skin);
	ClassDB::bind_method(D_METHOD("get_skeleton"), &FBXNode::get_skeleton);
	ClassDB::bind_method(D_METHOD("set_skeleton", "skeleton"), &FBXNode::set_skeleton);
	ClassDB::bind_method(D_METHOD("get_position"), &FBXNode::get_position);
	ClassDB::bind_method(D_METHOD("set_position", "position"), &FBXNode::set_position);
	ClassDB::bind_method(D_METHOD("get_rotation"), &FBXNode::get_rotation);
	ClassDB::bind_method(D_METHOD("set_rotation", "rotation"), &FBXNode::set_rotation);
	ClassDB::bind_method(D_METHOD("get_scale"), &FBXNode::get_scale);
	ClassDB::bind_method(D_METHOD("set_scale", "scale"), &FBXNode::set_scale);
	ClassDB::bind_method(D_METHOD("get_children"), &FBXNode::get_children);
	ClassDB::bind_method(D_METHOD("set_children", "children"), &FBXNode::set_children);
	ClassDB::bind_method(D_METHOD("get_additional_data", "extension_name"), &FBXNode::get_additional_data);
	ClassDB::bind_method(D_METHOD("set_additional_data", "extension_name", "additional_data"), &FBXNode::set_additional_data);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "parent"), "set_parent", "get_parent"); // GLTFNodeIndex
	ADD_PROPERTY(PropertyInfo(Variant::INT, "height"), "set_height", "get_height"); // int
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM3D, "xform"), "set_xform", "get_xform"); // Transform3D
	ADD_PROPERTY(PropertyInfo(Variant::INT, "camera"), "set_camera", "get_camera"); // GLTFCameraIndex
	ADD_PROPERTY(PropertyInfo(Variant::INT, "skin"), "set_skin", "get_skin"); // GLTFSkinIndex
	ADD_PROPERTY(PropertyInfo(Variant::INT, "skeleton"), "set_skeleton", "get_skeleton"); // GLTFSkeletonIndex
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "position"), "set_position", "get_position"); // Vector3
	ADD_PROPERTY(PropertyInfo(Variant::QUATERNION, "rotation"), "set_rotation", "get_rotation"); // Quaternion
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "scale"), "set_scale", "get_scale"); // Vector3
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "children"), "set_children", "get_children"); // Vector<int>
}

GLTFNodeIndex FBXNode::get_parent() {
	return parent;
}

void FBXNode::set_parent(GLTFNodeIndex p_parent) {
	parent = p_parent;
}

int FBXNode::get_height() {
	return height;
}

void FBXNode::set_height(int p_height) {
	height = p_height;
}

Transform3D FBXNode::get_xform() {
	return xform;
}

void FBXNode::set_xform(Transform3D p_xform) {
	xform = p_xform;
}

GLTFMeshIndex FBXNode::get_mesh() {
	return mesh;
}

void FBXNode::set_mesh(GLTFMeshIndex p_mesh) {
	mesh = p_mesh;
}

GLTFCameraIndex FBXNode::get_camera() {
	return camera;
}

void FBXNode::set_camera(GLTFCameraIndex p_camera) {
	camera = p_camera;
}

GLTFAnimationIndex FBXNode::get_light() {
	return light;
}

void FBXNode::set_light(GLTFAnimationIndex p_light) {
	light = p_light;
}

GLTFSkinIndex FBXNode::get_skin() {
	return skin;
}

void FBXNode::set_skin(GLTFSkinIndex p_skin) {
	skin = p_skin;
}

GLTFSkeletonIndex FBXNode::get_skeleton() {
	return skeleton;
}

void FBXNode::set_skeleton(GLTFSkeletonIndex p_skeleton) {
	skeleton = p_skeleton;
}

Vector3 FBXNode::get_position() {
	return position;
}

void FBXNode::set_position(Vector3 p_position) {
	position = p_position;
}

Quaternion FBXNode::get_rotation() {
	return rotation;
}

void FBXNode::set_rotation(Quaternion p_rotation) {
	rotation = p_rotation;
}

Vector3 FBXNode::get_scale() {
	return scale;
}

void FBXNode::set_scale(Vector3 p_scale) {
	scale = p_scale;
}

Vector<int> FBXNode::get_children() {
	return children;
}

void FBXNode::set_children(Vector<int> p_children) {
	children = p_children;
}

Variant FBXNode::get_additional_data(const StringName &p_extension_name) {
	return additional_data[p_extension_name];
}

void FBXNode::set_additional_data(const StringName &p_extension_name, Variant p_additional_data) {
	additional_data[p_extension_name] = p_additional_data;
}
