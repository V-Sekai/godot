/**************************************************************************/
/*  gltf_node.h                                                           */
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

#ifndef GLTF_NODE_H
#define GLTF_NODE_H

#include "../gltf_defines.h"

#include "core/io/resource.h"

class GLTFNode : public Resource {
	GDCLASS(GLTFNode, Resource);
	friend class GLTFDocument;
	friend class SkinTool;
	friend class FBXDocument;

private:
	// matrices need to be transformed to this
	GLTFNodeIndex parent = -1;
	int height = -1;
	Transform3D xform;
	GLTFMeshIndex mesh = -1;
	GLTFCameraIndex camera = -1;
	GLTFSkinIndex skin = -1;
	GLTFSkeletonIndex skeleton = -1;
	bool joint = false;
	Vector3 position;
	Quaternion rotation;
	Vector3 scale = Vector3(1, 1, 1);
	Vector<int> children;
	GLTFLightIndex light = -1;
	Dictionary additional_data;

protected:
	static void _bind_methods();

public:
	GLTFNodeIndex get_parent();
	void set_parent(GLTFNodeIndex p_parent);

	int get_height();
	void set_height(int p_height);

	Transform3D get_xform();
	void set_xform(Transform3D p_xform);

	GLTFMeshIndex get_mesh();
	void set_mesh(GLTFMeshIndex p_mesh);

	GLTFCameraIndex get_camera();
	void set_camera(GLTFCameraIndex p_camera);

	GLTFSkinIndex get_skin();
	void set_skin(GLTFSkinIndex p_skin);

	GLTFSkeletonIndex get_skeleton();
	void set_skeleton(GLTFSkeletonIndex p_skeleton);

	Vector3 get_position();
	void set_position(Vector3 p_position);

	Quaternion get_rotation();
	void set_rotation(Quaternion p_rotation);

	Vector3 get_scale();
	void set_scale(Vector3 p_scale);

	Vector<int> get_children();
	void set_children(Vector<int> p_children);

	GLTFLightIndex get_light();
	void set_light(GLTFLightIndex p_light);

	Variant get_additional_data(const StringName &p_extension_name);
	void set_additional_data(const StringName &p_extension_name, Variant p_additional_data);

	Dictionary to_dictionary() const {
		Dictionary dict;
		dict["parent"] = parent;
		dict["height"] = height;
		dict["xform"] = xform;
		dict["mesh"] = mesh;
		dict["camera"] = camera;
		dict["light"] = light;
		dict["skin"] = skin;
		dict["skeleton"] = skeleton;
		dict["joint"] = joint;
		dict["position"] = position;
		dict["rotation"] = rotation;
		dict["scale"] = scale;

		Array children_array;
		for (int i = 0; i < children.size(); ++i) {
			children_array.push_back(children[i]);
		}
		dict["children"] = children_array;

		dict["additional_data"] = additional_data;

		return dict;
	}
	Error from_dictionary(const Dictionary &dict) {
		ERR_FAIL_COND_V(!dict.has("parent"), ERR_INVALID_DATA);
		parent = dict["parent"];

		ERR_FAIL_COND_V(!dict.has("height"), ERR_INVALID_DATA);
		height = dict["height"];

		ERR_FAIL_COND_V(!dict.has("xform"), ERR_INVALID_DATA);
		xform = dict["xform"];

		ERR_FAIL_COND_V(!dict.has("mesh"), ERR_INVALID_DATA);
		mesh = dict["mesh"];

		ERR_FAIL_COND_V(!dict.has("camera"), ERR_INVALID_DATA);
		camera = dict["camera"];

		ERR_FAIL_COND_V(!dict.has("light"), ERR_INVALID_DATA);
		light = dict["light"];

		ERR_FAIL_COND_V(!dict.has("skin"), ERR_INVALID_DATA);
		skin = dict["skin"];

		ERR_FAIL_COND_V(!dict.has("skeleton"), ERR_INVALID_DATA);
		skeleton = dict["skeleton"];

		ERR_FAIL_COND_V(!dict.has("joint"), ERR_INVALID_DATA);
		joint = dict["joint"];

		ERR_FAIL_COND_V(!dict.has("position"), ERR_INVALID_DATA);
		position = dict["position"];

		ERR_FAIL_COND_V(!dict.has("rotation"), ERR_INVALID_DATA);
		rotation = dict["rotation"];

		ERR_FAIL_COND_V(!dict.has("scale"), ERR_INVALID_DATA);
		scale = dict["scale"];

		if (dict.has("children")) {
			Array children_array = dict["children"];
			children.clear();
			for (int i = 0; i < children_array.size(); ++i) {
				children.push_back(children_array[i]);
			}
		}

		if (dict.has("additional_data")) {
			additional_data = dict["additional_data"];
		}

		return OK;
	}
};

#endif // GLTF_NODE_H
