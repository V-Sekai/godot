/**************************************************************************/
/*  fbx_node.h                                                            */
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

#ifndef FBX_NODE_H
#define FBX_NODE_H

#include "../fbx_defines.h"

#include "core/io/resource.h"

class FBXNode : public Resource {
	GDCLASS(FBXNode, Resource);
	friend class FBXDocument;

private:
	// matrices need to be transformed to this
	FBXNodeIndex parent = -1;
	int height = -1;
	Transform3D xform;
	FBXMeshIndex mesh = -1;
	FBXSkinIndex skin = -1;
	FBXSkeletonIndex skeleton = -1;
	bool joint = false;
	Vector3 position;
	Quaternion rotation;
	Vector3 scale = Vector3(1, 1, 1);
	Vector<int> children;
	Dictionary additional_data;

protected:
	static void _bind_methods();

public:
	FBXNodeIndex get_parent();
	void set_parent(FBXNodeIndex p_parent);

	int get_height();
	void set_height(int p_height);

	Transform3D get_xform();
	void set_xform(Transform3D p_xform);

	FBXMeshIndex get_mesh();
	void set_mesh(FBXMeshIndex p_mesh);

	FBXSkinIndex get_skin();
	void set_skin(FBXSkinIndex p_skin);

	FBXSkeletonIndex get_skeleton();
	void set_skeleton(FBXSkeletonIndex p_skeleton);

	Vector3 get_position();
	void set_position(Vector3 p_position);

	Quaternion get_rotation();
	void set_rotation(Quaternion p_rotation);

	Vector3 get_scale();
	void set_scale(Vector3 p_scale);

	Vector<int> get_children();
	void set_children(Vector<int> p_children);

	Variant get_additional_data(const StringName &p_extension_name);
	void set_additional_data(const StringName &p_extension_name, Variant p_additional_data);
};

#endif // FBX_NODE_H
