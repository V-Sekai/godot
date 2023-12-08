/**************************************************************************/
/*  fbx_mesh.h                                                            */
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

#ifndef FBX_MESH_H
#define FBX_MESH_H

#include "../fbx_defines.h"

#include "scene/resources/importer_mesh.h"

class FBXMesh : public Resource {
	GDCLASS(FBXMesh, Resource);

private:
	Ref<ImporterMesh> mesh;
	Vector<float> blend_weights;
	Vector<int> blend_channels;
	TypedArray<Material> instance_materials;

protected:
	static void _bind_methods();

public:
	Ref<ImporterMesh> get_mesh();
	void set_mesh(Ref<ImporterMesh> p_mesh);
	Vector<float> get_blend_weights();
	void set_blend_weights(Vector<float> p_blend_weights);
	Vector<int> get_blend_channels();
	void set_blend_channels(Vector<int> p_blend_channels);
	TypedArray<Material> get_instance_materials();
	void set_instance_materials(TypedArray<Material> p_instance_materials);
};

#endif // FBX_MESH_H
