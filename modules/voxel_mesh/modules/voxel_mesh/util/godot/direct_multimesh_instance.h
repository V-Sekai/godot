/**************************************************************************/
/*  direct_multimesh_instance.h                                           */
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

#ifndef DIRECT_MULTIMESH_INSTANCE_H
#define DIRECT_MULTIMESH_INSTANCE_H

#include "../span.h"

#include "core/variant/variant.h"
#include "scene/3d/multimesh_instance_3d.h"
#include <core/templates/rid.h>

class World;

// Thin wrapper around VisualServer multimesh instance API
class DirectMultiMeshInstance {
public:
	DirectMultiMeshInstance();
	~DirectMultiMeshInstance();

	void create();
	void destroy();
	bool is_valid() const;
	void set_world(World3D *world);
	void set_multimesh(Ref<MultiMesh> multimesh);
	Ref<MultiMesh> get_multimesh() const;
	void set_transform(Transform3D world_transform);
	void set_visible(bool visible);
	void set_material_override(Ref<BaseMaterial3D> material);
	void set_cast_shadows_setting(RenderingServer::ShadowCastingSetting mode);

	static PackedFloat32Array
	make_transform_3d_bulk_array(Span<const Transform3D> transforms);

private:
	RID _multimesh_instance;
	Ref<MultiMesh> _multimesh;
};

#endif // DIRECT_MULTIMESH_INSTANCE_H
