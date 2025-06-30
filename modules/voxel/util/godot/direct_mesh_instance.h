/**************************************************************************/
/*  direct_mesh_instance.h                                                */
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

#pragma once

#include "../non_copyable.h"
#include "classes/geometry_instance_3d.h"
#include "classes/mesh.h"
#include "classes/rendering_server.h"
#include "macros.h"

ZN_GODOT_FORWARD_DECLARE(class World3D);

namespace zylann::godot {

// Thin wrapper around VisualServer mesh instance API
class DirectMeshInstance : public NonCopyable {
public:
	DirectMeshInstance();
	DirectMeshInstance(DirectMeshInstance &&src);
	~DirectMeshInstance();

	bool is_valid() const;
	void create();
	void destroy();
	void set_world(World3D *world);
	void set_transform(Transform3D world_transform);
	void set_mesh(Ref<Mesh> mesh);
	void set_material_override(Ref<Material> material);
	void set_visible(bool visible);
	void set_cast_shadows_setting(RenderingServer::ShadowCastingSetting mode);
	void set_shader_instance_parameter(StringName key, Variant value);
	void set_gi_mode(GeometryInstance3D::GIMode mode);
	void set_render_layers_mask(int mask);
	void set_interpolated(const bool enabled);

	Ref<Mesh> get_mesh() const;
	const Mesh *get_mesh_ptr() const;

	// void move_to(DirectMeshInstance &dst);

	void operator=(DirectMeshInstance &&src);

private:
	RID _mesh_instance;
	Ref<Mesh> _mesh;
};

} // namespace zylann::godot
