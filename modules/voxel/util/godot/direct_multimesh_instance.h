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

#pragma once

#include "../containers/span.h"
#include "../math/color8.h"
#include "../math/transform3f.h"
#include "../non_copyable.h"
#include "classes/geometry_instance_3d.h"
#include "classes/multimesh.h"
#include "classes/rendering_server.h"
#include "macros.h"

ZN_GODOT_FORWARD_DECLARE(class World3D);
ZN_GODOT_FORWARD_DECLARE(class Material);

namespace zylann::godot {

// Thin wrapper around VisualServer multimesh instance API
class DirectMultiMeshInstance : public zylann::NonCopyable {
public:
	DirectMultiMeshInstance();
	DirectMultiMeshInstance(DirectMultiMeshInstance &&src);
	~DirectMultiMeshInstance();

	void create();
	void destroy();
	bool is_valid() const;
	void set_world(World3D *world);
	void set_multimesh(Ref<MultiMesh> multimesh);
	Ref<MultiMesh> get_multimesh() const;
	void set_transform(Transform3D world_transform);
	void set_visible(bool visible);
	void set_material_override(Ref<Material> material);
	void set_cast_shadows_setting(RenderingServer::ShadowCastingSetting mode);
	void set_shader_instance_parameter(const StringName &key, const Variant &value);
	void set_render_layer(int render_layer);
	void set_gi_mode(GeometryInstance3D::GIMode mode);
	void set_interpolated(const bool enabled);

	void operator=(DirectMultiMeshInstance &&src);

	static void make_transform_3d_bulk_array(Span<const Transform3D> transforms, PackedFloat32Array &bulk_array);
	static void make_transform_3d_bulk_array(Span<const Transform3f> transforms, PackedFloat32Array &bulk_array);

	struct TransformAndColor8 {
		Transform3D transform;
		zylann::Color8 color;
	};

	static void make_transform_and_color8_3d_bulk_array(
			Span<const TransformAndColor8> data,
			PackedFloat32Array &bulk_array
	);

	struct TransformAndColor32 {
		Transform3D transform;
		Color color;
	};

	static void make_transform_and_color32_3d_bulk_array(
			Span<const TransformAndColor32> data,
			PackedFloat32Array &bulk_array
	);

private:
	RID _multimesh_instance;
	Ref<MultiMesh> _multimesh;
};

} // namespace zylann::godot
