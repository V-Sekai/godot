/**************************************************************************/
/*  direct_multimesh_instance.cpp                                         */
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

#include "direct_multimesh_instance.h"
#include "../profiling.h"

#include <scene/resources/world_3d.h>

DirectMultiMeshInstance::DirectMultiMeshInstance() {}

DirectMultiMeshInstance::~DirectMultiMeshInstance() {
	destroy();
}

bool DirectMultiMeshInstance::is_valid() const {
	return _multimesh_instance.is_valid();
}

void DirectMultiMeshInstance::create() {
	ERR_FAIL_COND(_multimesh_instance.is_valid());
	RenderingServer &vs = *RenderingServer::get_singleton();
	_multimesh_instance = vs.instance_create();
	vs.instance_set_visible(_multimesh_instance, true); // TODO Is it needed?
}

void DirectMultiMeshInstance::destroy() {
	if (_multimesh_instance.is_valid()) {
		RenderingServer &vs = *RenderingServer::get_singleton();
		vs.free(_multimesh_instance);
		_multimesh_instance = RID();
		_multimesh.unref();
	}
}

void DirectMultiMeshInstance::set_world(World3D *world) {
	ERR_FAIL_COND(!_multimesh_instance.is_valid());
	RenderingServer &vs = *RenderingServer::get_singleton();
	if (world != nullptr) {
		vs.instance_set_scenario(_multimesh_instance, world->get_scenario());
	} else {
		vs.instance_set_scenario(_multimesh_instance, RID());
	}
}

void DirectMultiMeshInstance::set_multimesh(Ref<MultiMesh> multimesh) {
	ERR_FAIL_COND(!_multimesh_instance.is_valid());
	RenderingServer &vs = *RenderingServer::get_singleton();
	if (multimesh.is_valid()) {
		if (_multimesh != multimesh) {
			vs.instance_set_base(_multimesh_instance, multimesh->get_rid());
		}
	} else {
		vs.instance_set_base(_multimesh_instance, RID());
	}
	_multimesh = multimesh;
}

Ref<MultiMesh> DirectMultiMeshInstance::get_multimesh() const {
	return _multimesh;
}

void DirectMultiMeshInstance::set_transform(Transform3D world_transform) {
	VOXEL_PROFILE_SCOPE();
	ERR_FAIL_COND(!_multimesh_instance.is_valid());
	RenderingServer &vs = *RenderingServer::get_singleton();
	vs.instance_set_transform(_multimesh_instance, world_transform);
}

void DirectMultiMeshInstance::set_visible(bool visible) {
	ERR_FAIL_COND(!_multimesh_instance.is_valid());
	RenderingServer &vs = *RenderingServer::get_singleton();
	vs.instance_set_visible(_multimesh_instance, visible);
}

void DirectMultiMeshInstance::set_material_override(
		Ref<BaseMaterial3D> material) {
	ERR_FAIL_COND(!_multimesh_instance.is_valid());
	RenderingServer &vs = *RenderingServer::get_singleton();
	if (material.is_valid()) {
		vs.instance_geometry_set_material_override(_multimesh_instance,
				material->get_rid());
	} else {
		vs.instance_geometry_set_material_override(_multimesh_instance, RID());
	}
}

void DirectMultiMeshInstance::set_cast_shadows_setting(
		RenderingServer::ShadowCastingSetting mode) {
	ERR_FAIL_COND(!_multimesh_instance.is_valid());
	RenderingServer &vs = *RenderingServer::get_singleton();
	vs.instance_geometry_set_cast_shadows_setting(_multimesh_instance, mode);
}

PackedFloat32Array DirectMultiMeshInstance::make_transform_3d_bulk_array(
		Span<const Transform3D> transforms) {
	VOXEL_PROFILE_SCOPE();

	PackedFloat32Array bulk_array;
	bulk_array.resize(transforms.size() * 12);
	CRASH_COND(transforms.size() * sizeof(Transform3D) / sizeof(float) !=
			static_cast<size_t>(bulk_array.size()));

	// memcpy(w.ptr(), _transform_cache.data(), bulk_array.size() *
	// sizeof(float));
	//  Nope, you can't memcpy that, nonono. It's said to be for performance, but
	//  doesnt specify why.

	for (size_t i = 0; i < transforms.size(); ++i) {
		float *ptr = bulk_array.ptrw() + 12 * i;
		const Transform3D &t = transforms[i];

		ptr[0] = t.basis.get_column(0).x;
		ptr[1] = t.basis.get_column(0).y;
		ptr[2] = t.basis.get_column(0).z;
		ptr[3] = t.origin.x;

		ptr[4] = t.basis.get_column(1).x;
		ptr[5] = t.basis.get_column(1).y;
		ptr[6] = t.basis.get_column(1).z;
		ptr[7] = t.origin.y;

		ptr[8] = t.basis.get_column(2).x;
		ptr[9] = t.basis.get_column(2).y;
		ptr[10] = t.basis.get_column(2).z;
		ptr[11] = t.origin.z;
	}

	return bulk_array;
}
