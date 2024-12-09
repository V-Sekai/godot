/**************************************************************************/
/*  ik_node_3d.cpp                                                        */
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

#include "ik_node_3d.h"

void IKConstraintNode3D::_propagate_transform_changed() {
	Vector<Ref<IKConstraintNode3D>> to_remove;

	for (Ref<IKConstraintNode3D> transform : children) {
		if (transform.is_null()) {
			to_remove.push_back(transform);
		} else {
			transform->_propagate_transform_changed();
		}
	}

	for (Ref<IKConstraintNode3D> transform : to_remove) {
		children.erase(transform);
	}

	dirty |= DIRTY_GLOBAL;
}

void IKConstraintNode3D::_update_local_transform() const {
	local_transform.basis = rotation.scaled(scale);
	dirty &= ~DIRTY_LOCAL;
}

void IKConstraintNode3D::rotate_local_with_global(const Basis &p_basis, bool p_propagate) {
	if (parent.get_ref().is_null()) {
		return;
	}
	Ref<IKConstraintNode3D> parent_ik_node = parent.get_ref();
	const Basis &new_rot = parent_ik_node->get_global_transform().basis;
	local_transform.basis = new_rot.inverse() * p_basis * new_rot * local_transform.basis;
	dirty |= DIRTY_GLOBAL;
	if (p_propagate) {
		_propagate_transform_changed();
	}
}

void IKConstraintNode3D::set_transform(const Transform3D &p_transform) {
	if (local_transform != p_transform) {
		local_transform = p_transform;
		dirty |= DIRTY_VECTORS;
		_propagate_transform_changed();
	}
}

void IKConstraintNode3D::set_global_transform(const Transform3D &p_transform) {
	Ref<IKConstraintNode3D> ik_node = parent.get_ref();
	Transform3D xform = ik_node.is_valid() ? ik_node->get_global_transform().affine_inverse() * p_transform : p_transform;
	local_transform = xform;
	dirty |= DIRTY_VECTORS;
	_propagate_transform_changed();
}

Transform3D IKConstraintNode3D::get_transform() const {
	if (dirty & DIRTY_LOCAL) {
		_update_local_transform();
	}

	return local_transform;
}

Transform3D IKConstraintNode3D::get_global_transform() const {
	if (dirty & DIRTY_GLOBAL) {
		if (dirty & DIRTY_LOCAL) {
			_update_local_transform();
		}
		Ref<IKConstraintNode3D> ik_node = parent.get_ref();
		if (ik_node.is_valid()) {
			global_transform = ik_node->get_global_transform() * local_transform;
		} else {
			global_transform = local_transform;
		}

		if (disable_scale) {
			global_transform.basis.orthogonalize();
		}

		dirty &= ~DIRTY_GLOBAL;
	}

	return global_transform;
}

void IKConstraintNode3D::set_disable_scale(bool p_enabled) {
	disable_scale = p_enabled;
}

bool IKConstraintNode3D::is_scale_disabled() const {
	return disable_scale;
}

void IKConstraintNode3D::set_parent(Ref<IKConstraintNode3D> p_parent) {
	if (p_parent.is_valid()) {
		p_parent->children.erase(this);
	}
	parent.set_ref(p_parent);
	if (p_parent.is_valid()) {
		p_parent->children.push_back(this);
	}
	_propagate_transform_changed();
}

Ref<IKConstraintNode3D> IKConstraintNode3D::get_parent() const {
	return parent.get_ref();
}

Vector3 IKConstraintNode3D::to_local(const Vector3 &p_global) const {
	return get_global_transform().affine_inverse().xform(p_global);
}

Vector3 IKConstraintNode3D::to_global(const Vector3 &p_local) const {
	return get_global_transform().xform(p_local);
}

IKConstraintNode3D::~IKConstraintNode3D() {
	cleanup();
}

void IKConstraintNode3D::_notification(int p_what) {
	if (p_what == NOTIFICATION_PREDELETE) {
		cleanup();
	}
}
void IKConstraintNode3D::cleanup() {
	for (Ref<IKConstraintNode3D> &child : children) {
		child->set_parent(Ref<IKConstraintNode3D>());
	}
}
