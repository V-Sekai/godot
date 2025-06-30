/**************************************************************************/
/*  voxel_instancer_rigidbody.cpp                                         */
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

#include "voxel_instancer_rigidbody.h"

namespace zylann::voxel {

VoxelInstancerRigidBody::VoxelInstancerRigidBody() {
	set_freeze_mode(RigidBody3D::FREEZE_MODE_STATIC);
	set_freeze_enabled(true);
}

int VoxelInstancerRigidBody::get_library_item_id() const {
	ERR_FAIL_COND_V(_parent == nullptr, -1);
	return _parent->get_library_item_id_from_render_block_index(_render_block_index);
}

void VoxelInstancerRigidBody::_notification(int p_what) {
	switch (p_what) {
		// TODO Optimization: this is also called when we quit the game or destroy the world
		// which can make things a bit slow, but I don't know if it can easily be avoided
		case NOTIFICATION_UNPARENTED:
			// The user could queue_free() that node in game,
			// so we have to notify the instancer to remove the multimesh instance and pointer
			if (_parent != nullptr) {
				_parent->on_body_removed(_data_block_position, _render_block_index, _instance_index);
				_parent = nullptr;
			}
			break;
	}
}

// This method exists to workaround not being able to add or remove children to the same parent,
// in case this is necessary in removal behaviors. But it requires the user to explicitly call that instead of
// queue_free().
void VoxelInstancerRigidBody::queue_free_and_notify_instancer() {
	queue_free();
	if (_parent != nullptr) {
		_parent->on_body_removed(_data_block_position, _render_block_index, _instance_index);
		_parent = nullptr;
	}
}

void VoxelInstancerRigidBody::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_library_item_id"), &VoxelInstancerRigidBody::get_library_item_id);
	ClassDB::bind_method(
			D_METHOD("queue_free_and_notify_instancer"), &VoxelInstancerRigidBody::queue_free_and_notify_instancer
	);
}

} // namespace zylann::voxel
