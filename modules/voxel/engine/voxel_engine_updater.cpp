/**************************************************************************/
/*  voxel_engine_updater.cpp                                              */
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

#include "voxel_engine_updater.h"
#include "../util/io/log.h"
#include "voxel_engine.h"

// Needed for doing `Node *root = SceneTree::get_root()`, Window* is forward-declared
#include "../util/godot/classes/scene_tree.h"
#include "../util/godot/classes/window.h"

namespace zylann::voxel {

bool g_updater_created = false;

VoxelEngineUpdater::VoxelEngineUpdater() {
	ZN_PRINT_VERBOSE("Creating VoxelEngineUpdater");
	set_process(true);
	// We don't want it to stop when the scene tree is paused
	set_process_mode(PROCESS_MODE_ALWAYS);
	g_updater_created = true;
}

VoxelEngineUpdater::~VoxelEngineUpdater() {
	g_updater_created = false;
}

void VoxelEngineUpdater::ensure_existence(SceneTree *st) {
	if (st == nullptr) {
		return;
	}
	if (g_updater_created) {
		return;
	}
	Node *root = st->get_root();
	for (int i = 0; i < root->get_child_count(); ++i) {
		VoxelEngineUpdater *u = Object::cast_to<VoxelEngineUpdater>(root->get_child(i));
		if (u != nullptr) {
			return;
		}
	}
	VoxelEngineUpdater *u = memnew(VoxelEngineUpdater);
	u->set_name("VoxelEngineUpdater_dont_touch_this");
	// TODO This can fail (for example if `Node::data.blocked > 0` while in `_ready()`) but Godot offers no API to check
	// anything. So if this fail, the node will leak.
	root->add_child(u);

	VoxelEngine::get_singleton().try_initialize_gpu_features();
}

void VoxelEngineUpdater::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_PROCESS:
			// To workaround the absence of API to have a custom server processing in the main loop
			zylann::voxel::VoxelEngine::get_singleton().process();
			break;

		case NOTIFICATION_PREDELETE:
			ZN_PRINT_VERBOSE("Deleting VoxelEngineUpdater");
			break;

		default:
			break;
	}
}

} // namespace zylann::voxel
