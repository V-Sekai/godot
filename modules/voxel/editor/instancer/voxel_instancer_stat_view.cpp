/**************************************************************************/
/*  voxel_instancer_stat_view.cpp                                         */
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

#include "voxel_instancer_stat_view.h"
#include "../../terrain/instancing/voxel_instance_library.h"
#include "../../terrain/instancing/voxel_instance_library_item.h"
#include "../../terrain/instancing/voxel_instancer.h"
#include "../../util/godot/classes/node.h"
#include "../../util/godot/classes/tree.h"

namespace zylann::voxel {

VoxelInstancerStatView::VoxelInstancerStatView() {
	_tree = memnew(Tree);
	_tree->set_columns(2);
	_tree->set_select_mode(Tree::SELECT_ROW);
	_tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	_tree->set_column_expand_ratio(0, 1);
	_tree->set_column_expand_ratio(1, 3);
	add_child(_tree);
}

void VoxelInstancerStatView::set_instancer(const VoxelInstancer *instancer) {
	_instancer = instancer;
	set_process(_instancer != nullptr);
}

void VoxelInstancerStatView::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_PROCESS:
			process();
			break;
	}
}

void VoxelInstancerStatView::process() {
	ERR_FAIL_COND(_instancer == nullptr);
	const VoxelInstancer &instancer = *_instancer;

	Ref<VoxelInstanceLibrary> library = instancer.get_library();

	instancer.debug_get_instance_counts(_count_per_layer);

	_tree->clear();

	if (library.is_null()) {
		return;
	}

	TreeItem *root = _tree->create_item();
	_tree->set_hide_root(true);

	for (auto it = _count_per_layer.begin(); it != _count_per_layer.end(); ++it) {
		TreeItem *tree_item = _tree->create_item(root);
		const VoxelInstanceLibraryItem *lib_item = library->get_item_const(it->first);
		ERR_CONTINUE(lib_item == nullptr);

		String name = lib_item->get_item_name();
		if (name == "") {
			name = "[" + String::num_int64(it->first) + "]";
		}

		tree_item->set_text(0, name);
		tree_item->set_text(1, String::num_int64(it->second));
	}
}

} // namespace zylann::voxel
