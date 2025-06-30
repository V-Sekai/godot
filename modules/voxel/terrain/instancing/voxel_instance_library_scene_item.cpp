/**************************************************************************/
/*  voxel_instance_library_scene_item.cpp                                 */
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

#include "voxel_instance_library_scene_item.h"

namespace zylann::voxel {

void VoxelInstanceLibrarySceneItem::set_scene(Ref<PackedScene> scene) {
	if (scene != _scene) {
		_scene = scene;
		notify_listeners(IInstanceLibraryItemListener::CHANGE_SCENE);
	}
}

Ref<PackedScene> VoxelInstanceLibrarySceneItem::get_scene() const {
	return _scene;
}

void VoxelInstanceLibrarySceneItem::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_scene", "scene"), &VoxelInstanceLibrarySceneItem::set_scene);
	ClassDB::bind_method(D_METHOD("get_scene"), &VoxelInstanceLibrarySceneItem::get_scene);

	ADD_PROPERTY(
			PropertyInfo(Variant::OBJECT, "scene", PROPERTY_HINT_RESOURCE_TYPE, PackedScene::get_class_static()),
			"set_scene",
			"get_scene"
	);
}

} // namespace zylann::voxel
