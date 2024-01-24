/**************************************************************************/
/*  register_types.cpp                                                    */
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

#include "register_types.h"
#include "constants/voxel_string_names.h"
#include "edition/voxel_tool.h"
#include "edition/voxel_tool_buffer.h"
#include "editor/editor_node.h"
#include "meshers/blocky/voxel_library.h"
#include "meshers/blocky/voxel_mesher_blocky.h"
#include "meshers/cubes/voxel_mesher_cubes.h"
#include "storage/voxel_buffer.h"
#include "storage/voxel_memory_pool.h"
#include "streams/vox_loader.h"
#include "util/macros.h"

#include <core/config/engine.h>

#include "editor/editor_plugin.h"
#include "vox_importer.h"

#ifdef VOXEL_RUN_TESTS
#include "tests/tests.h"
#endif

#ifndef _3D_DISABLED
static void _editor_init() {
	Ref<VoxelVoxImporter> import_vox;
	import_vox.instantiate();
	ResourceImporterScene::add_scene_importer(import_vox);
}
#endif

void initialize_voxel_mesh_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_EDITOR) {
		return;
	}
	VoxelMemoryPool::create_singleton();
	VoxelStringNames::create_singleton();

	// Misc
	ClassDB::register_class<Voxel>();
	ClassDB::register_class<VoxelLibrary>();
	ClassDB::register_class<VoxelColorPalette>();

	// Storage
	ClassDB::register_class<VoxelBuffer>();

	// Utilities
	ClassDB::register_class<VoxelRaycastResult>();
	ClassDB::register_class<VoxelTool>();
	// I had to bind this one despite it being useless as-is because otherwise
	// Godot lazily initializes its class. And this can happen in a thread,
	// causing crashes due to the concurrent access
	ClassDB::register_class<VoxelToolBuffer>();
	ClassDB::register_class<VoxelVoxLoader>();

	// Meshers
	ClassDB::register_abstract_class<VoxelMesher>();
	ClassDB::register_class<VoxelMesherBlocky>();
	ClassDB::register_class<VoxelMesherCubes>();

#ifndef _3D_DISABLED
	ClassDB::APIType prev_api = ClassDB::get_current_api();
	ClassDB::set_current_api(ClassDB::API_EDITOR);
	GDREGISTER_CLASS(VoxelVoxImporter);
	ClassDB::set_current_api(prev_api);
	EditorNode::add_init_callback(_editor_init);
#endif

	// Reminder: how to create a singleton accessible from scripts:
	// Engine::get_singleton()->add_singleton(Engine::Singleton("SingletonName",singleton_instance));

	PRINT_VERBOSE(
			String("Size of Object: {0}").format(varray((int)sizeof(Object))));
	PRINT_VERBOSE(String("Size of RefCounted: {0}")
						  .format(varray((int)sizeof(RefCounted))));
	PRINT_VERBOSE(String("Size of Node: {0}").format(varray((int)sizeof(Node))));
	PRINT_VERBOSE(String("Size of VoxelBuffer: {0}")
						  .format(varray((int)sizeof(VoxelBuffer))));

#ifdef VOXEL_RUN_TESTS
	run_voxel_tests();
#endif
}

void uninitialize_voxel_mesh_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_EDITOR) {
		return;
	}
	// At this point, the GDScript module has nullified
	// GDScriptLanguage::singleton!! That means it's impossible to free scripts
	// still referenced by VoxelServer. And that can happen, because users can
	// write custom generators, which run inside threads, and these threads are
	// hosted in the server... See
	// https://github.com/Zylann/godot_voxel/issues/189

	VoxelStringNames::destroy_singleton();

	// Do this last as VoxelServer might still be holding some refs to voxel
	// blocks
	unsigned int used_blocks =
			VoxelMemoryPool::get_singleton()->debug_get_used_blocks();
	if (used_blocks > 0) {
		ERR_PRINT(String("VoxelMemoryPool: "
						 "{0} memory blocks are still used when unregistering the "
						 "module. Recycling leak?")
						  .format(varray(used_blocks)));
	}
	VoxelMemoryPool::destroy_singleton();
}
