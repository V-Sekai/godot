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

#include "core/object/class_db.h"

#include "src/import/topology_data_importer.hpp"
#include "src/resources/baked_subdiv_mesh.hpp"
#include "src/resources/topology_data_mesh.hpp"
#include "src/subdivision/subdivision_baker.hpp"
#include "src/subdivision/subdivision_mesh.hpp"

#include "src/subdivision/quad_subdivider.hpp"
#include "src/subdivision/subdivider.hpp"
#include "src/subdivision/triangle_subdivider.hpp"

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#include "editor/import/editor_import_plugin.h"
#include "editor/import/resource_importer_scene.h"
#include "editor/subdiv_plugin.h"
#endif

#ifdef TESTS_ENABLED
#include "subdiv_test.hpp"
#endif

void initialize_subdiv_module(ModuleInitializationLevel p_level) {
#ifdef TOOLS_ENABLED
	if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
		ClassDB::APIType prev_api = ClassDB::get_current_api();
		ClassDB::set_current_api(ClassDB::API_EDITOR);
		EditorPlugins::add_by_type<SubdivEditorPlugin>();

		ClassDB::set_current_api(prev_api);
		return;
	}
#endif

	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	ClassDB::register_class<Subdivider>();
	ClassDB::register_class<QuadSubdivider>();
	ClassDB::register_class<TriangleSubdivider>();

	ClassDB::register_class<SubdivisionMesh>();
	ClassDB::register_class<SubdivisionBaker>();

	ClassDB::register_class<TopologyDataImporter>();

	ClassDB::register_class<TopologyDataMesh>();
	ClassDB::register_class<BakedSubdivMesh>();
}

void uninitialize_subdiv_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}
