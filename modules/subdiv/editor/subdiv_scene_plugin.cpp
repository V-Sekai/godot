/**************************************************************************/
/*  subdiv_scene_plugin.cpp                                               */
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

#include "subdiv_scene_plugin.h"

#include "scene/resources/3d/importer_mesh.h"

void SubdivScenePostImportPlugin::_bind_methods() {
	// No methods to bind - handled by parent class
}

SubdivScenePostImportPlugin::SubdivScenePostImportPlugin() {
	topology_importer.instantiate();
}

SubdivScenePostImportPlugin::~SubdivScenePostImportPlugin() {
}

void SubdivScenePostImportPlugin::get_internal_import_options(
		InternalImportCategory p_category,
		List<ResourceImporter::ImportOption> *r_options) {
	
	if (p_category == INTERNAL_IMPORT_CATEGORY_MESH_3D_NODE) {
		// Add subdivision options directly to list (following Godot pattern)
		// enabled option triggers visibility update for level/mode
		r_options->push_back(ResourceImporter::ImportOption(
			PropertyInfo(Variant::BOOL, "subdivision/enabled", PROPERTY_HINT_NONE, "", 
			             PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 
			false));
		r_options->push_back(ResourceImporter::ImportOption(
			PropertyInfo(Variant::INT, "subdivision/level", PROPERTY_HINT_RANGE, "0,3,1"), 
			0));
		r_options->push_back(ResourceImporter::ImportOption(
			PropertyInfo(Variant::INT, "subdivision/mode", PROPERTY_HINT_ENUM, "Baked (import time),Runtime (dynamic)"), 
			0));
	}
}

Variant SubdivScenePostImportPlugin::get_internal_option_visibility(
		InternalImportCategory p_category,
		const String &p_scene_import_type,
		const String &p_option,
		const HashMap<StringName, Variant> &p_options) const {
	
	if (p_category == INTERNAL_IMPORT_CATEGORY_MESH_3D_NODE) {
		// Only show subdivision level/mode options when enabled
		if (p_option == "subdivision/level" || p_option == "subdivision/mode") {
			return bool(p_options["subdivision/enabled"]);
		}
	}
	
	return true;
}

void SubdivScenePostImportPlugin::internal_process(
		InternalImportCategory p_category,
		Node *p_base_scene,
		Node *p_node,
		Ref<Resource> p_resource,
		const Dictionary &p_options) {
	
	if (p_category == INTERNAL_IMPORT_CATEGORY_MESH_3D_NODE) {
		// Get subdivision settings using parent class method
		Variant enabled_var = get_option_value("subdivision/enabled");
		bool enabled = enabled_var.operator bool();
		
		if (!enabled) {
			return;
		}
		
		Variant level_var = get_option_value("subdivision/level");
		Variant mode_var = get_option_value("subdivision/mode");
		int level = level_var.operator int();
		int mode = mode_var.operator int();
		
		// Process ImporterMeshInstance3D nodes
		ImporterMeshInstance3D *mesh_instance = Object::cast_to<ImporterMeshInstance3D>(p_node);
		if (mesh_instance && level > 0) {
			TopologyDataImporter::ImportMode import_mode = 
				mode == 0 ? TopologyDataImporter::BAKED_SUBDIV_MESH : 
				            TopologyDataImporter::IMPORTER_MESH;
			
			// Use topology importer to convert mesh instance
			topology_importer->convert_importer_meshinstance_to_subdiv(
				mesh_instance, import_mode, level);
		}
	}
}
