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
	if (p_category == INTERNAL_IMPORT_CATEGORY_MESH) {
		// Add subdivision options at mesh resource level
		r_options->push_back(ResourceImporter::ImportOption(
				PropertyInfo(Variant::BOOL, "subdivision/enabled", PROPERTY_HINT_NONE, "",
						PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED),
				false));
		r_options->push_back(ResourceImporter::ImportOption(
				PropertyInfo(Variant::INT, "subdivision/level", PROPERTY_HINT_RANGE, "0,3,1"),
				0));
	}
}

Variant SubdivScenePostImportPlugin::get_internal_option_visibility(
		InternalImportCategory p_category,
		const String &p_scene_import_type,
		const String &p_option,
		const HashMap<StringName, Variant> &p_options) const {
	if (p_category == INTERNAL_IMPORT_CATEGORY_MESH) {
		// Only show subdivision level when enabled
		if (p_option == "subdivision/level") {
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
	if (p_category == INTERNAL_IMPORT_CATEGORY_MESH) {
		// Process ImporterMesh resources directly (not nodes)
		bool enabled = false;
		if (p_options.has("subdivision/enabled")) {
			enabled = p_options["subdivision/enabled"];
		}

		if (!enabled) {
			return;
		}

		int level = 0;
		if (p_options.has("subdivision/level")) {
			level = p_options["subdivision/level"];
		}

		// Subdivide the ImporterMesh resource in-place
		// This preserves the import pipeline - no node conversion
		Ref<ImporterMesh> importer_mesh = p_resource;
		if (importer_mesh.is_valid() && level > 0) {
			topology_importer->subdivide_importer_mesh_in_place(importer_mesh, level);
		}
	}
}
