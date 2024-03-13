/**************************************************************************/
/*  subdiv_plugin.cpp                                                     */
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

#include "core/object/object.h"
#include "core/variant/variant.h"
#include "editor/import/editor_import_plugin.h"
#include "modules/gltf/gltf_document.h"

#include "scene/resources/packed_scene.h"
#include "subdiv_converter.h"
#include "subdiv_plugin.h"

void GodotSubdivImporter::_bind_methods() {
}

String GodotSubdivImporter::get_importer_name() const {
	return "godot-subdiv.importer";
}

String GodotSubdivImporter::get_visible_name() const {
	return "Godot Subdiv Importer";
}

void GodotSubdivImporter::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("glb");
	p_extensions->push_back("gltf");
}

String GodotSubdivImporter::get_save_extension() const {
	return "scn";
}

String GodotSubdivImporter::get_resource_type() const {
	return "PackedScene";
}

int GodotSubdivImporter::get_preset_count() const {
	return 1;
}

String GodotSubdivImporter::get_preset_name(int preset) const {
	return "Default";
}

float GodotSubdivImporter::get_priority() const {
	return 0;
}

int GodotSubdivImporter::get_import_order() const {
	return IMPORT_ORDER_SCENE;
}

bool GodotSubdivImporter::get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	return true;
}

void GodotSubdivImporter::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
	ImportOption option(PropertyInfo(Variant::INT, "import_as", PROPERTY_HINT_ENUM, "BakedSubdivMesh (bake at runtime),ImporterMesh (bake at import)"), "ImporterMesh (bake at import)");
	r_options->push_back(option);
	ImportOption option_2(PropertyInfo(Variant::INT, "subdivision_level", PROPERTY_HINT_RANGE, "0,2"), 0);
	r_options->push_back(option_2);
}

Error GodotSubdivImporter::import(const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	Ref<GLTFDocument> gltf = memnew(GLTFDocument);
	Ref<GLTFState> gltf_state = memnew(GLTFState);
	gltf->append_from_file(p_source_file, gltf_state, 0);
	Node *node = gltf->generate_scene(gltf_state);

	node->set_name(p_source_file.get_file().get_basename());

	int subdiv_level = p_options["subdivision_level"];
	int import_as = p_options["import_as"];

	Ref<SubdivConverter> subdiv_converter = Ref<SubdivConverter>(memnew(SubdivConverter(TopologyDataImporter::ImportMode(import_as), subdiv_level)));

	if (node != nullptr) {
		subdiv_converter->convert_importer_mesh_instances_recursively(node);
	} else {
		print_line("GLTF importer failed, so could not run convert code");
	}

	Ref<PackedScene> packed;
	packed.instantiate();
	packed->pack(node);
	node->queue_free();

	return ResourceSaver::save(packed, p_save_path + String(".scn"));
}

GodotSubdivImporter::GodotSubdivImporter() {
}
