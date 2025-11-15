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

#include "usd_document.h"
#include "usd_export_settings.h"
#include "usd_plugin.h"
#include "usd_state.h"

// USD import classes (GLTF-based)
#include "../gltf/extensions/gltf_document_extension_convert_importer_mesh.h"
#include "usd_import_document.h"
#include "usd_import_state.h"

#include "core/object/class_db.h"
#include "core/object/gdvirtual.gen.inc"
#include "core/object/object_macros.h"
#include "editor/plugins/editor_plugin.h"

#ifdef TOOLS_ENABLED
#include "editor_import/editor_scene_importer_usd.h"
#include "core/config/project_settings.h"
#include "editor/editor_node.h"

static void _editor_init() {
	Ref<EditorSceneFormatImporterUSD> import_usd;
	import_usd.instantiate();
	ResourceImporterScene::add_scene_importer(import_usd);
}
#endif // TOOLS_ENABLED

#define USD_REGISTER_DOCUMENT_EXTENSION(m_doc_ext_class) \
	Ref<m_doc_ext_class> extension_##m_doc_ext_class;    \
	extension_##m_doc_ext_class.instantiate();           \
	GLTFDocument::register_gltf_document_extension(extension_##m_doc_ext_class);

void initialize_openusd_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		// Export classes
		ClassDB::register_class<UsdDocument>();
		ClassDB::register_class<UsdExportSettings>();
		ClassDB::register_class<UsdState>();
		
		// Import classes (GLTF-based)
		GDREGISTER_CLASS(USDDocument);
		GDREGISTER_CLASS(USDState);
		bool is_editor = Engine::get_singleton()->is_editor_hint();
		if (!is_editor) {
			USD_REGISTER_DOCUMENT_EXTENSION(GLTFDocumentExtensionConvertImporterMesh);
		}
	} else if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
		// Export plugin
		ClassDB::register_class<USDPlugin>();
		EditorPlugins::add_by_type<USDPlugin>();
		Engine::get_singleton()->add_singleton(Engine::Singleton("USDPlugin", memnew(USDPlugin)));
		
		// Import importer
		GDREGISTER_CLASS(EditorSceneFormatImporterUSD);
		EditorNode::add_init_callback(_editor_init);
	}
}

void uninitialize_openusd_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		GLTFDocument::unregister_all_gltf_document_extensions();
	} else if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
		Engine::get_singleton()->remove_singleton("USDPlugin");
		// EditorPlugins::remove_by_type<USDPlugin>();
	}
}
