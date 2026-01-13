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

#include "subdiv_plugin.h"

#include "editor/import/3d/resource_importer_scene.h"
#include "subdiv_scene_plugin.h"

// Global scene plugin instance
static Ref<SubdivScenePostImportPlugin> scene_plugin;

void SubdivEditorPlugin::_bind_methods() {
}

void SubdivEditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			// Register subdivision plugin with standard scene importer
			scene_plugin.instantiate();
			ResourceImporterScene::add_post_importer_plugin(scene_plugin);
		} break;

		case NOTIFICATION_EXIT_TREE: {
			// Unregister scene plugin
			if (scene_plugin.is_valid()) {
				ResourceImporterScene::remove_post_importer_plugin(scene_plugin);
				scene_plugin.unref();
			}
		} break;
	}
}

SubdivEditorPlugin::SubdivEditorPlugin() {
}

SubdivEditorPlugin::~SubdivEditorPlugin() {
}
