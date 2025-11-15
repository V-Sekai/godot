/**************************************************************************/
/*  usd_plugin.h                                                          */
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

#pragma once

#include "editor/inspector/editor_inspector.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/plugins/editor_plugin.h"

// USD headers
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformOp.h>

PXR_NAMESPACE_USING_DIRECTIVE

class UsdDocument;
class UsdExportSettings;
class EditorFileDialog;

class USDPlugin : public EditorPlugin {
	GDCLASS(USDPlugin, EditorPlugin);

private:
	Button *hello_button;

	// USD Export components
	Ref<UsdDocument> _usd_document;
	Ref<UsdExportSettings> _export_settings;
	EditorInspector *_settings_inspector = nullptr;
	EditorFileDialog *_file_dialog = nullptr;

	// USD Import components
	EditorFileDialog *_import_file_dialog = nullptr;

	void _popup_usd_export_dialog();
	void _export_scene_as_usd(const String &p_file_path);

	void _popup_usd_import_dialog();
	void _import_usd_file(const String &p_file_path);

	// Helper method to print the prim hierarchy
	void _print_prim_hierarchy(const UsdPrim &p_prim, int p_indent);

	// Helper method to print the node hierarchy
	void _print_node_hierarchy(Node *p_node, int p_indent);

	// Helper method to extract and apply transform from a USD prim to a Godot node
	bool _apply_transform_from_usd_prim(const UsdPrim &p_prim, Node3D *p_node);

	// Helper method to convert a USD prim to a Godot node
	Node *_convert_prim_to_node(const UsdPrim &p_prim, Node *p_parent, Node *p_scene_root);

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	USDPlugin();
	~USDPlugin();

	virtual bool has_main_screen() const override;
	virtual String get_plugin_name() const override;

	void _on_hello_button_pressed();
};
