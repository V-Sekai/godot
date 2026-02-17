/**************************************************************************/
/*  mm_editor_plugin.cpp                                                  */
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

#include "mm_editor_plugin.h"

#include "scene/gui/button.h"
#include "scene/gui/tab_container.h"

MMEditorPlugin::MMEditorPlugin() {
	_editor = memnew(MMEditor);

	_gizmo_plugin.instantiate();
	add_node_3d_gizmo_plugin(_gizmo_plugin);

	_post_import_plugin.instantiate();
	add_scene_post_import_plugin(_post_import_plugin);

	_bottom_panel_button = add_control_to_bottom_panel(_editor, "MMEditor");
	_editor->connect("animation_visualization_requested", callable_mp(_gizmo_plugin.ptr(), &MMEditorGizmoPlugin::on_anim_viz_requested));
	make_visible(false);
}

MMEditorPlugin::~MMEditorPlugin() {
	remove_scene_post_import_plugin(_post_import_plugin);
	remove_control_from_bottom_panel(_bottom_panel_button);
	remove_node_3d_gizmo_plugin(_gizmo_plugin);
	remove_control_from_bottom_panel(_editor);
	if (_gizmo_plugin.is_valid()) {
		_gizmo_plugin.unref();
	}
}

void MMEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		_bottom_panel_button->show();
	} else {
		_bottom_panel_button->hide();
		hide_bottom_panel();
	}
}

bool MMEditorPlugin::handles(Object *p_object) const {
	return (Object::cast_to<MMCharacter>(p_object) != nullptr);
}

void MMEditorPlugin::edit(Object *p_object) {
	MMCharacter *character = Object::cast_to<MMCharacter>(p_object);
	make_visible(character != nullptr);

	if (!character) {
		return;
	}

	_editor->set_character(character);
}
