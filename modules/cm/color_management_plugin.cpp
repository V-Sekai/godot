/*************************************************************************/
/*  color_management_plugin.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "color_management_plugin.h"
#include "color_profile.h"
#include "color_transform.h"
#include "core/object/ref_counted.h"
#include "editor/editor_settings.h"
#include "servers/rendering_server.h"

void ColorManagementPlugin::_on_settings_change() {
	Viewport *viewport = get_viewport();
	if (!viewport) {
		return;
	}
	RenderingServer::get_singleton()->viewport_set_color_management(viewport->get_viewport_rid());
}

void ColorManagementPlugin::_register_settings() {
	EditorSettings *es = EditorSettings::get_singleton();

	EDITOR_DEF("interface/color_management/screen_profile", "");
	es->add_property_hint(PropertyInfo(Variant::STRING, "interface/color_management/screen_profile", PROPERTY_HINT_GLOBAL_FILE, "*.icc,*.icm", PROPERTY_USAGE_DEFAULT));

	EDITOR_DEF("interface/color_management/intent", "Perceptual");
	es->add_property_hint(PropertyInfo(Variant::STRING, "interface/color_management/intent", PROPERTY_HINT_ENUM,
			"Perceptual"
			","
			"Relative Colorimetric"
			","
			"Saturation"
			","
			"Absolute Colorimetric"));

	EDITOR_DEF("interface/color_management/black_point_compensation", true);
	es->add_property_hint(PropertyInfo(Variant::BOOL, "interface/color_management/black_point_compensation"));
}

void ColorManagementPlugin::_bind_methods() {
}

ColorManagementPlugin::ColorManagementPlugin() {
	EditorSettings *es = EditorSettings::get_singleton();
	es->connect("settings_changed", callable_mp(this, &ColorManagementPlugin::_on_settings_change));
	_register_settings();
	_on_settings_change();
}
