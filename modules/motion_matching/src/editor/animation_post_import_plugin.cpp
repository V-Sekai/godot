/**************************************************************************/
/*  animation_post_import_plugin.cpp                                      */
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

#include "animation_post_import_plugin.h"

#include "core/io/file_access.h"
#include "scene/animation/animation_player.h"
#include "scene/main/node.h"
#include "scene/resources/animation_library.h"

Variant AnimationPostImportPlugin::_get_option_visibility(const String &p_path, bool p_for_animation, const String &p_option) const {
	if (p_option == "export/animation_export_path") {
		return p_for_animation;
	}
	return true; // Default visibility for options we don't handle.
}

void AnimationPostImportPlugin::_get_import_options(const String &p_path) {
	add_import_option_advanced(Variant::STRING, "export/animation_export_path", "", PROPERTY_HINT_DIR, "");
}

void AnimationPostImportPlugin::_pre_process(Node *p_scene) {
	const String export_path = get_option_value("export/animation_export_path");

	if (export_path.is_empty()) {
		return;
	}

	Dictionary animations;
	_export_animations(p_scene, animations, export_path);

	Dictionary subresources = get_option_value("_subresources");
	subresources["animations"] = animations;
}

void AnimationPostImportPlugin::_bind_methods() {
	GDVIRTUAL_BIND(_get_import_options, "path");
	GDVIRTUAL_BIND(_get_option_visibility, "path", "for_animation", "option");
	GDVIRTUAL_BIND(_pre_process, "scene");
}

void AnimationPostImportPlugin::_export_animations(Node *p_node, Dictionary &p_animations, const String &p_export_path) {
	AnimationPlayer *anim_node = Object::cast_to<AnimationPlayer>(p_node);

	if (anim_node) {
		List<StringName> lib_list;
		anim_node->get_animation_library_list(&lib_list);
		for (const StringName &lib_name : lib_list) {
			Ref<AnimationLibrary> lib = anim_node->get_animation_library(lib_name);
			if (lib.is_null()) {
				continue;
			}
			List<StringName> anim_list;
			lib->get_animation_list(&anim_list);
			for (const StringName &anim_name : anim_list) {
				Dictionary animation;
				animation["save_to_file/enabled"] = true;
				animation["save_to_file/keep_custom_tracks"] = "";

				String clean_anim_name = String(anim_name).validate_filename();
				String file_path = p_export_path.path_join(clean_anim_name) + ".res";
				int idx = 1;
				while (FileAccess::exists(file_path)) {
					file_path = p_export_path.path_join(clean_anim_name + String::num_int64(idx)) + ".res";
					idx++;
				}

				animation["save_to_file/path"] = file_path;

				p_animations[anim_name] = animation;
			}
		}
	}

	for (int32_t i = 0; i < p_node->get_child_count(); i++) {
		_export_animations(p_node->get_child(i), p_animations, p_export_path);
	}
}
