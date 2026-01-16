/**************************************************************************/
/*  script_language_safegdscript.cpp                                      */
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

#include "script_language_safegdscript.h"
#include "../script_language_common.h"
#include "core/config/engine.h"
#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#include "script_safegdscript.h"
#ifdef TOOLS_ENABLED
#include "editor/editor_interface.h"
#endif
#include "scene/gui/control.h"
#include "scene/resources/texture.h"
#include "scene/resources/theme.h"
#include <string>
#include <unordered_set>

static SafeGDScriptLanguage *safegdscript_language;

void SafeGDScriptLanguage::init_language() {
	if (safegdscript_language == nullptr) {
		safegdscript_language = memnew(SafeGDScriptLanguage);
	}
}

void SafeGDScriptLanguage::deinit() {
	if (safegdscript_language) {
		ScriptServer::unregister_language(safegdscript_language);
		memdelete(safegdscript_language);
		safegdscript_language = nullptr;
	}
}

SafeGDScriptLanguage *SafeGDScriptLanguage::get_singleton() {
	return safegdscript_language;
}

// Internal ScriptLanguage API methods (no underscore prefix)
String SafeGDScriptLanguage::get_name() const {
	return "SafeGD";
}

void SafeGDScriptLanguage::init() {
}

String SafeGDScriptLanguage::get_type() const {
	return "SafeGDScript";
}

String SafeGDScriptLanguage::get_extension() const {
	return "safegd";
}

void SafeGDScriptLanguage::finish() {
}

Vector<String> SafeGDScriptLanguage::get_reserved_words() const {
	static const Vector<String> reserved_words{
		"if", "elif", "else", "for", "while", "match", "break", "continue", "return", "pass",
		"func", "class", "class_name", "extends", "is", "in", "as", "and", "or", "not",
		"var", "const", "static", "enum", "signal", "super", "self",
		"true", "false", "null", "void", "bool", "int", "float", "String",
		"Vector2", "Vector2i", "Vector3", "Vector3i", "Vector4", "Vector4i", "Color",
		"Array", "Dictionary", "PackedByteArray", "PackedInt32Array", "PackedInt64Array",
		"PackedFloat32Array", "PackedFloat64Array", "PackedStringArray", "PackedVector2Array",
		"PackedVector3Array", "PackedColorArray", "Node", "RefCounted", "Resource",
		"assert", "await", "yield"
	};
	return reserved_words;
}

bool SafeGDScriptLanguage::is_control_flow_keyword(const String &p_keyword) const {
	static const std::unordered_set<std::string> control_flow_keywords{
		"if", "elif", "else", "for", "while", "match", "break", "continue", "return", "pass", "assert"
	};
	return control_flow_keywords.find(p_keyword.utf8().get_data()) != control_flow_keywords.end();
}

Vector<String> SafeGDScriptLanguage::get_comment_delimiters() const {
	Vector<String> delimiters;
	delimiters.push_back("#");
	return delimiters;
}

Vector<String> SafeGDScriptLanguage::get_doc_comment_delimiters() const {
	Vector<String> delimiters;
	delimiters.push_back("##");
	return delimiters;
}

Vector<String> SafeGDScriptLanguage::get_string_delimiters() const {
	Vector<String> delimiters;
	delimiters.push_back("\" \"");
	delimiters.push_back("' '");
	delimiters.push_back("\"\"\" \"\"\"");
	return delimiters;
}

Ref<Script> SafeGDScriptLanguage::make_template(const String &p_template, const String &p_class_name, const String &p_base_class_name) const {
	SafeGDScript *script = memnew(SafeGDScript);
	return Ref<Script>(script);
}

bool SafeGDScriptLanguage::is_using_templates() {
	return false;
}

bool SafeGDScriptLanguage::validate(const String &p_script, const String &p_path, List<String> *r_functions, List<ScriptError> *r_errors, List<Warning> *r_warnings, HashSet<int> *r_safe_lines) const {
	return true; // For now, assume all SafeGDScript scripts are valid
}

String SafeGDScriptLanguage::validate_path(const String &p_path) const {
	return String();
}

Script *SafeGDScriptLanguage::create_script() const {
	SafeGDScript *script = memnew(SafeGDScript);
	return script;
}

bool SafeGDScriptLanguage::supports_builtin_mode() const {
	return true;
}

bool SafeGDScriptLanguage::supports_documentation() const {
	return false;
}

bool SafeGDScriptLanguage::can_inherit_from_file() const {
	return false;
}

int SafeGDScriptLanguage::find_function(const String &p_function, const String &p_code) const {
	return -1;
}

String SafeGDScriptLanguage::make_function(const String &p_class, const String &p_name, const PackedStringArray &p_args) const {
	return String();
}

Error SafeGDScriptLanguage::complete_code(const String &p_code, const String &p_path, Object *p_owner, List<ScriptLanguage::CodeCompletionOption> *r_options, bool &r_forced, String &r_call_hint) {
	return OK; // No code completion for SafeGDScript
}

void SafeGDScriptLanguage::auto_indent_code(String &p_code, int p_from_line, int p_to_line) const {
	// No auto-indent for SafeGDScript
}

void SafeGDScriptLanguage::add_global_constant(const StringName &p_variable, const Variant &p_value) {
}

void SafeGDScriptLanguage::add_named_global_constant(const StringName &p_name, const Variant &p_value) {
}

void SafeGDScriptLanguage::remove_named_global_constant(const StringName &p_name) {
}

Error SafeGDScriptLanguage::open_in_external_editor(const Ref<Script> &p_script, int p_line, int p_col) {
	return OK;
}

bool SafeGDScriptLanguage::overrides_external_editor() {
	return false;
}

void SafeGDScriptLanguage::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("safegd");
}

void SafeGDScriptLanguage::frame() {
#ifdef TOOLS_ENABLED
	static bool icon_registered = false;
	if (!icon_registered && Engine::get_singleton()->is_editor_hint()) {
		icon_registered = true;
		load_icon();
		EditorInterface::get_singleton()->get_base_control()->connect("theme_changed", callable_mp(this, &SafeGDScriptLanguage::load_icon));
	}
#endif
}

void SafeGDScriptLanguage::thread_enter() {
}

void SafeGDScriptLanguage::thread_exit() {
}

String SafeGDScriptLanguage::debug_get_error() const {
	return String();
}

int SafeGDScriptLanguage::debug_get_stack_level_count() const {
	return 0;
}

int SafeGDScriptLanguage::debug_get_stack_level_line(int p_level) const {
	return 0;
}

String SafeGDScriptLanguage::debug_get_stack_level_function(int p_level) const {
	return String();
}

String SafeGDScriptLanguage::debug_get_stack_level_source(int p_level) const {
	return String();
}

void SafeGDScriptLanguage::debug_get_stack_level_locals(int p_level, List<String> *p_locals, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
	// No debug locals available
}

void SafeGDScriptLanguage::debug_get_stack_level_members(int p_level, List<String> *p_members, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
	// No debug members available
}

ScriptInstance *SafeGDScriptLanguage::debug_get_stack_level_instance(int p_level) {
	return nullptr;
}

void SafeGDScriptLanguage::debug_get_globals(List<String> *p_globals, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
	// No debug globals available
}

String SafeGDScriptLanguage::debug_parse_stack_level_expression(int p_level, const String &p_expression, int p_max_subitems, int p_max_depth) {
	return String();
}

void SafeGDScriptLanguage::reload_all_scripts() {
}

void SafeGDScriptLanguage::reload_scripts(const Array &p_scripts, bool p_soft_reload) {
}

void SafeGDScriptLanguage::reload_tool_script(const Ref<Script> &p_script, bool p_soft_reload) {
}

void SafeGDScriptLanguage::get_public_functions(List<MethodInfo> *p_functions) const {
	// No public functions to add
}

void SafeGDScriptLanguage::get_public_constants(List<Pair<String, Variant>> *p_constants) const {
	// No public constants to add
}

void SafeGDScriptLanguage::get_public_annotations(List<MethodInfo> *p_annotations) const {
	// No public annotations to add
}

void SafeGDScriptLanguage::profiling_start() {
}

void SafeGDScriptLanguage::profiling_stop() {
}

void SafeGDScriptLanguage::profiling_set_save_native_calls(bool p_enable) {
}

int SafeGDScriptLanguage::profiling_get_accumulated_data(ProfilingInfo *p_info_arr, int p_info_max) {
	return 0;
}

int SafeGDScriptLanguage::profiling_get_frame_data(ProfilingInfo *p_info_arr, int p_info_max) {
	return 0;
}

#ifdef TOOLS_ENABLED

void SafeGDScriptLanguage::load_icon() {
	static bool reenter = false;
	if (reenter) {
		return;
	}
	reenter = true;
	Ref<FileAccess> fa = FileAccess::open("res://addons/godot_sandbox/SafeGDScript.svg", FileAccess::READ);
	if (Engine::get_singleton()->is_editor_hint() && fa.is_valid()) {
		Ref<Theme> editor_theme = EditorInterface::get_singleton()->get_editor_theme();
		if (editor_theme.is_valid() && !editor_theme->has_icon("SafeGDScript", "EditorIcons")) {
			Ref<Texture2D> tex = ResourceLoader::load("res://addons/godot_sandbox/SafeGDScript.svg");
			if (tex.is_valid()) {
				editor_theme->set_icon("SafeGDScript", "EditorIcons", tex);
			}
		}
	}
	reenter = false;
}

bool SafeGDScriptLanguage::handles_global_class_type(const String &p_type) const {
	return p_type == "SafeGDScript" || p_type == "Sandbox";
}

String SafeGDScriptLanguage::get_global_class_name(const String &p_path, String *r_base_type, String *r_icon_path, bool *r_is_abstract, bool *r_is_tool) const {
	if (!p_path.is_empty()) {
		if (r_base_type) {
			*r_base_type = "Sandbox";
		}
		if (r_icon_path) {
			*r_icon_path = "res://addons/godot_sandbox/SafeGDScript.svg";
		}
		if (r_is_abstract) {
			*r_is_abstract = false;
		}
		if (r_is_tool) {
			*r_is_tool = true;
		}
		return SafeGDScript::PathToGlobalName(p_path);
	}
	return String();
}
#endif
