/**************************************************************************/
/*  script_safegdscript.cpp                                               */
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

#include "script_safegdscript.h"

#include "../elf/script_elf.h"
#include "../elf/script_instance.h"
#include "../sandbox.h"
#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#include "core/object/class_db.h"
#include "core/variant/callable.h"
#include "script_instance_safegdscript.h"
#include "script_language_safegdscript.h"

static constexpr bool VERBOSE_LOGGING = false;
static Sandbox *compiler = nullptr;

void SafeGDScript::_bind_methods() {
}

// Internal Script API methods (no underscore prefix)
bool SafeGDScript::can_instantiate() const {
	return true;
}

Ref<Script> SafeGDScript::get_base_script() const {
	return Ref<Script>();
}

StringName SafeGDScript::get_global_name() const {
	return PathToGlobalName(this->path);
}

bool SafeGDScript::inherits_script(const Ref<Script> &p_script) const {
	return false;
}

StringName SafeGDScript::get_instance_base_type() const {
	return StringName("Sandbox");
}

ScriptInstance *SafeGDScript::instance_create(Object *p_for_object) {
	SafeGDScriptInstance *instance = memnew(SafeGDScriptInstance(p_for_object, Ref<SafeGDScript>(this)));
	instances.insert(instance);
	return instance;
}

PlaceHolderScriptInstance *SafeGDScript::placeholder_instance_create(Object *p_for_object) {
	return nullptr; // TODO: implement if needed
}

bool SafeGDScript::instance_has(const Object *p_object) const {
	return false;
}

bool SafeGDScript::has_source_code() const {
	return true;
}

String SafeGDScript::get_source_code() const {
	return source_code;
}

void SafeGDScript::set_source_code(const String &p_code) {
	source_code = p_code;
	compile_source_to_elf();
}

Error SafeGDScript::reload(bool p_keep_state) {
	compile_source_to_elf();
	return OK;
}

bool SafeGDScript::has_method(const StringName &p_method) const {
	if (p_method == StringName("_init")) {
		return true;
	}
	for (const MethodInfo &method_info : methods_info) {
		if (method_info.name == p_method) {
			return true;
		}
	}
	return false;
}

bool SafeGDScript::has_static_method(const StringName &p_method) const {
	return false;
}

MethodInfo SafeGDScript::get_method_info(const StringName &p_method) const {
	for (const MethodInfo &method_info : methods_info) {
		if (method_info.name == p_method) {
			return method_info;
		}
	}
	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("SafeGDScript::get_method_info: Method " + String(p_method) + " not found.");
	}
	return MethodInfo();
}

bool SafeGDScript::is_tool() const {
	return true;
}

bool SafeGDScript::is_valid() const {
	return !elf_data.is_empty();
}

bool SafeGDScript::is_abstract() const {
	return false;
}

ScriptLanguage *SafeGDScript::get_language() const {
	return get_safegdscript_language_singleton();
}

bool SafeGDScript::has_script_signal(const StringName &p_signal) const {
	return false;
}

void SafeGDScript::get_script_signal_list(List<MethodInfo> *r_signals) const {
	// No signals to add
}

bool SafeGDScript::get_property_default_value(const StringName &p_property, Variant &r_value) const {
	return false;
}

void SafeGDScript::update_exports() {
}

void SafeGDScript::get_script_method_list(List<MethodInfo> *p_list) const {
	for (const MethodInfo &method_info : methods_info) {
		p_list->push_back(method_info);
	}
}

void SafeGDScript::get_script_property_list(List<PropertyInfo> *p_list) const {
	if (instances.is_empty()) {
		if constexpr (VERBOSE_LOGGING) {
			ERR_PRINT("SafeGDScript::get_script_property_list: No instances available.");
		}
		return;
	}
	SafeGDScriptInstance *instance = *instances.begin();
	if (instance == nullptr) {
		if constexpr (VERBOSE_LOGGING) {
			ERR_PRINT("SafeGDScript::get_script_property_list: Instance is null.");
		}
		return;
	}
	// TODO: Get properties from sandbox if needed
}

int SafeGDScript::get_member_line(const StringName &p_member) const {
	return 0;
}

void SafeGDScript::get_constants(HashMap<StringName, Variant> *p_constants) {
	// No constants to add
}

void SafeGDScript::get_members(HashSet<StringName> *p_members) {
	// Add function names as members
	for (const MethodInfo &method_info : methods_info) {
		p_members->insert(method_info.name);
	}
}

const Variant SafeGDScript::get_rpc_config() const {
	return Variant();
}

#ifdef TOOLS_ENABLED
StringName SafeGDScript::get_doc_class_name() const {
	return get_global_name();
}

void SafeGDScript::_placeholder_erased(PlaceHolderScriptInstance *p_placeholder) {
	// Handle placeholder cleanup if needed
}

String SafeGDScript::get_class_icon_path() const {
	return String("res://addons/godot_sandbox/SafeGDScript.svg");
}

Vector<DocData::ClassDoc> SafeGDScript::get_documentation() const {
	return Vector<DocData::ClassDoc>();
}

bool SafeGDScript::is_placeholder_fallback_enabled() const {
	return false;
}
#endif

SafeGDScript::SafeGDScript() {
	source_code = R"GDScript(# SafeGDScript example

func somefunction():
	var counter = 0
	while counter < 10:
		counter += 1
	return counter

)GDScript";
}

SafeGDScript::~SafeGDScript() {
}

void SafeGDScript::set_path(const String &p_path, bool p_take_over) {
	if (p_path.is_empty()) {
		WARN_PRINT("SafeGDScript::set_path: Empty resource path.");
		return;
	}
	Script::set_path(p_path, p_take_over);
	this->path = p_path;
	if (!this->path.is_empty()) {
		Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ);
		if (file.is_valid()) {
			this->source_code = file->get_as_text();
		} else {
			ERR_PRINT("SafeGDScript::set_path: Failed to open file: " + p_path);
		}
	}
	this->compile_source_to_elf();
}

bool SafeGDScript::compile_source_to_elf() {
	if (this->source_code.is_empty()) {
		if constexpr (VERBOSE_LOGGING) {
			ERR_PRINT("SafeGDScript::compile_source_to_elf: No source code to compile.");
		}
		return false;
	}

	if (compiler == nullptr) {
		// Check if "gdscript.elf" exists in the addons/godot_sandbox/ directory
		String compiler_path = "res://addons/godot_sandbox/gdscript.elf";
		Ref<FileAccess> fa_check = FileAccess::open(compiler_path, FileAccess::READ);
		if (!fa_check.is_valid()) {
			ERR_PRINT("SafeGDScript::compile_source_to_elf: GDScript compiler ELF not found at " + compiler_path);
			return false;
		}
		compiler = memnew(Sandbox);
		Ref<ELFScript> compiler_script = ResourceLoader::load(compiler_path);
		if (!compiler_script.is_valid()) {
			ERR_PRINT("SafeGDScript::compile_source_to_elf: Failed to load GDScript compiler ELF resource.");
			memdelete(compiler);
			compiler = nullptr;
			return false;
		}
		compiler->set_program(compiler_script);
		if (!compiler->has_program_loaded()) {
			ERR_PRINT("SafeGDScript::compile_source_to_elf: Failed to initialize GDScript compiler sandbox.");
			memdelete(compiler);
			compiler = nullptr;
			return false;
		}
	}

	// Compile the source code to ELF using the compiler sandbox
	Callable::CallError error;
	Variant src_code_var = this->source_code;
	const Variant *args[] = { &src_code_var };
	Variant result = compiler->vmcall_fn(StringName("compile"), args, 1, error);
	if (error.error != Callable::CallError::CALL_OK) {
		ERR_PRINT("SafeGDScript::compile_source_to_elf: Compilation failed with error code " + itos(static_cast<int>(error.error)));
		return false;
	}
	// Expecting the result to be a PackedByteArray containing the ELF binary
	if (result.get_type() != Variant::PACKED_BYTE_ARRAY) {
		ERR_PRINT("SafeGDScript::compile_source_to_elf: Compilation did not return a PackedByteArray.");
		return false;
	}

	this->elf_data = result;
	if (elf_data.is_empty()) {
		ERR_PRINT("SafeGDScript::compile_source_to_elf: Compilation returned empty ELF data.");
		return false;
	}

	this->update_methods_info();

	for (SafeGDScriptInstance *instance : instances) {
		instance->reset_to(this->elf_data);
	}

	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("SafeGDScript::compile_source_to_elf: Successfully compiled " + this->path + " to ELF (" + itos(this->elf_data.size()) + " bytes)");
	}

	return true;
}

void SafeGDScript::remove_instance(SafeGDScriptInstance *p_instance) {
	instances.erase(p_instance);
}

void SafeGDScript::update_methods_info() {
	Sandbox::BinaryInfo info = Sandbox::get_program_info_from_binary(this->elf_data);
	this->methods_info.clear();
	for (const String &func_name : info.functions) {
		methods_info.push_back(MethodInfo(func_name));
	}

	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("SafeGDScript::update_methods_info: Updated methods info with " + itos(methods_info.size()) + " methods.");
	}
}

// Function to get the SafeGDScript language singleton
ScriptLanguage *get_safegdscript_language_singleton() {
	return SafeGDScriptLanguage::get_singleton();
}
