/**************************************************************************/
/*  script_instance_safegdscript.cpp                                      */
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

#include "script_instance_safegdscript.h"

#include "../elf/script_elf.h"
#include "../elf/script_instance.h"
#include "../elf/script_instance_helper.h"
#include "../sandbox.h"
#include "../scoped_tree_base.h"
#include "core/object/object.h"
#include "core/templates/local_vector.h"
#include "core/variant/callable.h"
#include "scene/main/node.h"
#include "script_language_safegdscript.h"
#include "script_safegdscript.h"

static constexpr bool VERBOSE_LOGGING = false;

bool SafeGDScriptInstance::set(const StringName &p_name, const Variant &p_value) {
	static const StringName s_script("script");
	static const StringName s_program("program");
	if (p_name == s_script || p_name == s_program) {
		return false;
	}

	auto [sandbox, created] = get_sandbox();
	if (sandbox) {
		ScopedTreeBase stb(sandbox, Object::cast_to<Node>(this->owner));
		if (sandbox->set_property(p_name, p_value)) {
			return true;
		}
	}
	return false;
}

bool SafeGDScriptInstance::get(const StringName &p_name, Variant &r_ret) const {
	static const StringName s_script("script");
	if (p_name == s_script) {
		r_ret = this->script;
		return true;
	}
	auto [sandbox, created] = get_sandbox();
	if (sandbox) {
		ScopedTreeBase stb(sandbox, Object::cast_to<Node>(this->owner));
		if (sandbox->get_property(p_name, r_ret)) {
			return true;
		}
	}
	return false;
}

String SafeGDScriptInstance::to_string(bool *r_is_valid) {
	if (r_is_valid) {
		*r_is_valid = true;
	}
	return "<SafeGDScript>";
}

void SafeGDScriptInstance::notification(int p_notification, bool p_reversed) {
}

Variant SafeGDScriptInstance::callp(
		const StringName &p_method,
		const Variant **p_args, int p_argument_count,
		Callable::CallError &r_error) {
	// When the script instance must have a sandbox as owner,
	// use _enter_tree to get the sandbox instance.
	// Also, avoid calling internal methods.
	if (!this->auto_created_sandbox) {
		if (p_method == StringName("_enter_tree")) {
			current_sandbox->load_buffer(script->get_content());
		}
	}

	auto [sandbox, created] = get_sandbox();
	if (!sandbox) {
		r_error.error = Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL;
		return Variant();
	}
	const auto address = sandbox->cached_address_of(p_method.hash(), p_method);
	if (address == 0) {
		const bool found = sandbox->has_method(p_method);
		if (!found) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
			return Variant();
		}
		Array args;
		for (int i = 0; i < p_argument_count; i++) {
			args.push_back(*p_args[i]);
		}
		r_error.error = Callable::CallError::CALL_OK;
		return sandbox->callv(p_method, args);
	}
	ScopedTreeBase stb(sandbox, Object::cast_to<Node>(this->owner));
	return sandbox->vmcall_address(address, p_args, p_argument_count, r_error);
}

void SafeGDScriptInstance::get_method_list(List<MethodInfo> *p_list) const {
	for (const MethodInfo &method_info : script->methods_info) {
		if constexpr (VERBOSE_LOGGING) {
			ERR_PRINT("SafeGDScriptInstance::get_method_list: method " + String(method_info.name));
		}
		p_list->push_back(method_info);
	}
}

void SafeGDScriptInstance::get_property_list(List<PropertyInfo> *p_properties) const {
	auto [sandbox, created] = get_sandbox();
	if (!sandbox) {
		return;
	}

	std::vector<PropertyInfo> prop_list = sandbox->create_sandbox_property_list();

	// Sandboxed properties
	const std::vector<SandboxProperty> &properties = sandbox->get_properties();

	for (const SandboxProperty &property : properties) {
		if constexpr (VERBOSE_LOGGING) {
			printf("SafeGDScriptInstance::get_property_list %s\n", String(property.name()).utf8().ptr());
			fflush(stdout);
		}
		PropertyInfo prop;
		prop.name = property.name();
		prop.type = property.type();
		prop.usage = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_SCRIPT_VARIABLE | PROPERTY_USAGE_NIL_IS_VARIANT;
		p_properties->push_back(prop);
	}
	for (int i = 0; i < prop_list.size(); i++) {
		const PropertyInfo &prop = prop_list[i];
		if constexpr (VERBOSE_LOGGING) {
			printf("SafeGDScriptInstance::get_property_list %s\n", String(prop.name).utf8().ptr());
			fflush(stdout);
		}
		if (prop.name == StringName("program")) {
			continue;
		}
		p_properties->push_back(prop);
	}
}

Variant::Type SafeGDScriptInstance::get_property_type(const StringName &p_name, bool *r_is_valid) const {
	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("SafeGDScriptInstance::get_property_type " + p_name);
	}
	auto [sandbox, created] = get_sandbox();
	if (sandbox) {
		if (const SandboxProperty *prop = sandbox->find_property_or_null(p_name)) {
			if (r_is_valid) {
				*r_is_valid = true;
			}
			return prop->type();
		}
	}
	if (r_is_valid) {
		*r_is_valid = false;
	}
	return Variant::NIL;
}

void SafeGDScriptInstance::validate_property(PropertyInfo &p_property) const {
	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("SafeGDScriptInstance::validate_property");
	}
	// Property validation can be added here if needed
}

bool SafeGDScriptInstance::has_method(const StringName &p_method) const {
	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("SafeGDScriptInstance::has_method " + p_method);
	}
	for (const MethodInfo &method_info : script->methods_info) {
		if (method_info.name == p_method) {
			return true;
		}
	}
	return false;
}

int SafeGDScriptInstance::get_method_argument_count(const StringName &p_method, bool *r_is_valid) const {
	if (r_is_valid) {
		*r_is_valid = false;
	}
	for (const MethodInfo &method_info : script->methods_info) {
		if (method_info.name == p_method) {
			if (r_is_valid) {
				*r_is_valid = true;
			}
			return method_info.arguments.size();
		}
	}
	return 0;
}

bool SafeGDScriptInstance::property_can_revert(const StringName &p_name) const {
	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("SafeGDScriptInstance::property_can_revert " + p_name);
	}
	return false;
}

bool SafeGDScriptInstance::property_get_revert(const StringName &p_name, Variant &r_ret) const {
	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("SafeGDScriptInstance::property_get_revert " + p_name);
	}
	r_ret = Variant();
	return false;
}

void SafeGDScriptInstance::refcount_incremented() {
}

bool SafeGDScriptInstance::refcount_decremented() {
	return false;
}

Object *SafeGDScriptInstance::get_owner() {
	return owner;
}

Ref<Script> SafeGDScriptInstance::get_script() const {
	return script;
}

bool SafeGDScriptInstance::is_placeholder() const {
	return false;
}

ScriptLanguage *SafeGDScriptInstance::get_language() {
	return SafeGDScriptLanguage::get_singleton();
}

void SafeGDScriptInstance::reset_to(const PackedByteArray &p_elf_data) {
	auto [sandbox, created] = get_sandbox();
	if (sandbox) {
		sandbox->load_buffer(p_elf_data);
	}
}

static std::unordered_map<SafeGDScript *, Sandbox *> sandbox_instances;

std::tuple<Sandbox *, bool> SafeGDScriptInstance::get_sandbox() const {
	auto it = sandbox_instances.find(this->script.ptr());
	if (it != sandbox_instances.end()) {
		return { it->second, true };
	}

	Sandbox *sandbox_ptr = Object::cast_to<Sandbox>(this->owner);
	if (sandbox_ptr != nullptr) {
		return { sandbox_ptr, false };
	}

	ERR_PRINT("SafeGDScriptInstance: owner is not a Sandbox");
	if constexpr (VERBOSE_LOGGING) {
		fprintf(stderr, "SafeGDScriptInstance: owner is instead a '%s'!\n", this->owner->get_class().utf8().get_data());
	}
	return { nullptr, false };
}

static Sandbox *create_sandbox(Object *p_owner, const Ref<SafeGDScript> &p_script) {
	auto it = sandbox_instances.find(p_script.ptr());
	if (it != sandbox_instances.end()) {
		return it->second;
	}

	Sandbox *sandbox_ptr = memnew(Sandbox);
	sandbox_ptr->set_tree_base(Object::cast_to<Node>(p_owner));
	sandbox_ptr->set_unboxed_arguments(false);
	sandbox_ptr->load_buffer(p_script->get_content());
	sandbox_instances.insert_or_assign(p_script.ptr(), sandbox_ptr);
	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("SafeGDScriptInstance: created sandbox for " + Object::cast_to<Node>(p_owner)->get_name());
	}

	return sandbox_ptr;
}

SafeGDScriptInstance::SafeGDScriptInstance(Object *p_owner, const Ref<SafeGDScript> p_script) :
		owner(p_owner), script(p_script) {
	this->current_sandbox = Object::cast_to<Sandbox>(p_owner);
	this->auto_created_sandbox = (this->current_sandbox == nullptr);
	if (auto_created_sandbox) {
		this->current_sandbox = create_sandbox(p_owner, p_script);
	}
	if (this->current_sandbox) {
		this->current_sandbox->set_tree_base(Object::cast_to<Node>(owner));
	}
}

SafeGDScriptInstance::~SafeGDScriptInstance() {
	if (current_sandbox && auto_created_sandbox) {
		sandbox_instances.erase(script.ptr());
		memdelete(current_sandbox);
	}
	if (script.is_valid()) {
		script->remove_instance(this);
	}
}
