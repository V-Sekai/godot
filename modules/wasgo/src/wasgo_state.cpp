/**************************************************************************/
/*  wasgo_state.cpp                                                       */
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

#include "wasgo_state.h"
#include "core/io/marshalls.h"
#include "wasgo_callable.h"
#include "wasgo_runtime.h"
#include "wasm_export.h"
#include <cstdint>

void WasGoState::_initialize() {
	_stop();
	reference_object(this);
	ERR_FAIL_COND(!wasm_script.is_valid());
	String err_string;
	module_inst = WasGoRuntime::get_singleton()->instantiate_module(wasm_script->get_rid(),
			stack_size,
			heap_size,
			err_string);
	ERR_FAIL_COND_MSG(module_inst == nullptr, vformat("Failed to instantiate wasm module %s: \"%s\".", get_path(), err_string));

	exec_env = wasm_runtime_create_exec_env(module_inst, stack_size);
	wasm_runtime_set_user_data(exec_env, this);
	void *wasgo_func = wasm_runtime_lookup_function(module_inst, "test");
	printf("found func %p\n", wasgo_func);

	if ((notification_callback = wasm_runtime_lookup_function(module_inst, "_notification"))) {
		print_line("The notification callback found.");
	}
	if ((ready_callback = wasm_runtime_lookup_function(module_inst, "_ready"))) {
		print_line("The ready callback found.");
	}
	if ((process_callback = wasm_runtime_lookup_function(module_inst, "_process"))) {
		set_process(true);
		print_line("The process callback found.");
	}
	if ((physics_process_callback = wasm_runtime_lookup_function(module_inst, "_physics_process"))) {
		set_physics_process(true);
		print_line("The physics_process callback found.");
	}
	if (wasm_runtime_lookup_function(module_inst, "_input") && (input_callback = wasm_runtime_lookup_function(module_inst, "_wasgo_input"))) {
		set_process_input(true);
		print_line("The input callback found.");
	}
	if (wasm_runtime_lookup_function(module_inst, "_unhandled_input") && (unhandled_input_callback = wasm_runtime_lookup_function(module_inst, "_wasgo_unhandled_input"))) {
		set_process_unhandled_input(true);
		print_line("The unhandled_input callback found.");
	}
	if (wasm_runtime_lookup_function(module_inst, "_unhandled_key_input") && (unhandled_key_input_callback = wasm_runtime_lookup_function(module_inst, "_wasgo_unhandled_key_input"))) {
		set_process_unhandled_key_input(true);
		print_line("The unhandled_key_input callback found.");
	}
}

void WasGoState::_stop() {
	if (exec_env) {
		wasm_runtime_destroy_exec_env(exec_env);
	}
	if (module_inst) {
		if (wasm_buffer)
			wasm_runtime_module_free(module_inst, wasm_buffer);
		wasm_runtime_deinstantiate(module_inst);
	}
	last_id = 0;
	exec_env = nullptr;
	module_inst = nullptr;
	createdObjects.clear();
	createdObjectsReverse.clear();
	referencedObjects.clear();
	referencedObjectsReverse.clear();
	notification_callback = nullptr;
	ready_callback = nullptr;
	process_callback = nullptr;
	physics_process_callback = nullptr;
	input_callback = nullptr;
	unhandled_input_callback = nullptr;
	unhandled_key_input_callback = nullptr;
}

void WasGoState::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_wasm_script", "wasm_script"), &WasGoState::set_wasm_script);
	ClassDB::bind_method(D_METHOD("get_wasm_script"), &WasGoState::get_wasm_script);
	ClassDB::bind_method(D_METHOD("set_properties", "properties"), &WasGoState::set_properties);
	ClassDB::bind_method(D_METHOD("get_properties"), &WasGoState::get_properties);
	// ClassDB::bind_method(D_METHOD("set_property", "property", "value"), &WasGoState::set_property);
	// ClassDB::bind_method(D_METHOD("get_property", "property"), &WasGoState::get_property);
	ClassDB::bind_method(D_METHOD("set_stack_size", "stack_size"), &WasGoState::set_stack_size);
	ClassDB::bind_method(D_METHOD("get_stack_size"), &WasGoState::get_stack_size);
	ClassDB::bind_method(D_METHOD("set_heap_size", "heap_size"), &WasGoState::set_heap_size);
	ClassDB::bind_method(D_METHOD("get_heap_size"), &WasGoState::get_heap_size);

	// for the wrappers
	ClassDB::bind_method(D_METHOD("set_property_bool", "property", "value"), &WasGoState::set_property_bool);
	ClassDB::bind_method(D_METHOD("get_property_bool", "property"), &WasGoState::get_property_bool);
	ClassDB::bind_method(D_METHOD("set_property_int", "property", "value"), &WasGoState::set_property_int);
	ClassDB::bind_method(D_METHOD("get_property_int", "property"), &WasGoState::get_property_int);
	ClassDB::bind_method(D_METHOD("set_property_float", "property", "value"), &WasGoState::set_property_float);
	ClassDB::bind_method(D_METHOD("get_property_float", "property"), &WasGoState::get_property_float);
	ClassDB::bind_method(D_METHOD("set_property_string", "property", "value"), &WasGoState::set_property_string);
	ClassDB::bind_method(D_METHOD("get_property_string", "property"), &WasGoState::get_property_string);
	ClassDB::bind_method(D_METHOD("set_property_vector2", "property", "value"), &WasGoState::set_property_vector2);
	ClassDB::bind_method(D_METHOD("get_property_vector2", "property"), &WasGoState::get_property_vector2);
	ClassDB::bind_method(D_METHOD("set_property_rect2", "property", "value"), &WasGoState::set_property_rect2);
	ClassDB::bind_method(D_METHOD("get_property_rect2", "property"), &WasGoState::get_property_rect2);
	ClassDB::bind_method(D_METHOD("set_property_vector3", "property", "value"), &WasGoState::set_property_vector3);
	ClassDB::bind_method(D_METHOD("get_property_vector3", "property"), &WasGoState::get_property_vector3);
	ClassDB::bind_method(D_METHOD("set_property_transform2d", "property", "value"), &WasGoState::set_property_transform2d);
	ClassDB::bind_method(D_METHOD("get_property_transform2d", "property"), &WasGoState::get_property_transform2d);
	ClassDB::bind_method(D_METHOD("set_property_plane", "property", "value"), &WasGoState::set_property_plane);
	ClassDB::bind_method(D_METHOD("get_property_plane", "property"), &WasGoState::get_property_plane);
	ClassDB::bind_method(D_METHOD("set_property_quat", "property", "value"), &WasGoState::set_property_quat);
	ClassDB::bind_method(D_METHOD("get_property_quat", "property"), &WasGoState::get_property_quat);
	ClassDB::bind_method(D_METHOD("set_property_aabb", "property", "value"), &WasGoState::set_property_aabb);
	ClassDB::bind_method(D_METHOD("get_property_aabb", "property"), &WasGoState::get_property_aabb);
	ClassDB::bind_method(D_METHOD("set_property_basis", "property", "value"), &WasGoState::set_property_basis);
	ClassDB::bind_method(D_METHOD("get_property_basis", "property"), &WasGoState::get_property_basis);
	ClassDB::bind_method(D_METHOD("set_property_transform", "property", "value"), &WasGoState::set_property_transform);
	ClassDB::bind_method(D_METHOD("get_property_transform", "property"), &WasGoState::get_property_transform);
	ClassDB::bind_method(D_METHOD("set_property_color", "property", "value"), &WasGoState::set_property_color);
	ClassDB::bind_method(D_METHOD("get_property_color", "property"), &WasGoState::get_property_color);
	ClassDB::bind_method(D_METHOD("set_property_nodepath", "property", "value"), &WasGoState::set_property_nodepath);
	ClassDB::bind_method(D_METHOD("get_property_nodepath", "property"), &WasGoState::get_property_nodepath);

	ClassDB::bind_method(D_METHOD("set_int_property", "value", "key"), &WasGoState::set_int_property);

	// callbacks
	ClassDB::bind_method(D_METHOD("_input", "p_event"), &WasGoState::_input);
	ClassDB::bind_method(D_METHOD("_unhandled_input", "p_event"), &WasGoState::_unhandled_input);
	ClassDB::bind_method(D_METHOD("_unhandled_key_input", "p_event"), &WasGoState::_unhandled_key_input);

	ClassDB::bind_method(D_METHOD("get_callable", "func"), &WasGoState::get_callable);

	ADD_GROUP("script", "script_");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "script_binary", PROPERTY_HINT_RESOURCE_TYPE, "WasmResource"), "set_wasm_script", "get_wasm_script");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "script_properties", PROPERTY_HINT_NONE, ""), "set_properties", "get_properties");

	ADD_GROUP("runtime", "runtime_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "runtime_stack_size", PROPERTY_HINT_NONE, ""), "set_stack_size", "get_stack_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "runtime_heap_size", PROPERTY_HINT_NONE, ""), "set_heap_size", "get_heap_size");
	ADD_GROUP("sync", "sync_");
}

void WasGoState::_validate_property(PropertyInfo &property) const {
}
void WasGoState::_notification(uint32_t p_what) {
	if (!Engine::get_singleton()->is_editor_hint() || p_what == NOTIFICATION_READY) { // only run in game but use the ready function so that the properties autopopulate
		switch (p_what) {
			case NOTIFICATION_READY: {
				if (ready_callback) {
					if (!wasm_runtime_call_wasm(exec_env, ready_callback, 0, nullptr)) {
						printf("Wasm ready callback failed. %s\n", wasm_runtime_get_exception(module_inst));
					}
				}
			} break;
			case NOTIFICATION_INTERNAL_PROCESS:
			case NOTIFICATION_PROCESS: {
				if (process_callback) {
					float delta = get_process_delta_time();
					uint32_t argv[2];
					memcpy(argv, &delta, sizeof(delta));
					if (!wasm_runtime_call_wasm(exec_env, process_callback, 1, argv)) {
						printf("Wasm process callback failed. %s\n", wasm_runtime_get_exception(module_inst));
					}
				}
			} break;
			case NOTIFICATION_INTERNAL_PHYSICS_PROCESS:
			case NOTIFICATION_PHYSICS_PROCESS: {
				if (physics_process_callback) {
					float delta = get_physics_process_delta_time();
					uint32_t argv[2];
					memcpy(argv, &delta, sizeof(delta));
					if (!wasm_runtime_call_wasm(exec_env, physics_process_callback, 1, argv)) {
						printf("Wasm physics process callback failed. %s\n", wasm_runtime_get_exception(module_inst));
					}
				}

			} break;
			default: {
			} break;
		}

		if (notification_callback) {
			uint32_t argv[2] = { 0, p_what }; // argv[0] is the return value
			if (!wasm_runtime_call_wasm(exec_env, notification_callback, 1, argv)) {
				printf("wasm notification callback failed. %s\n", wasm_runtime_get_exception(module_inst));
			}
		}
	}
}

WasGoState::WasGoState() {
	properties = {};
	stack_size = 8192;
	heap_size = 8192;
	_stop();
}

WasGoState::~WasGoState() {
	_stop();
}

Variant WasGoState::lookup_object(WasGoID id) {
	if (createdObjectsReverse.has(id)) {
		return createdObjectsReverse[id];
	} else if (referencedObjectsReverse.has(id)) {
		return referencedObjectsReverse[id];
	}
	Array print_arr;
	print_arr.append(id);
	print_error(String("Invalid WasGoID: {0}").format(print_arr));
	return Variant();
}
Variant WasGoState::lookup_createdObject(WasGoID id) {
	if (createdObjectsReverse.has(id)) {
		return createdObjectsReverse[id];
	}
	Array print_arr;
	print_arr.append(id);
	print_error(String("Invalid WasGoID: {0}").format(print_arr));
	return Variant();
}

Variant WasGoState::lookup_referencedObject(WasGoID id) {
	if (referencedObjectsReverse.has(id)) {
		return referencedObjectsReverse[id];
	}
	Array print_arr;
	print_arr.append(id);
	print_error(String("Invalid WasGoID: {0}").format(print_arr));
	return Variant();
}
// Variant *WasGoState::lookup_variant(WasGoID id){
//     if(createdVariantsReverse.has(id)){
// 		return &createdVariantsReverse[id];
// 	}
// 	return nullptr;
// }

WasGoState::WasGoID WasGoState::lookup_wasgo_object(Variant obj) {
	if (createdObjects.has(obj)) {
		return createdObjects[obj];
	} else if (referencedObjects.has(obj)) {
		return referencedObjects[obj];
	}
	print_error("Invalid Object Lookup");
	return 0;
}
WasGoState::WasGoID WasGoState::lookup_wasgo_createdObject(Variant obj) {
	if (createdObjects.has(obj)) {
		return createdObjects[obj];
	}
	print_error("Invalid Object Lookup");
	return 0;
}
WasGoState::WasGoID WasGoState::lookup_wasgo_referencedObject(Variant obj) {
	if (referencedObjects.has(obj)) {
		return referencedObjects[obj];
	}
	print_error("Invalid Object Lookup");
	return 0;
}

bool WasGoState::is_active() {
	return module_inst && exec_env;
}

void WasGoState::set_stack_size(int p_stack_size) {
	// I don't think you can dynamically change the stack and heap sizes, so we're gonna only change it if the wasm module is not active
	if (!is_active()) {
		stack_size = p_stack_size;
	}
}
int WasGoState::get_stack_size() {
	return stack_size;
}

void WasGoState::set_heap_size(int p_heap_size) {
	// I don't think you can dynamically change the stack and heap sizes, so we're gonna only change it if the wasm module is not active
	if (!is_active()) {
		heap_size = p_heap_size;
	}
}
int WasGoState::get_heap_size() {
	return heap_size;
}

void WasGoState::set_wasm_script(Ref<WasmResource> p_wasm_script) {
	// Only change it if the wasm module is not active
	ERR_FAIL_COND(is_active()); // Fail instead of stopping.
	if (!is_active()) {
		_stop();
	}
	wasm_script = p_wasm_script;
	_initialize();
	if (is_inside_tree() && ready_callback) { // call the ready callback again because we must have missed the first one
		if (!wasm_runtime_call_wasm(exec_env, ready_callback, 0, nullptr)) {
			printf("wasm ready callback failed. %s\n", wasm_runtime_get_exception(module_inst));
		}
	}
}
Ref<WasmResource> WasGoState::get_wasm_script() {
	return wasm_script;
}

void WasGoState::set_properties(Dictionary p_properties) {
	// I don't think you can dynamically change the stack and heap sizes, so we're gonna only change it if the wasm module is not active
	//  if (!is_active()) {
	properties = p_properties;
	// }
}
Dictionary WasGoState::get_properties() {
	return properties;
}
void WasGoState::set_property(String key, Variant value) {
	// I don't think you can dynamically change the stack and heap sizes, so we're gonna only change it if the wasm module is not active
	//  if (!is_active()) {
	properties[key] = value;
	// }
}
Variant WasGoState::get_property(String key) {
	return properties[key];
}

WasGoState::WasGoID WasGoState::generate_id() {
	last_id++;
	return last_id;
}

WasGoState::WasGoID WasGoState::create_object(Variant obj) {
	WasGoID wasgo_id = 0;
	if (obj.get_type() != Variant::OBJECT || !obj.is_zero()) {
		if (createdObjects.has(obj)) {
			wasgo_id = createdObjects[obj];
		} else if (referencedObjects.has(obj)) {
			wasgo_id = referencedObjects[obj];
		} else {
			wasgo_id = generate_id();
			// createdObjects.set(obj, wasgo_id);
			// createdObjectsReverse.set(wasgo_id, obj);
			createdObjects[obj] = wasgo_id;
			createdObjectsReverse[wasgo_id] = obj;
		}
	}
	return wasgo_id;
}

WasGoState::WasGoID WasGoState::reference_object(Variant obj) {
	WasGoID wasgo_id = 0;
	if (obj.get_type() != Variant::OBJECT || !obj.is_zero()) {
		if (referencedObjects.has(obj)) {
			wasgo_id = referencedObjects[obj];
		}
		if (createdObjects.has(obj)) {
			wasgo_id = createdObjects[obj];
		} else {
			wasgo_id = generate_id();
			// referencedObjects.set(obj, wasgo_id);
			// referencedObjectsReverse.set(wasgo_id, obj);
			referencedObjects[obj] = wasgo_id;
			referencedObjectsReverse[wasgo_id] = obj;
		}
	}
	return wasgo_id;
}

// WasGoState::WasGoID WasGoState::create_variant(Variant var){
// 	WasGoID wasgo_id = generate_id();
// 	// createdVariants.set(var, wasgo_id);
// 	createdVariantsReverse.set(wasgo_id, var);
// 	return wasgo_id;
// }

// WasGoState::WasGoID WasGoState::create_object(Object obj) {
// 	WasGoID wasgo_id = 0;
// 	if (createdObjects.has(obj.get_instance_id())) {
// 		wasgo_id = createdObjects[wasgo_id];
// 	} else {
// 		wasgo_id = generate_id();
// 		createdObjects.set(obj.get_instance_id(), wasgo_id);
// 		createdObjectsReverse.set(wasgo_id, obj.get_instance_id());
// 	}
// 	return wasgo_id;
// }

// WasGoState::WasGoID WasGoState::reference_object(Object obj) {
// 	return reference_object(obj.get_instance_id());
// }

// WasGoState::WasGoID WasGoState::reference_object(Ref<Object> ref) {
// 	WasGoID wasgo_id = 0;
// 	//TODO: Figure out how to handle this case
// 	// if (referencedObjects.has(obj_id)) {
// 	// 	wasgo_id = referencedObjects[wasgo_id];
// 	// } else {
// 	// 	wasgo_id = generate_id();
// 	// 	referencedObjects.set(obj_id, wasgo_id);
// 	// 	referencedObjectsReverse.set(wasgo_id, obj_id);
// 	// }
// 	return wasgo_id;
// }

WasGoState::WasGoID WasGoState::handle_return_variant(Variant var) {
	// WasGoID id = 0;
	// if(var.is_ref()){
	// 	if(var.get_type() == Variant::OBJECT){
	// 		RefPtr ref = RefPtr(var);
	// 		Object *obj = (Object *)ref.get_data();
	// 		id = reference_object(obj);
	// 	} else {
	// 		//We shouldn't be able to get here
	// 		printf("ERROR: unexpected ref type.");
	// 	}
	// } else {
	// 	// if (var.get_type() == Variant::OBJECT) {
	// 	// 	Object *obj = (Object *)var;
	// 	// 	if (obj) {
	// 	// 		id = create_object(obj);
	// 	// 	}
	// 	// } else {
	// 	// 	id = create_variant(var);
	// 	// }
	// 	id = create_object(var);
	// }
	// return id;
	return create_object(var);
}

// Regular Node Callbacks
//  void WasGoState::_enter_tree(){
//      if(is_active()){
//          call_wasm_function("_enter_tree");
//      }
//  }
//  void WasGoState::_exit_tree(){
//      if(is_active()){
//          call_wasm_function("_exit_tree");
//      }
//  }
//  String WasGoState::_get_configuration_warning(){
//      if(is_active()){
//          call_wasm_function("_get_configuration_warning");
//      }
//  }
void WasGoState::_input(const Ref<InputEvent> &p_event) {
	if (input_callback) {
		uint32_t argv[1];
		argv[0] = reference_object(p_event);
		if (!wasm_runtime_call_wasm(exec_env, input_callback, 1, argv)) {
			printf("wasm input callback failed. %s\n", wasm_runtime_get_exception(module_inst));
		}
	}
}
// void WasGoState::_physics_process(float delta){
//     if(is_active()){
//         call_wasm_function("_physics_process");
//     }
// }
// void WasGoState::_process(float delta){
//     if(is_active()){
//         call_wasm_function("_process");
//     }
// }
// void WasGoState::_ready(){
//     if(is_active()){
//         call_wasm_function("_ready");
//     }
// }
void WasGoState::_unhandled_input(Ref<InputEvent> p_event) {
	if (unhandled_input_callback) {
		uint32_t argv[1];
		argv[0] = reference_object(p_event);
		if (!wasm_runtime_call_wasm(exec_env, unhandled_input_callback, 1, argv)) {
			printf("wasm unhandled input callback failed. %s\n", wasm_runtime_get_exception(module_inst));
		}
	}
}
void WasGoState::_unhandled_key_input(Ref<InputEventKey> p_event) {
	if (unhandled_key_input_callback) {
		uint32_t argv[1];
		WasGoID wasgo_id = reference_object(p_event.ptr());
		argv[0] = wasgo_id;
		if (!wasm_runtime_call_wasm(exec_env, unhandled_key_input_callback, 1, argv)) {
			printf("wasm unhandled key input callback failed. %s\n", wasm_runtime_get_exception(module_inst));
		}
	}
}

WasGoState::WasGoID _wasgo_this_node(wasm_exec_env_t p_exec_env) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	if (state) {
		return state->lookup_wasgo_object(state);
	}
	return 0;
}

int _wasgo_get_property_bool(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	return (int)state->get_property(name);
}
void _wasgo_set_property_bool(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, int value) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	state->set_property(name, value);
}

int _wasgo_get_property_int(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	return (int)state->get_property(name);
}
void _wasgo_set_property_int(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, int value) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	state->set_property(name, value);
}

float _wasgo_get_property_float(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	return (float)state->get_property(name);
}
void _wasgo_set_property_float(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, float value) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	state->set_property(name, value);
}

void _wasgo_get_property_string(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	String ret = state->get_property(name);
	encode_variant(ret, value, value_size);
}
void _wasgo_set_property_string(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	Variant value_var;
	decode_variant(value_var, value, value_size);
	state->set_property(name, value_var);
}

void _wasgo_get_property_vector2(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	Vector2 ret = state->get_property(name);
	encode_variant(ret, value, value_size);
}
void _wasgo_set_property_vector2(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	Variant value_var;
	decode_variant(value_var, value, value_size);
	state->set_property(name, value_var);
}

void _wasgo_get_property_rect2(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	Rect2 ret = state->get_property(name);
	encode_variant(ret, value, value_size);
}
void _wasgo_set_property_rect2(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	Variant value_var;
	decode_variant(value_var, value, value_size);
	state->set_property(name, value_var);
}

void _wasgo_get_property_vector3(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	Vector3 ret = state->get_property(name);
	encode_variant(ret, value, value_size);
}
void _wasgo_set_property_vector3(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	Variant value_var;
	decode_variant(value_var, value, value_size);
	state->set_property(name, value_var);
}

void _wasgo_get_property_transform2d(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	Transform2D ret = state->get_property(name);
	encode_variant(ret, value, value_size);
}
void _wasgo_set_property_transform2d(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	Variant value_var;
	decode_variant(value_var, value, value_size);
	state->set_property(name, value_var);
}

void _wasgo_get_property_plane(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	Plane ret = state->get_property(name);
	encode_variant(ret, value, value_size);
}
void _wasgo_set_property_plane(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	Variant value_var;
	decode_variant(value_var, value, value_size);
	state->set_property(name, value_var);
}

void _wasgo_get_property_quat(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	Quaternion ret = state->get_property(name);
	encode_variant(ret, value, value_size);
}
void _wasgo_set_property_quat(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	Variant value_var;
	decode_variant(value_var, value, value_size);
	state->set_property(name, value_var);
}

void _wasgo_get_property_basis(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	Basis ret = state->get_property(name);
	encode_variant(ret, value, value_size);
}
void _wasgo_set_property_basis(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	Variant value_var;
	decode_variant(value_var, value, value_size);
	state->set_property(name, value_var);
}

void _wasgo_get_property_aabb(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	AABB ret = state->get_property(name);
	encode_variant(ret, value, value_size);
}
void _wasgo_set_property_aabb(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	Variant value_var;
	decode_variant(value_var, value, value_size);
	state->set_property(name, value_var);
}

void _wasgo_get_property_transform(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	Transform3D ret = state->get_property(name);
	encode_variant(ret, value, value_size);
}
void _wasgo_set_property_transform(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	Variant value_var;
	decode_variant(value_var, value, value_size);
	state->set_property(name, value_var);
}

void _wasgo_get_property_color(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	Color ret = state->get_property(name);
	encode_variant(ret, value, value_size);
}
void _wasgo_set_property_color(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	Variant value_var;
	decode_variant(value_var, value, value_size);
	state->set_property(name, value_var);
}

void _wasgo_get_property_nodepath(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	NodePath ret = state->get_property(name);
	encode_variant(ret, value, value_size);
}
void _wasgo_set_property_nodepath(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	Variant value_var;
	decode_variant(value_var, value, value_size);
	state->set_property(name, value_var);
}

WasGoState::WasGoID _wasgo_get_property_object(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	Object *ret = state->get_property(name);
	return state->lookup_wasgo_object(ret);
}
void _wasgo_set_property_object(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, WasGoState::WasGoID p_wasgo_id) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	Variant name = String();
	decode_variant(name, property_name, property_name_size);
	Object *obj = state->lookup_object(p_wasgo_id);
	if (obj) {
		state->set_property(name, obj);
	}
}

void _wasgo_set_process(wasm_exec_env_t p_exec_env, bool p_enable) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	state->set_process(p_enable);
}
void _wasgo_set_physics_process(wasm_exec_env_t p_exec_env, bool p_enable) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	state->set_physics_process(p_enable);
}
void _wasgo_set_process_internal(wasm_exec_env_t p_exec_env, bool p_enable) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	state->set_process_internal(p_enable);
}
void _wasgo_set_physics_process_internal(wasm_exec_env_t p_exec_env, bool p_enable) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	state->set_physics_process_internal(p_enable);
}
void _wasgo_set_process_input(wasm_exec_env_t p_exec_env, bool p_enable) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	state->set_process_input(p_enable);
}
void _wasgo_set_process_unhandled_input(wasm_exec_env_t p_exec_env, bool p_enable) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	state->set_process_unhandled_input(p_enable);
}
void _wasgo_set_process_unhandled_key_input(wasm_exec_env_t p_exec_env, bool p_enable) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	state->set_process_unhandled_key_input(p_enable);
}

bool _wasgo_is_processing(wasm_exec_env_t p_exec_env) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	return state->is_processing();
}
bool _wasgo_is_physics_processing(wasm_exec_env_t p_exec_env) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	return state->is_physics_processing();
}
bool _wasgo_is_processing_internal(wasm_exec_env_t p_exec_env) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	return state->is_processing_internal();
}
bool _wasgo_is_physics_processing_internal(wasm_exec_env_t p_exec_env) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	return state->is_physics_processing_internal();
}
bool _wasgo_is_processing_input(wasm_exec_env_t p_exec_env) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	return state->is_processing_input();
}
bool _wasgo_is_processing_unhandled_input(wasm_exec_env_t p_exec_env) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	return state->is_processing_unhandled_input();
}
bool _wasgo_is_processing_unhandled_key_input(wasm_exec_env_t p_exec_env) {
	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(p_exec_env);
	return state->is_processing_unhandled_key_input();
}

bool WasGoState::get_property_bool(String key) {
	return get_property(key);
}
int WasGoState::get_property_int(String key) {
	return get_property(key);
}
float WasGoState::get_property_float(String key) {
	return get_property(key);
}
String WasGoState::get_property_string(String key) {
	return get_property(key);
}
Vector2 WasGoState::get_property_vector2(String key) {
	return get_property(key);
}
Rect2 WasGoState::get_property_rect2(String key) {
	return get_property(key);
}
Vector3 WasGoState::get_property_vector3(String key) {
	return get_property(key);
}
Transform2D WasGoState::get_property_transform2d(String key) {
	return get_property(key);
}
Plane WasGoState::get_property_plane(String key) {
	return get_property(key);
}
Quaternion WasGoState::get_property_quat(String key) {
	return get_property(key);
}
AABB WasGoState::get_property_aabb(String key) {
	return get_property(key);
}
Basis WasGoState::get_property_basis(String key) {
	return get_property(key);
}
Transform3D WasGoState::get_property_transform(String key) {
	return get_property(key);
}
Color WasGoState::get_property_color(String key) {
	return get_property(key);
}
NodePath WasGoState::get_property_nodepath(String key) {
	return get_property(key);
}

void WasGoState::set_property_bool(String key, bool p_value) {
	set_property(key, p_value);
}
void WasGoState::set_property_int(String key, int p_value) {
	set_property(key, p_value);
}
void WasGoState::set_property_float(String key, float p_value) {
	set_property(key, p_value);
}
void WasGoState::set_property_string(String key, String p_value) {
	set_property(key, p_value);
}
void WasGoState::set_property_vector2(String key, Vector2 p_value) {
	set_property(key, p_value);
}
void WasGoState::set_property_rect2(String key, Rect2 p_value) {
	set_property(key, p_value);
}
void WasGoState::set_property_vector3(String key, Vector3 p_value) {
	set_property(key, p_value);
}
void WasGoState::set_property_transform2d(String key, Transform2D p_value) {
	set_property(key, p_value);
}
void WasGoState::set_property_plane(String key, Plane p_value) {
	set_property(key, p_value);
}
void WasGoState::set_property_quat(String key, Quaternion p_value) {
	set_property(key, p_value);
}
void WasGoState::set_property_aabb(String key, AABB p_value) {
	set_property(key, p_value);
}
void WasGoState::set_property_basis(String key, Basis p_value) {
	set_property(key, p_value);
}
void WasGoState::set_property_transform(String key, Transform3D p_value) {
	set_property(key, p_value);
}
void WasGoState::set_property_color(String key, Color p_value) {
	set_property(key, p_value);
}
void WasGoState::set_property_nodepath(String key, NodePath p_value) {
	set_property(key, p_value);
}

void WasGoState::set_int_property(int p_value, String key) {
	set_property(key, p_value);
}

Callable WasGoState::get_callable(String p_func) {
	return (Callable)memnew(WasGoCallable(this, p_func));
}
