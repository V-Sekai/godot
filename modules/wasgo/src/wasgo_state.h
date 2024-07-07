/**************************************************************************/
/*  wasgo_state.h                                                         */
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

/*
This file defines the WasGoState class.
A WasGoState contains everything needed to run a wasm script.
When creating a game with Wasm, the user makes these nodes and puts them in their scene.
These get saved with the scene and when the game is run they're used by the Wasm Runtime Singleton to create working wasm instances.

These nodes also handle things like translating wasm commands to Godot functions.
*/

#ifndef WASGO_STATE_H
#define WASGO_STATE_H

#include "core/variant/variant.h"
#include "resource_wasm.h"
#include "scene/main/node.h"
#include "wasm_export.h"

class WasGoState : public Node {
	friend class WasGoCallable;
	GDCLASS(WasGoState, Node);

	// Properties
	Ref<WasmResource> wasm_script;
	Dictionary properties = {}; // Properties to be passed to the script when it starts up
	uint32_t stack_size = 8192;
	uint32_t heap_size = 8192;

public:
	enum InterpolationType {
		NONE,
		LERP,
		SLERP
	};
	typedef uint32_t WasGoID;

#define NULL_WASGO_ID 0
	struct _EncodedVariant {
		WasGoID _id = NULL_WASGO_ID; // data exists on the godot side, we don't need to actually encode anything
		std::string _data = ""; // data encoded as a string, we will need to decode and store on wasm side
		_EncodedVariant() {
		}
		_EncodedVariant(WasGoID p_id) {
			_id = p_id;
		}
		_EncodedVariant(std::string p_data) {
			_data = p_data;
		}
	};

	WasGoState();
	~WasGoState();

	void _initialize();
	void _stop();

	void _validate_property(PropertyInfo &property) const;
	void _notification(uint32_t p_what);
	static void _bind_methods();

	void set_stack_size(int p_stack_size);
	int get_stack_size();

	void set_heap_size(int p_heap_size);
	int get_heap_size();

	void set_wasm_script(Ref<WasmResource> p_wasm_script);
	Ref<WasmResource> get_wasm_script();

	void set_properties(Dictionary p_properties);
	Dictionary get_properties();
	void set_property(String key, Variant value);
	Variant get_property(String key);
	// different getters and setters for different types to make wrapping easier
	bool get_property_bool(String key);
	int get_property_int(String key);
	float get_property_float(String key);
	String get_property_string(String key);
	Vector2 get_property_vector2(String key);
	Rect2 get_property_rect2(String key);
	Vector3 get_property_vector3(String key);
	Transform2D get_property_transform2d(String key);
	Plane get_property_plane(String key);
	Quaternion get_property_quat(String key);
	AABB get_property_aabb(String key);
	Basis get_property_basis(String key);
	Transform3D get_property_transform(String key);
	Color get_property_color(String key);
	NodePath get_property_nodepath(String key);

	void set_property_bool(String key, bool p_value);
	void set_property_int(String key, int p_value);
	void set_property_float(String key, float p_value);
	void set_property_string(String key, String p_value);
	void set_property_vector2(String key, Vector2 p_value);
	void set_property_rect2(String key, Rect2 p_value);
	void set_property_vector3(String key, Vector3 p_value);
	void set_property_transform2d(String key, Transform2D p_value);
	void set_property_plane(String key, Plane p_value);
	void set_property_quat(String key, Quaternion p_value);
	void set_property_aabb(String key, AABB p_value);
	void set_property_basis(String key, Basis p_value);
	void set_property_transform(String key, Transform3D p_value);
	void set_property_color(String key, Color p_value);
	void set_property_nodepath(String key, NodePath p_value);

	void set_int_property(int p_value, String key);

	void init(wasm_module_t instance);

	// Starting from here
	int call_wasm_function(String func_name, int argc = 0, uint32_t *argv = nullptr);

	int call_object_function(WasGoID object_id, int argc, uint32_t *argv);
	// int call_variant_function(WasGoID object_id, int argc, uint32_t *argv);
	// int call_array_function(WasGoID array_id, int argc, uint32_t *argv);
	// int call_dictionary_function(WasGoID dictionary_id, int argc, uint32_t *argv);

	Variant lookup_object(WasGoID id);
	Variant lookup_createdObject(WasGoID id);
	Variant lookup_referencedObject(WasGoID id);
	// Variant *lookup_variant(WasGoID id);

	WasGoID lookup_wasgo_object(Variant obj);
	WasGoID lookup_wasgo_createdObject(Variant obj);
	WasGoID lookup_wasgo_referencedObject(Variant obj);
	// WasGoID lookup_variant_reverse(Variant *);

	bool is_active();

	// the regular node callbacks
	//  void _enter_tree();
	//  void _exit_tree();
	//  String _get_configuration_warning();
	//  void _input(InputEvent event);
	//  void _physics_process(float delta);
	//  void _process(float delta);
	//  void _ready();
	//  void _unhandled_input(InputEvent event);
	//  void _unhandled_key_input(InputEventKey event);

	WasGoID generate_id();
	// WasGoID create_variant(Variant var);
	WasGoID create_object(Variant obj);
	WasGoID reference_object(Variant obj);
	// WasGoID reference_object(Ref<Object> ref);

	WasGoID handle_return_variant(Variant var);

	void _input(const Ref<InputEvent> &p_event);
	void _unhandled_input(Ref<InputEvent> p_event);
	void _unhandled_key_input(Ref<InputEventKey> p_event);

	Callable get_callable(String p_func);

private:
	wasm_module_inst_t module_inst = NULL;
	wasm_exec_env_t exec_env = NULL;
	uint32_t wasm_buffer = 0;

	WasGoID last_id = 0;

	Ref<WasmResource> wasm_resource;

	wasm_function_inst_t notification_callback = nullptr;
	wasm_function_inst_t ready_callback = nullptr;
	wasm_function_inst_t process_callback = nullptr;
	wasm_function_inst_t physics_process_callback = nullptr;
	wasm_function_inst_t input_callback = nullptr;
	wasm_function_inst_t unhandled_input_callback = nullptr;
	wasm_function_inst_t unhandled_key_input_callback = nullptr;

	Dictionary createdObjects;
	Dictionary createdObjectsReverse;
	Dictionary referencedObjects;
	Dictionary referencedObjectsReverse;

	// Only for special types that are not Objects. No referencing allowed. Everything is pass by value
	//  HashMap<Variant, WasGoID> createdVariants;
	//  HashMap<WasGoID, Variant> createdVariantsReverse;

	HashMap<WasGoID, InterpolationType> networkedVariants;

	// HashMap<WasGoID, int> createdObjectCounts;
	// HashMap<WasGoID, int> referencedObjectCounts;
	// HashMap<WasGoID, int> referencedArraysCounts;
	// HashMap<WasGoID, int> referencedDictionaryCounts;
};

// WRAPPERS
WasGoState::WasGoID _wasgo_this_node(wasm_exec_env_t exec_env);

int _wasgo_get_property_bool(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size);
void _wasgo_set_property_bool(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, int value);

int _wasgo_get_property_int(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size);
void _wasgo_set_property_int(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, int value);

float _wasgo_get_property_float(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size);
void _wasgo_set_property_float(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, float value);

void _wasgo_get_property_string(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size);
void _wasgo_set_property_string(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size);

void _wasgo_get_property_vector2(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size);
void _wasgo_set_property_vector2(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size);

void _wasgo_get_property_rect2(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size);
void _wasgo_set_property_rect2(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size);

void _wasgo_get_property_vector3(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size);
void _wasgo_set_property_vector3(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size);

void _wasgo_get_property_transform2d(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size);
void _wasgo_set_property_transform2d(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size);

void _wasgo_get_property_plane(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size);
void _wasgo_set_property_plane(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size);

void _wasgo_get_property_quat(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size);
void _wasgo_set_property_quat(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size);

void _wasgo_get_property_basis(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size);
void _wasgo_set_property_basis(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size);

void _wasgo_get_property_aabb(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size);
void _wasgo_set_property_aabb(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size);

void _wasgo_get_property_transform(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size);
void _wasgo_set_property_transform(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size);

void _wasgo_get_property_color(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size);
void _wasgo_set_property_color(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size);

void _wasgo_get_property_nodepath(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size);
void _wasgo_set_property_nodepath(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, uint8_t *value, int value_size);

WasGoState::WasGoID _wasgo_get_property_object(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size);
void _wasgo_set_property_object(wasm_exec_env_t p_exec_env, const uint8_t *property_name, int property_name_size, WasGoState::WasGoID p_wasgo_id);

void _wasgo_set_process(wasm_exec_env_t p_exec_env, bool p_enable);
void _wasgo_set_physics_process(wasm_exec_env_t p_exec_env, bool p_enable);
void _wasgo_set_process_internal(wasm_exec_env_t p_exec_env, bool p_enable);
void _wasgo_set_physics_process_internal(wasm_exec_env_t p_exec_env, bool p_enable);
void _wasgo_set_process_input(wasm_exec_env_t p_exec_env, bool p_enable);
void _wasgo_set_process_unhandled_input(wasm_exec_env_t p_exec_env, bool p_enable);
void _wasgo_set_process_unhandled_key_input(wasm_exec_env_t p_exec_env, bool p_enable);
bool _wasgo_is_processing(wasm_exec_env_t p_exec_env);
bool _wasgo_is_physics_processing(wasm_exec_env_t p_exec_env);
bool _wasgo_is_processing_internal(wasm_exec_env_t p_exec_env);
bool _wasgo_is_physics_processing_internal(wasm_exec_env_t p_exec_env);
bool _wasgo_is_processing_input(wasm_exec_env_t p_exec_env);
bool _wasgo_is_processing_unhandled_input(wasm_exec_env_t p_exec_env);
bool _wasgo_is_processing_unhandled_key_input(wasm_exec_env_t p_exec_env);

#endif