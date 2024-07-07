/**************************************************************************/
/*  wasgo_native_wrappers.h                                               */
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

#ifndef WASGO_NATIVE_WRAPPERS_H
#define WASGO_NATIVE_WRAPPERS_H
#include "src/wasgo_state.h"
#include "wasm_export.h"
// #include "core/io/marshalls.h"
// #include "core/variant/variant.h"
// #include "scene/register_scene_types.h"
// #include "scene/scene_string_names.h"
#include "api/shared/wasgo_encode_decode.h"
// // #include "scene/2d/animated_sprite.h"
// #include "scene/2d/area_2d.h"
// #include "scene/2d/audio_stream_player_2d.h"
// #include "scene/2d/back_buffer_copy.h"
// #include "scene/2d/camera_2d.h"
// // #include "scene/2d/canvas_item.h"
// #include "scene/2d/canvas_modulate.h"
// #include "scene/2d/collision_object_2d.h"
// #include "scene/2d/collision_polygon_2d.h"
// #include "scene/2d/collision_shape_2d.h"
// #include "scene/2d/cpu_particles_2d.h"
// // #include "scene/2d/joints_2d.h"
// #include "scene/2d/light_2d.h"
// #include "scene/2d/light_occluder_2d.h"
// #include "scene/2d/line_2d.h"
// #include "scene/2d/line_builder.h"
// #include "scene/2d/mesh_instance_2d.h"
// #include "scene/2d/multimesh_instance_2d.h"
// // #include "scene/2d/navigation_2d.h"
// // #include "scene/2d/navigation_polygon.h"
// #include "scene/2d/node_2d.h"
// #include "scene/2d/parallax_background.h"
// #include "scene/2d/parallax_layer.h"
// #include "scene/2d/gpu_particles_2d.h"
// #include "scene/2d/path_2d.h"
// #include "scene/2d/physics_body_2d.h"
// #include "scene/2d/polygon_2d.h"
// #include "scene/2d/position_2d.h"
// #include "scene/2d/ray_cast_2d.h"
// #include "scene/2d/remote_transform_2d.h"
// #include "scene/2d/skeleton_2d.h"
// #include "scene/2d/sprite_2d.h"
// #include "scene/2d/tile_map.h"
// #include "scene/2d/touch_screen_button.h"
// #include "servers/rendering/renderer_canvas_cull.h"
// #include "scene/3d/area_3d.h"
// #include "scene/3d/xr_nodes.h"
// #include "scene/3d/audio_stream_player_3d.h"
// // #include "scene/3d/baked_lightmap.h"
// #include "scene/3d/bone_attachment_3d.h"
// #include "scene/3d/camera_3d.h"
// #include "scene/3d/collision_object_3d.h"
// #include "scene/3d/collision_polygon_3d.h"
// #include "scene/3d/collision_shape_3d.h"
// #include "scene/3d/cpu_particles_3d.h"
// // #include "scene/3d/gi_probe.h"
// // #include "scene/3d/immediate_geometry_3d.h"
// // #include "scene/3d/interpolated_camera.h"
// #include "core/input/input.h"
// #include "scene/3d/light_3d.h"
// // #include "scene/3d/listener_3d.h"
// #include "scene/3d/mesh_instance_3d.h"
// #include "scene/3d/multimesh_instance_3d.h"
// #include "scene/3d/navigation_agent_3d.h"
// #include "scene/3d/navigation_region_3d.h"
// #include "scene/3d/navigation_obstacle_3d.h"
// #include "scene/3d/gpu_particles_3d.h"
// #include "scene/3d/path_3d.h"
// #include "scene/3d/physics_body_3d.h"
// // #include "scene/3d/physics_joint_3d.h"
// #include "scene/3d/position_3d.h"
// #include "scene/3d/proximity_group_3d.h"
// #include "scene/3d/ray_cast_3d.h"
// #include "scene/3d/reflection_probe.h"
// #include "scene/3d/remote_transform_3d.h"
// #include "scene/3d/skeleton_3d.h"
// // #include "scene/3d/soft_body_3d.h"
// #include "scene/3d/node_3d.h"
// #include "scene/3d/velocity_tracker_3d.h"
// #include "scene/3d/spring_arm_3d.h"
// #include "scene/3d/sprite_3d.h"
// #include "scene/3d/vehicle_body_3d.h"
// // #include "scene/3d/visibility_notifier_3d.h"
// #include "scene/3d/visual_instance_3d.h"
// #include "scene/3d/voxelizer.h"
// #include "scene/3d/world_environment.h"
// #include "scene/animation/animation_blend_space_1d.h"
// #include "scene/animation/animation_blend_space_2d.h"
// #include "scene/animation/animation_blend_tree.h"
// // #include "scene/animation/animation_cache.h"
// #include "scene/animation/animation_node_state_machine.h"
// #include "scene/animation/animation_player.h"
// #include "scene/animation/animation_tree.h"
// #include "scene/animation/root_motion_view.h"
// #include "scene/3d/skeleton_ik_3d.h"
// #include "scene/animation/tween.h"
// #include "scene/audio/audio_stream_player.h"
// #include "scene/debugger/scene_debugger.h"
// #include "scene/gui/base_button.h"
// #include "scene/gui/box_container.h"
// #include "scene/gui/button.h"
// #include "scene/gui/center_container.h"
// #include "scene/gui/check_box.h"
// #include "scene/gui/check_button.h"
// #include "scene/gui/color_picker.h"
// #include "scene/gui/color_rect.h"
// #include "scene/gui/container.h"
// #include "scene/gui/control.h"
// #include "scene/gui/dialogs.h"
// #include "scene/gui/file_dialog.h"
// #include "scene/gui/gradient_edit.h"
// #include "scene/gui/graph_edit.h"
// #include "scene/gui/graph_node.h"
// #include "scene/gui/grid_container.h"
// #include "scene/gui/item_list.h"
// #include "scene/gui/label.h"
// #include "scene/gui/line_edit.h"
// #include "scene/gui/link_button.h"
// #include "scene/gui/margin_container.h"
// #include "scene/gui/menu_button.h"
// #include "scene/gui/nine_patch_rect.h"
// #include "scene/gui/option_button.h"
// #include "scene/gui/panel.h"
// #include "scene/gui/panel_container.h"
// #include "scene/gui/popup.h"
// #include "scene/gui/popup_menu.h"
// #include "scene/gui/progress_bar.h"
// #include "scene/gui/range.h"
// #include "scene/gui/reference_rect.h"
// #include "scene/gui/rich_text_effect.h"
// #include "scene/gui/rich_text_label.h"
// #include "scene/gui/scroll_bar.h"
// #include "scene/gui/scroll_container.h"
// #include "scene/gui/separator.h"
// // #include "scene/gui/shortcut.h"
// #include "scene/gui/slider.h"
// #include "scene/gui/spin_box.h"
// #include "scene/gui/split_container.h"
// #include "scene/gui/tab_container.h"
// // #include "scene/gui/tabs.h"
// #include "scene/gui/text_edit.h"
// #include "scene/gui/texture_button.h"
// #include "scene/gui/texture_progress_bar.h"
// #include "scene/gui/texture_rect.h"
// // #include "scene/gui/tool_button.h"
// #include "scene/gui/tree.h"
// #include "scene/gui/video_player.h"
// #include "scene/gui/subviewport_container.h"
// #include "scene/main/canvas_layer.h"
// #include "scene/main/http_request.h"
// #include "scene/main/instance_placeholder.h"
// #include "scene/main/node.h"
// #include "scene/main/resource_preloader.h"
// #include "scene/main/scene_tree.h"
// #include "scene/main/timer.h"
// #include "scene/main/viewport.h"
// #include "scene/resources/animation.h"
// #include "scene/resources/audio_stream_sample.h"
// #include "scene/resources/bit_map.h"
// #include "scene/resources/box_shape_3d.h"
// #include "scene/resources/capsule_shape_3d.h"
// #include "scene/resources/capsule_shape_2d.h"
// #include "scene/resources/circle_shape_2d.h"
// #include "scene/resources/concave_polygon_shape_3d.h"
// #include "scene/resources/concave_polygon_shape_2d.h"
// #include "scene/resources/convex_polygon_shape_3d.h"
// #include "scene/resources/convex_polygon_shape_2d.h"
// #include "scene/resources/curve.h"
// #include "scene/resources/cylinder_shape_3d.h"
// #include "scene/resources/default_theme/default_theme.h"
// #include "scene/resources/environment.h"
// #include "scene/resources/font.h"
// #include "scene/resources/gradient.h"
// #include "scene/resources/height_map_shape_3d.h"
// // #include "scene/resources/line_shape_2d.h"
// #include "scene/resources/material.h"
// #include "scene/resources/mesh.h"
// #include "scene/resources/mesh_data_tool.h"
// #include "scene/resources/mesh_library.h"
// #include "scene/resources/multimesh.h"
// #include "scene/resources/packed_scene.h"
// #include "scene/resources/particles_material.h"
// #include "scene/resources/physics_material.h"
// #include "scene/resources/polygon_path_finder.h"
// #include "scene/resources/primitive_meshes.h"
// // #include "scene/resources/ray_shape_2d.h"
// #include "scene/resources/rectangle_shape_2d.h"
// #include "scene/resources/resource_format_text.h"
// #include "scene/resources/segment_shape_2d.h"
// #include "scene/resources/shader.h"
// #include "scene/resources/shape_3d.h"
// #include "scene/resources/shape_2d.h"
// #include "scene/resources/skin.h"
// #include "scene/resources/sky.h"
// #include "scene/resources/sphere_shape_3d.h"
// #include "scene/resources/style_box.h"
// #include "scene/resources/surface_tool.h"
// #include "scene/resources/text_file.h"
// #include "scene/resources/texture.h"
// #include "scene/resources/theme.h"
// #include "scene/resources/tile_set.h"
// #include "scene/resources/video_stream.h"
// #include "scene/resources/visual_shader.h"
// #include "scene/resources/visual_shader_nodes.h"
// #include "scene/resources/world_3d.h"
// #include "scene/resources/world_2d.h"

// bool _wasgo_variant_check_id(WasGoState::WasGoID id){
//     return false;
// }
// WasGoState::_EncodedVariant _wasgo_variant_constructor(Variant::Type type, WasGoState::_EncodedVariant *p_args, const int p_argcount){

// }
inline void _wasgo_variant_deconstructor(wasm_exec_env_t exec_env, WasGoState::WasGoID id) {
}
// WasGoState::_EncodedVariant _wasgo_variant_call(WasGoState::WasGoID id, const std::string &p_method, WasGoState::_EncodedVariant *p_args, const int p_argcount){

// }
// WasGoState::_EncodedVariant _wasgo_variant_call_op(WasGoState::WasGoID id, const Variant::Operator p_op, WasGoState::_EncodedVariant *p_args, const int p_argcount){

// }
// WasGoState::_EncodedVariant _wasgo_variant_call_const(WasGoState::WasGoID id, const std::string &p_method, WasGoState::_EncodedVariant *p_args, const int p_argcount){

// }
// static WasGoState::_EncodedVariant _wasgo_variant_call_static(std::string type, const std::string &p_method, WasGoState::_EncodedVariant *p_args, const int p_argcount){

// }

// WasGoState::WasGoID _wasgo_Input_constructor() {
// 	return 0;
//     }
//     void _wasgo_Input_destructor(WasGoState::WasGoID p_wasgo_id){

//     }

// void _wasgo_Input_wrapper_action_press(wasm_exec_env_t exec_env, const uint8_t * p_action, int p_wasgo_buffer_size_action, float p_strength){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_action = String();
// 			decode_variant(v_action, p_action, p_wasgo_buffer_size_action);
// 			Variant * action = &v_action;
// 			Variant v_strength = (float) p_strength;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_action, & v_strength};
// 			Variant wasgo_ret = caller->call((String)"action_press", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: action_press");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Input_wrapper_action_release(wasm_exec_env_t exec_env, const uint8_t * p_action, int p_wasgo_buffer_size_action){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_action = String();
// 			decode_variant(v_action, p_action, p_wasgo_buffer_size_action);
// 			Variant * action = &v_action;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_action};
// 			Variant wasgo_ret = caller->call((String)"action_release", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: action_release");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Input_wrapper_add_joy_mapping(wasm_exec_env_t exec_env, const uint8_t * p_mapping, int p_wasgo_buffer_size_mapping, bool p_update_existing){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_mapping = String();
// 			decode_variant(v_mapping, p_mapping, p_wasgo_buffer_size_mapping);
// 			Variant * mapping = &v_mapping;
// 			Variant v_update_existing = (bool) p_update_existing;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_mapping, & v_update_existing};
// 			Variant wasgo_ret = caller->call((String)"add_joy_mapping", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: add_joy_mapping");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Input_wrapper_get_accelerometer(wasm_exec_env_t exec_env, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_local_ret = caller->call((String)"get_accelerometer", varargs, 0, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_accelerometer");}
// 		}
// 	}}

// float _wasgo_Input_wrapper_get_action_strength(wasm_exec_env_t exec_env, const uint8_t * p_action, int p_wasgo_buffer_size_action){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_action = String();
// 			decode_variant(v_action, p_action, p_wasgo_buffer_size_action);
// 			Variant * action = &v_action;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_action};
// 			float wasgo_ret = (float) caller->call((String)"get_action_strength", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_action_strength");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return float();
// }

// WasGoState::WasGoID _wasgo_Input_wrapper_get_connected_joypads(wasm_exec_env_t exec_env){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"get_connected_joypads", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_connected_joypads");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// WasGoState::WasGoID _wasgo_Input_wrapper_get_current_cursor_shape(wasm_exec_env_t exec_env){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"get_current_cursor_shape", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_current_cursor_shape");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// void _wasgo_Input_wrapper_get_gravity(wasm_exec_env_t exec_env, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_local_ret = caller->call((String)"get_gravity", varargs, 0, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_gravity");}
// 		}
// 	}}

// void _wasgo_Input_wrapper_get_gyroscope(wasm_exec_env_t exec_env, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_local_ret = caller->call((String)"get_gyroscope", varargs, 0, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_gyroscope");}
// 		}
// 	}}

// float _wasgo_Input_wrapper_get_joy_axis(wasm_exec_env_t exec_env, int p_device, int p_axis){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_device = (int) p_device;
// 			Variant v_axis = (int) p_axis;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_device, & v_axis};
// 			float wasgo_ret = (float) caller->call((String)"get_joy_axis", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_joy_axis");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return float();
// }

// int _wasgo_Input_wrapper_get_joy_axis_index_from_string(wasm_exec_env_t exec_env, const uint8_t * p_axis, int p_wasgo_buffer_size_axis){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_axis = String();
// 			decode_variant(v_axis, p_axis, p_wasgo_buffer_size_axis);
// 			Variant * axis = &v_axis;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_axis};
// 			int wasgo_ret = (int) caller->call((String)"get_joy_axis_index_from_string", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_joy_axis_index_from_string");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// void _wasgo_Input_wrapper_get_joy_axis_string(wasm_exec_env_t exec_env, uint8_t * wasgo_ret, int wasgo_ret_len, int p_axis_index){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_axis_index = (int) p_axis_index;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_axis_index};
// 			Variant wasgo_local_ret = caller->call((String)"get_joy_axis_string", varargs, 1, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_joy_axis_string");}
// 		}
// 	}}

// int _wasgo_Input_wrapper_get_joy_button_index_from_string(wasm_exec_env_t exec_env, const uint8_t * p_button, int p_wasgo_buffer_size_button){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_button = String();
// 			decode_variant(v_button, p_button, p_wasgo_buffer_size_button);
// 			Variant * button = &v_button;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_button};
// 			int wasgo_ret = (int) caller->call((String)"get_joy_button_index_from_string", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_joy_button_index_from_string");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// void _wasgo_Input_wrapper_get_joy_button_string(wasm_exec_env_t exec_env, uint8_t * wasgo_ret, int wasgo_ret_len, int p_button_index){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_button_index = (int) p_button_index;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_button_index};
// 			Variant wasgo_local_ret = caller->call((String)"get_joy_button_string", varargs, 1, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_joy_button_string");}
// 		}
// 	}}

// void _wasgo_Input_wrapper_get_joy_guid(wasm_exec_env_t exec_env, uint8_t * wasgo_ret, int wasgo_ret_len, int p_device){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_device = (int) p_device;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_device};
// 			Variant wasgo_local_ret = caller->call((String)"get_joy_guid", varargs, 1, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_joy_guid");}
// 		}
// 	}}

// void _wasgo_Input_wrapper_get_joy_name(wasm_exec_env_t exec_env, uint8_t * wasgo_ret, int wasgo_ret_len, int p_device){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_device = (int) p_device;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_device};
// 			Variant wasgo_local_ret = caller->call((String)"get_joy_name", varargs, 1, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_joy_name");}
// 		}
// 	}}

// float _wasgo_Input_wrapper_get_joy_vibration_duration(wasm_exec_env_t exec_env, int p_device){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_device = (int) p_device;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_device};
// 			float wasgo_ret = (float) caller->call((String)"get_joy_vibration_duration", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_joy_vibration_duration");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return float();
// }

// void _wasgo_Input_wrapper_get_joy_vibration_strength(wasm_exec_env_t exec_env, uint8_t * wasgo_ret, int wasgo_ret_len, int p_device){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_device = (int) p_device;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_device};
// 			Variant wasgo_local_ret = caller->call((String)"get_joy_vibration_strength", varargs, 1, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_joy_vibration_strength");}
// 		}
// 	}}

// void _wasgo_Input_wrapper_get_last_mouse_speed(wasm_exec_env_t exec_env, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			// Variant wasgo_local_ret = caller->call((String)"get_last_mouse_speed", varargs, 0, error);
// 			Vector2 mouse_speed = caller->get_last_mouse_speed();
// 			Variant wasgo_local_ret = mouse_speed;
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			// if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_last_mouse_speed");}
// 		}
// 	}}

// void _wasgo_Input_wrapper_get_magnetometer(wasm_exec_env_t exec_env, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_local_ret = caller->call((String)"get_magnetometer", varargs, 0, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_magnetometer");}
// 		}
// 	}}

// int _wasgo_Input_wrapper_get_mouse_button_mask(wasm_exec_env_t exec_env){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"get_mouse_button_mask", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_mouse_button_mask");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// WasGoState::WasGoID _wasgo_Input_wrapper_get_mouse_mode(wasm_exec_env_t exec_env){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"get_mouse_mode", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_mouse_mode");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// int _wasgo_Input_wrapper_is_action_just_pressed(wasm_exec_env_t exec_env, const uint8_t * p_action, int p_wasgo_buffer_size_action){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_action = String();
// 			decode_variant(v_action, p_action, p_wasgo_buffer_size_action);
// 			Variant * action = &v_action;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_action};
// 			int wasgo_ret = (int) caller->call((String)"is_action_just_pressed", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_action_just_pressed");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Input_wrapper_is_action_just_released(wasm_exec_env_t exec_env, const uint8_t * p_action, int p_wasgo_buffer_size_action){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_action = String();
// 			decode_variant(v_action, p_action, p_wasgo_buffer_size_action);
// 			Variant * action = &v_action;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_action};
// 			int wasgo_ret = (int) caller->call((String)"is_action_just_released", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_action_just_released");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Input_wrapper_is_action_pressed(wasm_exec_env_t exec_env, const uint8_t * p_action, int p_wasgo_buffer_size_action){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_action = String();
// 			decode_variant(v_action, p_action, p_wasgo_buffer_size_action);
// 			Variant * action = &v_action;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_action};
// 			int wasgo_ret = (int) caller->call((String)"is_action_pressed", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_action_pressed");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Input_wrapper_is_joy_button_pressed(wasm_exec_env_t exec_env, int p_device, int p_button){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_device = (int) p_device;
// 			Variant v_button = (int) p_button;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_device, & v_button};
// 			int wasgo_ret = (int) caller->call((String)"is_joy_button_pressed", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_joy_button_pressed");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Input_wrapper_is_joy_known(wasm_exec_env_t exec_env, int p_device){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_device = (int) p_device;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_device};
// 			int wasgo_ret = (int) caller->call((String)"is_joy_known", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_joy_known");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Input_wrapper_is_key_pressed(wasm_exec_env_t exec_env, int p_scancode){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_scancode = (int) p_scancode;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_scancode};
// 			int wasgo_ret = (int) caller->call((String)"is_key_pressed", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_key_pressed");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Input_wrapper_is_mouse_button_pressed(wasm_exec_env_t exec_env, int p_button){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_button = (int) p_button;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_button};
// 			int wasgo_ret = (int) caller->call((String)"is_mouse_button_pressed", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_mouse_button_pressed");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// void _wasgo_Input_wrapper_joy_connection_changed(wasm_exec_env_t exec_env, int p_device, bool p_connected, const uint8_t * p_name, int p_wasgo_buffer_size_name, const uint8_t * p_guid, int p_wasgo_buffer_size_guid){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_device = (int) p_device;
// 			Variant v_connected = (bool) p_connected;
// 			Variant v_name = String();
// 			decode_variant(v_name, p_name, p_wasgo_buffer_size_name);
// 			Variant * name = &v_name;
// 			Variant v_guid = String();
// 			decode_variant(v_guid, p_guid, p_wasgo_buffer_size_guid);
// 			Variant * guid = &v_guid;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_device, & v_connected, & v_name, & v_guid};
// 			Variant wasgo_ret = caller->call((String)"joy_connection_changed", varargs, 4, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: joy_connection_changed");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Input_wrapper_parse_input_event(wasm_exec_env_t exec_env, WasGoState::WasGoID p_event){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_event = state->lookup_object(p_event);
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_event};
// 			Variant wasgo_ret = caller->call((String)"parse_input_event", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: parse_input_event");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Input_wrapper_remove_joy_mapping(wasm_exec_env_t exec_env, const uint8_t * p_guid, int p_wasgo_buffer_size_guid){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_guid = String();
// 			decode_variant(v_guid, p_guid, p_wasgo_buffer_size_guid);
// 			Variant * guid = &v_guid;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_guid};
// 			Variant wasgo_ret = caller->call((String)"remove_joy_mapping", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: remove_joy_mapping");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Input_wrapper_set_custom_mouse_cursor(wasm_exec_env_t exec_env, WasGoState::WasGoID p_image, int p_shape, const uint8_t * p_hotspot, int p_wasgo_buffer_size_hotspot){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_image = state->lookup_object(p_image);
// 			Variant v_shape = state->lookup_object(p_shape);
// 			Variant v_hotspot = Vector2();
// 			decode_variant(v_hotspot, p_hotspot, p_wasgo_buffer_size_hotspot);
// 			Variant * hotspot = &v_hotspot;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_image, & v_shape, & v_hotspot};
// 			Variant wasgo_ret = caller->call((String)"set_custom_mouse_cursor", varargs, 3, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_custom_mouse_cursor");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Input_wrapper_set_default_cursor_shape(wasm_exec_env_t exec_env, int p_shape){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_shape = state->lookup_object(p_shape);
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_shape};
// 			Variant wasgo_ret = caller->call((String)"set_default_cursor_shape", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_default_cursor_shape");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Input_wrapper_set_mouse_mode(wasm_exec_env_t exec_env, int p_mode){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			// Variant v_mode = state->lookup_object(p_mode);
// 			// Callable::CallError error;
// 			// const Variant* varargs[] = {& v_mode};
// 			// Variant wasgo_ret = caller->call((String)"set_mouse_mode", varargs, 1, error);

// 			// if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_mouse_mode");}

// 			caller->set_mouse_mode((Input::MouseMode) p_mode);
// 		}
// 	}
// 	return;
// }

// void _wasgo_Input_wrapper_set_use_accumulated_input(wasm_exec_env_t exec_env, bool p_enable){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_enable = (bool) p_enable;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enable};
// 			Variant wasgo_ret = caller->call((String)"set_use_accumulated_input", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_use_accumulated_input");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Input_wrapper_start_joy_vibration(wasm_exec_env_t exec_env, int p_device, float p_weak_magnitude, float p_strong_magnitude, float p_duration){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_device = (int) p_device;
// 			Variant v_weak_magnitude = (float) p_weak_magnitude;
// 			Variant v_strong_magnitude = (float) p_strong_magnitude;
// 			Variant v_duration = (float) p_duration;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_device, & v_weak_magnitude, & v_strong_magnitude, & v_duration};
// 			Variant wasgo_ret = caller->call((String)"start_joy_vibration", varargs, 4, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: start_joy_vibration");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Input_wrapper_stop_joy_vibration(wasm_exec_env_t exec_env, int p_device){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_device = (int) p_device;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_device};
// 			Variant wasgo_ret = caller->call((String)"stop_joy_vibration", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: stop_joy_vibration");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Input_wrapper_vibrate_handheld(wasm_exec_env_t exec_env, int p_duration_ms){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_duration_ms = (int) p_duration_ms;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_duration_ms};
// 			Variant wasgo_ret = caller->call((String)"vibrate_handheld", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: vibrate_handheld");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Input_wrapper_warp_mouse_position(wasm_exec_env_t exec_env, const uint8_t * p_to, int p_wasgo_buffer_size_to){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Input *caller = Input::get_singleton();
// 		if(caller){
// 			Variant v_to = Vector2();
// 			decode_variant(v_to, p_to, p_wasgo_buffer_size_to);
// 			Variant * to = &v_to;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_to};
// 			Variant wasgo_ret = caller->call((String)"warp_mouse_position", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: warp_mouse_position");}
// 		}
// 	}
// 	return;
// }

//     WasGoState::WasGoID _wasgo_InputEvent_constructor(){
//         return 0;
//     }
//     void _wasgo_InputEvent_destructor(WasGoState::WasGoID p_wasgo_id){

//     }

// int _wasgo_InputEvent_wrapper_accumulate(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, WasGoState::WasGoID p_with_event){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEvent *caller = (InputEvent *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_with_event = state->lookup_object(p_with_event);
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_with_event};
// 			int wasgo_ret = (int) caller->call((String)"accumulate", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: accumulate");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// void _wasgo_InputEvent_wrapper_as_text(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEvent *caller = (InputEvent *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_local_ret = caller->call((String)"as_text", varargs, 0, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: as_text");}
// 		}
// 	}}

// float _wasgo_InputEvent_wrapper_get_action_strength(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_action, int p_wasgo_buffer_size_action){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEvent *caller = (InputEvent *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_action = String();
// 			decode_variant(v_action, p_action, p_wasgo_buffer_size_action);
// 			Variant * action = &v_action;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_action};
// 			float wasgo_ret = (float) caller->call((String)"get_action_strength", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_action_strength");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return float();
// }

// int _wasgo_InputEvent_wrapper_get_device(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEvent *caller = (InputEvent *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"get_device", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_device");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_InputEvent_wrapper_is_action(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_action, int p_wasgo_buffer_size_action){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEvent *caller = (InputEvent *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_action = String();
// 			decode_variant(v_action, p_action, p_wasgo_buffer_size_action);
// 			Variant * action = &v_action;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_action};
// 			int wasgo_ret = (int) caller->call((String)"is_action", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_action");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_InputEvent_wrapper_is_action_pressed(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_action, int p_wasgo_buffer_size_action, bool p_allow_echo){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEvent *caller = (InputEvent *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_action = String();
// 			decode_variant(v_action, p_action, p_wasgo_buffer_size_action);
// 			Variant * action = &v_action;
// 			Variant v_allow_echo = (bool) p_allow_echo;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_action, & v_allow_echo};
// 			int wasgo_ret = (int) caller->call((String)"is_action_pressed", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_action_pressed");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_InputEvent_wrapper_is_action_released(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_action, int p_wasgo_buffer_size_action){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEvent *caller = (InputEvent *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_action = String();
// 			decode_variant(v_action, p_action, p_wasgo_buffer_size_action);
// 			Variant * action = &v_action;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_action};
// 			int wasgo_ret = (int) caller->call((String)"is_action_released", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_action_released");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_InputEvent_wrapper_is_action_type(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEvent *caller = (InputEvent *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_action_type", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_action_type");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_InputEvent_wrapper_is_echo(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEvent *caller = (InputEvent *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_echo", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_echo");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_InputEvent_wrapper_is_pressed(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEvent *caller = (InputEvent *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_pressed", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_pressed");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// void _wasgo_InputEvent_wrapper_set_device(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, int p_device){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEvent *caller = (InputEvent *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_device = (int) p_device;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_device};
// 			Variant wasgo_ret = caller->call((String)"set_device", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_device");}
// 		}
// 	}
// 	return;
// }

// int _wasgo_InputEvent_wrapper_shortcut_match(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, WasGoState::WasGoID p_event){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEvent *caller = (InputEvent *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_event = state->lookup_object(p_event);
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_event};
// 			int wasgo_ret = (int) caller->call((String)"shortcut_match", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: shortcut_match");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// WasGoState::WasGoID _wasgo_InputEvent_wrapper_xformed_by(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_xform, int p_wasgo_buffer_size_xform, const uint8_t * p_local_ofs, int p_wasgo_buffer_size_local_ofs){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEvent *caller = (InputEvent *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_xform = Transform2D();
// 			decode_variant(v_xform, p_xform, p_wasgo_buffer_size_xform);
// 			Variant * xform = &v_xform;
// 			Variant v_local_ofs = Vector2();
// 			decode_variant(v_local_ofs, p_local_ofs, p_wasgo_buffer_size_local_ofs);
// 			Variant * local_ofs = &v_local_ofs;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_xform, & v_local_ofs};
// 			Variant wasgo_ret = caller->call((String)"xformed_by", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: xformed_by");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

//     WasGoState::WasGoID _wasgo_InputEventAction_constructor(){
//         return 0;
//     }
//     void _wasgo_InputEventAction_destructor(WasGoState::WasGoID p_wasgo_id){

//     }

// void _wasgo_InputEventAction_wrapper_get_action(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventAction *caller = (InputEventAction *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_local_ret = caller->call((String)"get_action", varargs, 0, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_action");}
// 		}
// 	}}

// float _wasgo_InputEventAction_wrapper_get_strength(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventAction *caller = (InputEventAction *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			float wasgo_ret = (float) caller->call((String)"get_strength", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_strength");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return float();
// }

// void _wasgo_InputEventAction_wrapper_set_action(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_action, int p_wasgo_buffer_size_action){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventAction *caller = (InputEventAction *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_action = String();
// 			decode_variant(v_action, p_action, p_wasgo_buffer_size_action);
// 			Variant * action = &v_action;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_action};
// 			Variant wasgo_ret = caller->call((String)"set_action", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_action");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_InputEventAction_wrapper_set_pressed(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_pressed){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventAction *caller = (InputEventAction *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_pressed = (bool) p_pressed;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_pressed};
// 			Variant wasgo_ret = caller->call((String)"set_pressed", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_pressed");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_InputEventAction_wrapper_set_strength(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, float p_strength){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventAction *caller = (InputEventAction *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_strength = (float) p_strength;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_strength};
// 			Variant wasgo_ret = caller->call((String)"set_strength", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_strength");}
// 		}
// 	}
// 	return;
// }

//     WasGoState::WasGoID _wasgo_InputEventKey_constructor(){
//         return 0;
//     }
//     void _wasgo_InputEventKey_destructor(WasGoState::WasGoID p_wasgo_id){

//     }

// int _wasgo_InputEventKey_wrapper_get_scancode(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventKey *caller = (InputEventKey *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"get_scancode", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_scancode");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_InputEventKey_wrapper_get_scancode_with_modifiers(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventKey *caller = (InputEventKey *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"get_scancode_with_modifiers", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_scancode_with_modifiers");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_InputEventKey_wrapper_get_unicode(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventKey *caller = (InputEventKey *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"get_unicode", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_unicode");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// void _wasgo_InputEventKey_wrapper_set_echo(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_echo){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventKey *caller = (InputEventKey *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_echo = (bool) p_echo;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_echo};
// 			Variant wasgo_ret = caller->call((String)"set_echo", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_echo");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_InputEventKey_wrapper_set_pressed(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_pressed){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventKey *caller = (InputEventKey *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_pressed = (bool) p_pressed;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_pressed};
// 			Variant wasgo_ret = caller->call((String)"set_pressed", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_pressed");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_InputEventKey_wrapper_set_scancode(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, int p_scancode){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventKey *caller = (InputEventKey *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_scancode = (int) p_scancode;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_scancode};
// 			Variant wasgo_ret = caller->call((String)"set_scancode", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_scancode");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_InputEventKey_wrapper_set_unicode(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, int p_unicode){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventKey *caller = (InputEventKey *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_unicode = (int) p_unicode;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_unicode};
// 			Variant wasgo_ret = caller->call((String)"set_unicode", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_unicode");}
// 		}
// 	}
// 	return;
// }

//     WasGoState::WasGoID _wasgo_InputEventMouse_constructor(){
//         return 0;
//     }
//     void _wasgo_InputEventMouse_destructor(WasGoState::WasGoID p_wasgo_id){

//     }

// int _wasgo_InputEventMouse_wrapper_get_button_mask(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventMouse *caller = (InputEventMouse *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"get_button_mask", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_button_mask");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// void _wasgo_InputEventMouse_wrapper_get_global_position(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventMouse *caller = (InputEventMouse *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_local_ret = caller->call((String)"get_global_position", varargs, 0, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_global_position");}
// 		}
// 	}}

// void _wasgo_InputEventMouse_wrapper_get_position(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventMouse *caller = (InputEventMouse *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_local_ret = caller->call((String)"get_position", varargs, 0, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_position");}
// 		}
// 	}}

// void _wasgo_InputEventMouse_wrapper_set_button_mask(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, int p_button_mask){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventMouse *caller = (InputEventMouse *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_button_mask = (int) p_button_mask;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_button_mask};
// 			Variant wasgo_ret = caller->call((String)"set_button_mask", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_button_mask");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_InputEventMouse_wrapper_set_global_position(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_global_position, int p_wasgo_buffer_size_global_position){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventMouse *caller = (InputEventMouse *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_global_position = Vector2();
// 			decode_variant(v_global_position, p_global_position, p_wasgo_buffer_size_global_position);
// 			Variant * global_position = &v_global_position;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_global_position};
// 			Variant wasgo_ret = caller->call((String)"set_global_position", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_global_position");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_InputEventMouse_wrapper_set_position(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_position, int p_wasgo_buffer_size_position){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventMouse *caller = (InputEventMouse *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_position = Vector2();
// 			decode_variant(v_position, p_position, p_wasgo_buffer_size_position);
// 			Variant * position = &v_position;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_position};
// 			Variant wasgo_ret = caller->call((String)"set_position", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_position");}
// 		}
// 	}
// 	return;
// }

//     WasGoState::WasGoID _wasgo_InputEventMouseButton_constructor(){
//         return 0;
//     }
//     void _wasgo_InputEventMouseButton_destructor(WasGoState::WasGoID p_wasgo_id){

//     }

// int _wasgo_InputEventMouseButton_wrapper_get_button_index(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventMouseButton *caller = (InputEventMouseButton *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"get_button_index", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_button_index");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// float _wasgo_InputEventMouseButton_wrapper_get_factor(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventMouseButton *caller = (InputEventMouseButton *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			float wasgo_ret = (float) caller->call((String)"get_factor", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_factor");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return float();
// }

// int _wasgo_InputEventMouseButton_wrapper_is_doubleclick(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventMouseButton *caller = (InputEventMouseButton *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_doubleclick", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_doubleclick");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// void _wasgo_InputEventMouseButton_wrapper_set_button_index(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, int p_button_index){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventMouseButton *caller = (InputEventMouseButton *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_button_index = (int) p_button_index;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_button_index};
// 			Variant wasgo_ret = caller->call((String)"set_button_index", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_button_index");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_InputEventMouseButton_wrapper_set_doubleclick(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_doubleclick){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventMouseButton *caller = (InputEventMouseButton *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_doubleclick = (bool) p_doubleclick;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_doubleclick};
// 			Variant wasgo_ret = caller->call((String)"set_doubleclick", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_doubleclick");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_InputEventMouseButton_wrapper_set_factor(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, float p_factor){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventMouseButton *caller = (InputEventMouseButton *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_factor = (float) p_factor;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_factor};
// 			Variant wasgo_ret = caller->call((String)"set_factor", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_factor");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_InputEventMouseButton_wrapper_set_pressed(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_pressed){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventMouseButton *caller = (InputEventMouseButton *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_pressed = (bool) p_pressed;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_pressed};
// 			Variant wasgo_ret = caller->call((String)"set_pressed", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_pressed");}
// 		}
// 	}
// 	return;
// }

//     WasGoState::WasGoID _wasgo_InputEventMouseMotion_constructor(){
//         return 0;
//     }
//     void _wasgo_InputEventMouseMotion_destructor(WasGoState::WasGoID p_wasgo_id){

//     }

// float _wasgo_InputEventMouseMotion_wrapper_get_pressure(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventMouseMotion *caller = (InputEventMouseMotion *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			float wasgo_ret = (float) caller->call((String)"get_pressure", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_pressure");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return float();
// }

// void _wasgo_InputEventMouseMotion_wrapper_get_relative(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventMouseMotion *caller = (InputEventMouseMotion *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_local_ret = caller->call((String)"get_relative", varargs, 0, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			// if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_relative");}
// 		}
// 	}}

// void _wasgo_InputEventMouseMotion_wrapper_get_speed(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventMouseMotion *caller = (InputEventMouseMotion *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_local_ret = caller->call((String)"get_speed", varargs, 0, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_speed");}
// 		}
// 	}}

// void _wasgo_InputEventMouseMotion_wrapper_get_tilt(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventMouseMotion *caller = (InputEventMouseMotion *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_local_ret = caller->call((String)"get_tilt", varargs, 0, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_tilt");}
// 		}
// 	}}

// void _wasgo_InputEventMouseMotion_wrapper_set_pressure(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, float p_pressure){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventMouseMotion *caller = (InputEventMouseMotion *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_pressure = (float) p_pressure;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_pressure};
// 			Variant wasgo_ret = caller->call((String)"set_pressure", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_pressure");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_InputEventMouseMotion_wrapper_set_relative(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_relative, int p_wasgo_buffer_size_relative){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventMouseMotion *caller = (InputEventMouseMotion *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_relative = Vector2();
// 			decode_variant(v_relative, p_relative, p_wasgo_buffer_size_relative);
// 			Variant * relative = &v_relative;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_relative};
// 			Variant wasgo_ret = caller->call((String)"set_relative", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_relative");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_InputEventMouseMotion_wrapper_set_speed(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_speed, int p_wasgo_buffer_size_speed){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventMouseMotion *caller = (InputEventMouseMotion *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_speed = Vector2();
// 			decode_variant(v_speed, p_speed, p_wasgo_buffer_size_speed);
// 			Variant * speed = &v_speed;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_speed};
// 			Variant wasgo_ret = caller->call((String)"set_speed", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_speed");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_InputEventMouseMotion_wrapper_set_tilt(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_tilt, int p_wasgo_buffer_size_tilt){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventMouseMotion *caller = (InputEventMouseMotion *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_tilt = Vector2();
// 			decode_variant(v_tilt, p_tilt, p_wasgo_buffer_size_tilt);
// 			Variant * tilt = &v_tilt;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_tilt};
// 			Variant wasgo_ret = caller->call((String)"set_tilt", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_tilt");}
// 		}
// 	}
// 	return;
// }

//     WasGoState::WasGoID _wasgo_InputEventWithModifiers_constructor(){
//         return 0;
//     }
//     void _wasgo_InputEventWithModifiers_destructor(WasGoState::WasGoID p_wasgo_id){

//     }

// int _wasgo_InputEventWithModifiers_wrapper_get_alt(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventWithModifiers *caller = (InputEventWithModifiers *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"get_alt", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_alt");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_InputEventWithModifiers_wrapper_get_command(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventWithModifiers *caller = (InputEventWithModifiers *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"get_command", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_command");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_InputEventWithModifiers_wrapper_get_control(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventWithModifiers *caller = (InputEventWithModifiers *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"get_control", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_control");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_InputEventWithModifiers_wrapper_get_metakey(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventWithModifiers *caller = (InputEventWithModifiers *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"get_metakey", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_metakey");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_InputEventWithModifiers_wrapper_get_shift(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventWithModifiers *caller = (InputEventWithModifiers *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"get_shift", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_shift");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// void _wasgo_InputEventWithModifiers_wrapper_set_alt(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enable){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventWithModifiers *caller = (InputEventWithModifiers *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enable = (bool) p_enable;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enable};
// 			Variant wasgo_ret = caller->call((String)"set_alt", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_alt");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_InputEventWithModifiers_wrapper_set_command(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enable){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventWithModifiers *caller = (InputEventWithModifiers *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enable = (bool) p_enable;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enable};
// 			Variant wasgo_ret = caller->call((String)"set_command", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_command");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_InputEventWithModifiers_wrapper_set_control(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enable){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventWithModifiers *caller = (InputEventWithModifiers *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enable = (bool) p_enable;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enable};
// 			Variant wasgo_ret = caller->call((String)"set_control", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_control");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_InputEventWithModifiers_wrapper_set_metakey(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enable){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventWithModifiers *caller = (InputEventWithModifiers *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enable = (bool) p_enable;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enable};
// 			Variant wasgo_ret = caller->call((String)"set_metakey", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_metakey");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_InputEventWithModifiers_wrapper_set_shift(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enable){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		InputEventWithModifiers *caller = (InputEventWithModifiers *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enable = (bool) p_enable;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enable};
// 			Variant wasgo_ret = caller->call((String)"set_shift", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_shift");}
// 		}
// 	}
// 	return;
// }

//     WasGoState::WasGoID _wasgo_Node_constructor(){
//         return 0;
//     }
//     void _wasgo_Node_destructor(WasGoState::WasGoID p_wasgo_id){

//     }

// void _wasgo_Node_wrapper_add_child(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, WasGoState::WasGoID p_node, bool p_legible_unique_name){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_node = state->lookup_object(p_node);
// 			Variant v_legible_unique_name = (bool) p_legible_unique_name;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_node, & v_legible_unique_name};
// 			Variant wasgo_ret = caller->call((String)"add_child", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: add_child");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_add_child_below_node(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, WasGoState::WasGoID p_node, WasGoState::WasGoID p_child_node, bool p_legible_unique_name){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_node = state->lookup_object(p_node);
// 			Variant v_child_node = state->lookup_object(p_child_node);
// 			Variant v_legible_unique_name = (bool) p_legible_unique_name;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_node, & v_child_node, & v_legible_unique_name};
// 			Variant wasgo_ret = caller->call((String)"add_child_below_node", varargs, 3, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: add_child_below_node");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_add_to_group(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_group, int p_wasgo_buffer_size_group, bool p_persistent){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_group = String();
// 			decode_variant(v_group, p_group, p_wasgo_buffer_size_group);
// 			Variant * group = &v_group;
// 			Variant v_persistent = (bool) p_persistent;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_group, & v_persistent};
// 			Variant wasgo_ret = caller->call((String)"add_to_group", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: add_to_group");}
// 		}
// 	}
// 	return;
// }

// int _wasgo_Node_wrapper_can_process(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"can_process", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: can_process");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// WasGoState::WasGoID _wasgo_Node_wrapper_duplicate(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, int p_flags){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_flags = (int) p_flags;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_flags};
// 			Variant wasgo_ret = caller->call((String)"duplicate", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: duplicate");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// WasGoState::WasGoID _wasgo_Node_wrapper_find_node(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_mask, int p_wasgo_buffer_size_mask, bool p_recursive, bool p_owned){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_mask = String();
// 			decode_variant(v_mask, p_mask, p_wasgo_buffer_size_mask);
// 			Variant * mask = &v_mask;
// 			Variant v_recursive = (bool) p_recursive;
// 			Variant v_owned = (bool) p_owned;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_mask, & v_recursive, & v_owned};
// 			Variant wasgo_ret = caller->call((String)"find_node", varargs, 3, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: find_node");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// WasGoState::WasGoID _wasgo_Node_wrapper_find_parent(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_mask, int p_wasgo_buffer_size_mask){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_mask = String();
// 			decode_variant(v_mask, p_mask, p_wasgo_buffer_size_mask);
// 			Variant * mask = &v_mask;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_mask};
// 			Variant wasgo_ret = caller->call((String)"find_parent", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: find_parent");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// WasGoState::WasGoID _wasgo_Node_wrapper_get_child(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, int p_idx){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_idx = (int) p_idx;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_idx};
// 			Variant wasgo_ret = caller->call((String)"get_child", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_child");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// int _wasgo_Node_wrapper_get_child_count(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"get_child_count", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_child_count");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// WasGoState::WasGoID _wasgo_Node_wrapper_get_children(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"get_children", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_children");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// WasGoState::WasGoID _wasgo_Node_wrapper_get_custom_multiplayer(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"get_custom_multiplayer", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_custom_multiplayer");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// void _wasgo_Node_wrapper_get_filename(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_local_ret = caller->call((String)"get_filename", varargs, 0, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_filename");}
// 		}
// 	}}

// WasGoState::WasGoID _wasgo_Node_wrapper_get_groups(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"get_groups", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_groups");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// int _wasgo_Node_wrapper_get_index(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"get_index", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_index");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// WasGoState::WasGoID _wasgo_Node_wrapper_get_multiplayer(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"get_multiplayer", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_multiplayer");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// void _wasgo_Node_wrapper_get_name(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_local_ret = caller->call((String)"get_name", varargs, 0, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_name");}
// 		}
// 	}}

// int _wasgo_Node_wrapper_get_network_master(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"get_network_master", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_network_master");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// WasGoState::WasGoID _wasgo_Node_wrapper_get_node(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_path, int p_wasgo_buffer_size_path){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_path = NodePath();
// 			decode_variant(v_path, p_path, p_wasgo_buffer_size_path);
// 			Variant * path = &v_path;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_path};
// 			Variant wasgo_ret = caller->call((String)"get_node", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_node");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// WasGoState::WasGoID _wasgo_Node_wrapper_get_node_and_resource(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_path, int p_wasgo_buffer_size_path){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_path = NodePath();
// 			decode_variant(v_path, p_path, p_wasgo_buffer_size_path);
// 			Variant * path = &v_path;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_path};
// 			Variant wasgo_ret = caller->call((String)"get_node_and_resource", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_node_and_resource");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// WasGoState::WasGoID _wasgo_Node_wrapper_get_node_or_null(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_path, int p_wasgo_buffer_size_path){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_path = NodePath();
// 			decode_variant(v_path, p_path, p_wasgo_buffer_size_path);
// 			Variant * path = &v_path;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_path};
// 			Variant wasgo_ret = caller->call((String)"get_node_or_null", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_node_or_null");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// WasGoState::WasGoID _wasgo_Node_wrapper_get_owner(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"get_owner", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_owner");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// WasGoState::WasGoID _wasgo_Node_wrapper_get_parent(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"get_parent", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_parent");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// void _wasgo_Node_wrapper_get_path(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_local_ret = caller->call((String)"get_path", varargs, 0, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_path");}
// 		}
// 	}}

// void _wasgo_Node_wrapper_get_path_to(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len, WasGoState::WasGoID p_node){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_node = state->lookup_object(p_node);
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_node};
// 			Variant wasgo_local_ret = caller->call((String)"get_path_to", varargs, 1, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_path_to");}
// 		}
// 	}}

// WasGoState::WasGoID _wasgo_Node_wrapper_get_pause_mode(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"get_pause_mode", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_pause_mode");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// float _wasgo_Node_wrapper_get_physics_process_delta_time(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			float wasgo_ret = (float) caller->call((String)"get_physics_process_delta_time", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_physics_process_delta_time");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return float();
// }

// int _wasgo_Node_wrapper_get_position_in_parent(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"get_position_in_parent", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_position_in_parent");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// float _wasgo_Node_wrapper_get_process_delta_time(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			float wasgo_ret = (float) caller->call((String)"get_process_delta_time", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_process_delta_time");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return float();
// }

// int _wasgo_Node_wrapper_get_process_priority(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"get_process_priority", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_process_priority");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Node_wrapper_get_scene_instance_load_placeholder(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"get_scene_instance_load_placeholder", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_scene_instance_load_placeholder");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// WasGoState::WasGoID _wasgo_Node_wrapper_get_tree(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"get_tree", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_tree");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// WasGoState::WasGoID _wasgo_Node_wrapper_get_viewport(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"get_viewport", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_viewport");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// int _wasgo_Node_wrapper_has_node(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_path, int p_wasgo_buffer_size_path){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_path = NodePath();
// 			decode_variant(v_path, p_path, p_wasgo_buffer_size_path);
// 			Variant * path = &v_path;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_path};
// 			int wasgo_ret = (int) caller->call((String)"has_node", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: has_node");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Node_wrapper_has_node_and_resource(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_path, int p_wasgo_buffer_size_path){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_path = NodePath();
// 			decode_variant(v_path, p_path, p_wasgo_buffer_size_path);
// 			Variant * path = &v_path;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_path};
// 			int wasgo_ret = (int) caller->call((String)"has_node_and_resource", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: has_node_and_resource");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Node_wrapper_is_a_parent_of(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, WasGoState::WasGoID p_node){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_node = state->lookup_object(p_node);
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_node};
// 			int wasgo_ret = (int) caller->call((String)"is_a_parent_of", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_a_parent_of");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Node_wrapper_is_displayed_folded(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_displayed_folded", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_displayed_folded");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Node_wrapper_is_greater_than(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, WasGoState::WasGoID p_node){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_node = state->lookup_object(p_node);
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_node};
// 			int wasgo_ret = (int) caller->call((String)"is_greater_than", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_greater_than");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Node_wrapper_is_in_group(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_group, int p_wasgo_buffer_size_group){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_group = String();
// 			decode_variant(v_group, p_group, p_wasgo_buffer_size_group);
// 			Variant * group = &v_group;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_group};
// 			int wasgo_ret = (int) caller->call((String)"is_in_group", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_in_group");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Node_wrapper_is_inside_tree(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_inside_tree", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_inside_tree");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Node_wrapper_is_network_master(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_network_master", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_network_master");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Node_wrapper_is_physics_processing(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_physics_processing", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_physics_processing");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Node_wrapper_is_physics_processing_internal(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_physics_processing_internal", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_physics_processing_internal");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Node_wrapper_is_processing(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_processing", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_processing");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Node_wrapper_is_processing_input(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_processing_input", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_processing_input");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Node_wrapper_is_processing_internal(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_processing_internal", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_processing_internal");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Node_wrapper_is_processing_unhandled_input(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_processing_unhandled_input", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_processing_unhandled_input");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Node_wrapper_is_processing_unhandled_key_input(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_processing_unhandled_key_input", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_processing_unhandled_key_input");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// void _wasgo_Node_wrapper_move_child(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, WasGoState::WasGoID p_child_node, int p_to_position){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_child_node = state->lookup_object(p_child_node);
// 			Variant v_to_position = (int) p_to_position;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_child_node, & v_to_position};
// 			Variant wasgo_ret = caller->call((String)"move_child", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: move_child");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_print_stray_nodes(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"print_stray_nodes", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: print_stray_nodes");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_print_tree(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"print_tree", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: print_tree");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_print_tree_pretty(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"print_tree_pretty", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: print_tree_pretty");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_propagate_call(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_method, int p_wasgo_buffer_size_method, WasGoState::WasGoID p_args, bool p_parent_first){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_method = String();
// 			decode_variant(v_method, p_method, p_wasgo_buffer_size_method);
// 			Variant * method = &v_method;
// 			Variant v_args = state->lookup_object(p_args);
// 			Variant v_parent_first = (bool) p_parent_first;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_method, & v_args, & v_parent_first};
// 			Variant wasgo_ret = caller->call((String)"propagate_call", varargs, 3, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: propagate_call");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_propagate_notification(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, int p_what){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_what = (int) p_what;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_what};
// 			Variant wasgo_ret = caller->call((String)"propagate_notification", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: propagate_notification");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_queue_free(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"queue_free", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: queue_free");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_raise(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"raise", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: raise");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_remove_and_skip(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"remove_and_skip", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: remove_and_skip");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_remove_child(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, WasGoState::WasGoID p_node){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_node = state->lookup_object(p_node);
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_node};
// 			Variant wasgo_ret = caller->call((String)"remove_child", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: remove_child");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_remove_from_group(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_group, int p_wasgo_buffer_size_group){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_group = String();
// 			decode_variant(v_group, p_group, p_wasgo_buffer_size_group);
// 			Variant * group = &v_group;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_group};
// 			Variant wasgo_ret = caller->call((String)"remove_from_group", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: remove_from_group");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_replace_by(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, WasGoState::WasGoID p_node, bool p_keep_data){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_node = state->lookup_object(p_node);
// 			Variant v_keep_data = (bool) p_keep_data;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_node, & v_keep_data};
// 			Variant wasgo_ret = caller->call((String)"replace_by", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: replace_by");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_request_ready(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"request_ready", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: request_ready");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_rpc(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len, const uint8_t * p_method, int p_wasgo_buffer_size_method){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_method = String();
// 			decode_variant(v_method, p_method, p_wasgo_buffer_size_method);
// 			Variant * method = &v_method;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_method};
// 			Variant wasgo_local_ret = caller->call((String)"rpc", varargs, 1, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: rpc");}
// 		}
// 	}}

// void _wasgo_Node_wrapper_rpc_config(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_method, int p_wasgo_buffer_size_method, WasGoState::WasGoID p_mode){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_method = String();
// 			decode_variant(v_method, p_method, p_wasgo_buffer_size_method);
// 			Variant * method = &v_method;
// 			Variant v_mode = state->lookup_object(p_mode);
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_method, & v_mode};
// 			Variant wasgo_ret = caller->call((String)"rpc_config", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: rpc_config");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_rpc_id(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len, int p_peer_id, const uint8_t * p_method, int p_wasgo_buffer_size_method){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_peer_id = (int) p_peer_id;
// 			Variant v_method = String();
// 			decode_variant(v_method, p_method, p_wasgo_buffer_size_method);
// 			Variant * method = &v_method;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_peer_id, & v_method};
// 			Variant wasgo_local_ret = caller->call((String)"rpc_id", varargs, 2, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: rpc_id");}
// 		}
// 	}}

// void _wasgo_Node_wrapper_rpc_unreliable(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len, const uint8_t * p_method, int p_wasgo_buffer_size_method){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_method = String();
// 			decode_variant(v_method, p_method, p_wasgo_buffer_size_method);
// 			Variant * method = &v_method;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_method};
// 			Variant wasgo_local_ret = caller->call((String)"rpc_unreliable", varargs, 1, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: rpc_unreliable");}
// 		}
// 	}}

// void _wasgo_Node_wrapper_rpc_unreliable_id(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len, int p_peer_id, const uint8_t * p_method, int p_wasgo_buffer_size_method){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_peer_id = (int) p_peer_id;
// 			Variant v_method = String();
// 			decode_variant(v_method, p_method, p_wasgo_buffer_size_method);
// 			Variant * method = &v_method;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_peer_id, & v_method};
// 			Variant wasgo_local_ret = caller->call((String)"rpc_unreliable_id", varargs, 2, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: rpc_unreliable_id");}
// 		}
// 	}}

// void _wasgo_Node_wrapper_rset(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_property, int p_wasgo_buffer_size_property, const uint8_t * p_value, int p_wasgo_buffer_size_value){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Variant v_value = Variant();
// 			decode_variant(v_value, p_value, p_wasgo_buffer_size_value);
// 			Variant * value = &v_value;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property, & v_value};
// 			Variant wasgo_ret = caller->call((String)"rset", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: rset");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_rset_config(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_property, int p_wasgo_buffer_size_property, WasGoState::WasGoID p_mode){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Variant v_mode = state->lookup_object(p_mode);
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property, & v_mode};
// 			Variant wasgo_ret = caller->call((String)"rset_config", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: rset_config");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_rset_id(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, int p_peer_id, const uint8_t * p_property, int p_wasgo_buffer_size_property, const uint8_t * p_value, int p_wasgo_buffer_size_value){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_peer_id = (int) p_peer_id;
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Variant v_value = Variant();
// 			decode_variant(v_value, p_value, p_wasgo_buffer_size_value);
// 			Variant * value = &v_value;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_peer_id, & v_property, & v_value};
// 			Variant wasgo_ret = caller->call((String)"rset_id", varargs, 3, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: rset_id");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_rset_unreliable(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_property, int p_wasgo_buffer_size_property, const uint8_t * p_value, int p_wasgo_buffer_size_value){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Variant v_value = Variant();
// 			decode_variant(v_value, p_value, p_wasgo_buffer_size_value);
// 			Variant * value = &v_value;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property, & v_value};
// 			Variant wasgo_ret = caller->call((String)"rset_unreliable", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: rset_unreliable");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_rset_unreliable_id(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, int p_peer_id, const uint8_t * p_property, int p_wasgo_buffer_size_property, const uint8_t * p_value, int p_wasgo_buffer_size_value){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_peer_id = (int) p_peer_id;
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Variant v_value = Variant();
// 			decode_variant(v_value, p_value, p_wasgo_buffer_size_value);
// 			Variant * value = &v_value;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_peer_id, & v_property, & v_value};
// 			Variant wasgo_ret = caller->call((String)"rset_unreliable_id", varargs, 3, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: rset_unreliable_id");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_set_custom_multiplayer(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, WasGoState::WasGoID p_api){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_api = state->lookup_object(p_api);
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_api};
// 			Variant wasgo_ret = caller->call((String)"set_custom_multiplayer", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_custom_multiplayer");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_set_display_folded(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_fold){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_fold = (bool) p_fold;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_fold};
// 			Variant wasgo_ret = caller->call((String)"set_display_folded", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_display_folded");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_set_filename(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_filename, int p_wasgo_buffer_size_filename){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_filename = String();
// 			decode_variant(v_filename, p_filename, p_wasgo_buffer_size_filename);
// 			Variant * filename = &v_filename;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_filename};
// 			Variant wasgo_ret = caller->call((String)"set_filename", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_filename");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_set_name(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_name, int p_wasgo_buffer_size_name){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_name = String();
// 			decode_variant(v_name, p_name, p_wasgo_buffer_size_name);
// 			Variant * name = &v_name;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_name};
// 			Variant wasgo_ret = caller->call((String)"set_name", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_name");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_set_network_master(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, int p_id, bool p_recursive){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_id = (int) p_id;
// 			Variant v_recursive = (bool) p_recursive;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_id, & v_recursive};
// 			Variant wasgo_ret = caller->call((String)"set_network_master", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_network_master");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_set_owner(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, WasGoState::WasGoID p_owner){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_owner = state->lookup_object(p_owner);
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_owner};
// 			Variant wasgo_ret = caller->call((String)"set_owner", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_owner");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_set_pause_mode(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, WasGoState::WasGoID p_mode){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_mode = state->lookup_object(p_mode);
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_mode};
// 			Variant wasgo_ret = caller->call((String)"set_pause_mode", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_pause_mode");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_set_physics_process(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enable){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enable = (bool) p_enable;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enable};
// 			Variant wasgo_ret = caller->call((String)"set_physics_process", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_physics_process");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_set_physics_process_internal(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enable){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enable = (bool) p_enable;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enable};
// 			Variant wasgo_ret = caller->call((String)"set_physics_process_internal", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_physics_process_internal");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_set_process(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enable){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enable = (bool) p_enable;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enable};
// 			Variant wasgo_ret = caller->call((String)"set_process", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_process");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_set_process_input(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enable){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enable = (bool) p_enable;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enable};
// 			Variant wasgo_ret = caller->call((String)"set_process_input", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_process_input");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_set_process_internal(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enable){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enable = (bool) p_enable;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enable};
// 			Variant wasgo_ret = caller->call((String)"set_process_internal", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_process_internal");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_set_process_priority(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, int p_priority){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_priority = (int) p_priority;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_priority};
// 			Variant wasgo_ret = caller->call((String)"set_process_priority", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_process_priority");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_set_process_unhandled_input(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enable){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enable = (bool) p_enable;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enable};
// 			Variant wasgo_ret = caller->call((String)"set_process_unhandled_input", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_process_unhandled_input");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_set_process_unhandled_key_input(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enable){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enable = (bool) p_enable;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enable};
// 			Variant wasgo_ret = caller->call((String)"set_process_unhandled_key_input", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_process_unhandled_key_input");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_set_scene_instance_load_placeholder(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_load_placeholder){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_load_placeholder = (bool) p_load_placeholder;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_load_placeholder};
// 			Variant wasgo_ret = caller->call((String)"set_scene_instance_load_placeholder", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_scene_instance_load_placeholder");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node_wrapper_update_configuration_warning(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node *caller = (Node *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"update_configuration_warning", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: update_configuration_warning");}
// 		}
// 	}
// 	return;
// }

//     WasGoState::WasGoID _wasgo_Resource_constructor(){
//         return 0;
//     }
//     void _wasgo_Resource_destructor(WasGoState::WasGoID p_wasgo_id){

//     }

// WasGoState::WasGoID _wasgo_Resource_wrapper_duplicate(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_subresources){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Resource *caller = (Resource *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_subresources = (bool) p_subresources;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_subresources};
// 			Variant wasgo_ret = caller->call((String)"duplicate", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: duplicate");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// WasGoState::WasGoID _wasgo_Resource_wrapper_get_local_scene(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Resource *caller = (Resource *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"get_local_scene", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_local_scene");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// void _wasgo_Resource_wrapper_get_name(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Resource *caller = (Resource *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_local_ret = caller->call((String)"get_name", varargs, 0, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_name");}
// 		}
// 	}}

// void _wasgo_Resource_wrapper_get_path(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Resource *caller = (Resource *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_local_ret = caller->call((String)"get_path", varargs, 0, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_path");}
// 		}
// 	}}

// void _wasgo_Resource_wrapper_get_rid(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Resource *caller = (Resource *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_local_ret = caller->call((String)"get_rid", varargs, 0, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_rid");}
// 		}
// 	}}

// int _wasgo_Resource_wrapper_is_local_to_scene(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Resource *caller = (Resource *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_local_to_scene", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_local_to_scene");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// void _wasgo_Resource_wrapper_set_local_to_scene(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enable){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Resource *caller = (Resource *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enable = (bool) p_enable;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enable};
// 			Variant wasgo_ret = caller->call((String)"set_local_to_scene", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_local_to_scene");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Resource_wrapper_set_name(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_name, int p_wasgo_buffer_size_name){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Resource *caller = (Resource *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_name = String();
// 			decode_variant(v_name, p_name, p_wasgo_buffer_size_name);
// 			Variant * name = &v_name;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_name};
// 			Variant wasgo_ret = caller->call((String)"set_name", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_name");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Resource_wrapper_set_path(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_path, int p_wasgo_buffer_size_path){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Resource *caller = (Resource *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_path = String();
// 			decode_variant(v_path, p_path, p_wasgo_buffer_size_path);
// 			Variant * path = &v_path;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_path};
// 			Variant wasgo_ret = caller->call((String)"set_path", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_path");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Resource_wrapper_setup_local_to_scene(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Resource *caller = (Resource *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"setup_local_to_scene", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: setup_local_to_scene");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Resource_wrapper_take_over_path(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_path, int p_wasgo_buffer_size_path){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Resource *caller = (Resource *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_path = String();
// 			decode_variant(v_path, p_path, p_wasgo_buffer_size_path);
// 			Variant * path = &v_path;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_path};
// 			Variant wasgo_ret = caller->call((String)"take_over_path", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: take_over_path");}
// 		}
// 	}
// 	return;
// }

//     WasGoState::WasGoID _wasgo_SceneTree_constructor(){
//         return 0;
//     }
//     void _wasgo_SceneTree_destructor(WasGoState::WasGoID p_wasgo_id){

//     }

// void _wasgo_SceneTree_wrapper_call_group(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len, const uint8_t * p_group, int p_wasgo_buffer_size_group, const uint8_t * p_method, int p_wasgo_buffer_size_method){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_group = String();
// 			decode_variant(v_group, p_group, p_wasgo_buffer_size_group);
// 			Variant * group = &v_group;
// 			Variant v_method = String();
// 			decode_variant(v_method, p_method, p_wasgo_buffer_size_method);
// 			Variant * method = &v_method;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_group, & v_method};
// 			Variant wasgo_local_ret = caller->call((String)"call_group", varargs, 2, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: call_group");}
// 		}
// 	}}

// void _wasgo_SceneTree_wrapper_call_group_flags(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len, int p_flags, const uint8_t * p_group, int p_wasgo_buffer_size_group, const uint8_t * p_method, int p_wasgo_buffer_size_method){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_flags = (int) p_flags;
// 			Variant v_group = String();
// 			decode_variant(v_group, p_group, p_wasgo_buffer_size_group);
// 			Variant * group = &v_group;
// 			Variant v_method = String();
// 			decode_variant(v_method, p_method, p_wasgo_buffer_size_method);
// 			Variant * method = &v_method;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_flags, & v_group, & v_method};
// 			Variant wasgo_local_ret = caller->call((String)"call_group_flags", varargs, 3, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: call_group_flags");}
// 		}
// 	}}

// WasGoState::WasGoID _wasgo_SceneTree_wrapper_change_scene(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_path, int p_wasgo_buffer_size_path){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_path = String();
// 			decode_variant(v_path, p_path, p_wasgo_buffer_size_path);
// 			Variant * path = &v_path;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_path};
// 			Variant wasgo_ret = caller->call((String)"change_scene", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: change_scene");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// WasGoState::WasGoID _wasgo_SceneTree_wrapper_change_scene_to(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, WasGoState::WasGoID p_packed_scene){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_packed_scene = state->lookup_object(p_packed_scene);
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_packed_scene};
// 			Variant wasgo_ret = caller->call((String)"change_scene_to", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: change_scene_to");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// WasGoState::WasGoID _wasgo_SceneTree_wrapper_create_timer(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, float p_time_sec, bool p_pause_mode_process){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_time_sec = (float) p_time_sec;
// 			Variant v_pause_mode_process = (bool) p_pause_mode_process;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_time_sec, & v_pause_mode_process};
// 			Variant wasgo_ret = caller->call((String)"create_timer", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: create_timer");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// WasGoState::WasGoID _wasgo_SceneTree_wrapper_get_current_scene(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"get_current_scene", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_current_scene");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// WasGoState::WasGoID _wasgo_SceneTree_wrapper_get_edited_scene_root(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"get_edited_scene_root", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_edited_scene_root");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// int _wasgo_SceneTree_wrapper_get_frame(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"get_frame", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_frame");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// WasGoState::WasGoID _wasgo_SceneTree_wrapper_get_multiplayer(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"get_multiplayer", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_multiplayer");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// WasGoState::WasGoID _wasgo_SceneTree_wrapper_get_network_connected_peers(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"get_network_connected_peers", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_network_connected_peers");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// WasGoState::WasGoID _wasgo_SceneTree_wrapper_get_network_peer(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"get_network_peer", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_network_peer");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// int _wasgo_SceneTree_wrapper_get_network_unique_id(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"get_network_unique_id", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_network_unique_id");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_SceneTree_wrapper_get_node_count(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"get_node_count", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_node_count");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// WasGoState::WasGoID _wasgo_SceneTree_wrapper_get_nodes_in_group(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_group, int p_wasgo_buffer_size_group){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_group = String();
// 			decode_variant(v_group, p_group, p_wasgo_buffer_size_group);
// 			Variant * group = &v_group;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_group};
// 			Variant wasgo_ret = caller->call((String)"get_nodes_in_group", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_nodes_in_group");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// WasGoState::WasGoID _wasgo_SceneTree_wrapper_get_root(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"get_root", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_root");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// int _wasgo_SceneTree_wrapper_get_rpc_sender_id(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"get_rpc_sender_id", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_rpc_sender_id");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_SceneTree_wrapper_has_group(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_name, int p_wasgo_buffer_size_name){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_name = String();
// 			decode_variant(v_name, p_name, p_wasgo_buffer_size_name);
// 			Variant * name = &v_name;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_name};
// 			int wasgo_ret = (int) caller->call((String)"has_group", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: has_group");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_SceneTree_wrapper_has_network_peer(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"has_network_peer", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: has_network_peer");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_SceneTree_wrapper_is_debugging_collisions_hint(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_debugging_collisions_hint", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_debugging_collisions_hint");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_SceneTree_wrapper_is_debugging_navigation_hint(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_debugging_navigation_hint", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_debugging_navigation_hint");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_SceneTree_wrapper_is_input_handled(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_input_handled", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_input_handled");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_SceneTree_wrapper_is_multiplayer_poll_enabled(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_multiplayer_poll_enabled", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_multiplayer_poll_enabled");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_SceneTree_wrapper_is_network_server(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_network_server", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_network_server");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_SceneTree_wrapper_is_paused(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_paused", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_paused");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_SceneTree_wrapper_is_refusing_new_network_connections(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_refusing_new_network_connections", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_refusing_new_network_connections");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_SceneTree_wrapper_is_using_font_oversampling(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_using_font_oversampling", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_using_font_oversampling");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// void _wasgo_SceneTree_wrapper_notify_group(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_group, int p_wasgo_buffer_size_group, int p_notification){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_group = String();
// 			decode_variant(v_group, p_group, p_wasgo_buffer_size_group);
// 			Variant * group = &v_group;
// 			Variant v_notification = (int) p_notification;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_group, & v_notification};
// 			Variant wasgo_ret = caller->call((String)"notify_group", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: notify_group");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_SceneTree_wrapper_notify_group_flags(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, int p_call_flags, const uint8_t * p_group, int p_wasgo_buffer_size_group, int p_notification){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_call_flags = (int) p_call_flags;
// 			Variant v_group = String();
// 			decode_variant(v_group, p_group, p_wasgo_buffer_size_group);
// 			Variant * group = &v_group;
// 			Variant v_notification = (int) p_notification;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_call_flags, & v_group, & v_notification};
// 			Variant wasgo_ret = caller->call((String)"notify_group_flags", varargs, 3, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: notify_group_flags");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_SceneTree_wrapper_queue_delete(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, WasGoState::WasGoID p_obj){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_obj = state->lookup_object(p_obj);
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_obj};
// 			Variant wasgo_ret = caller->call((String)"queue_delete", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: queue_delete");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_SceneTree_wrapper_quit(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, int p_exit_code){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_exit_code = (int) p_exit_code;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_exit_code};
// 			Variant wasgo_ret = caller->call((String)"quit", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: quit");}
// 		}
// 	}
// 	return;
// }

// WasGoState::WasGoID _wasgo_SceneTree_wrapper_reload_current_scene(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"reload_current_scene", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: reload_current_scene");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// void _wasgo_SceneTree_wrapper_set_auto_accept_quit(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enabled){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enabled = (bool) p_enabled;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enabled};
// 			Variant wasgo_ret = caller->call((String)"set_auto_accept_quit", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_auto_accept_quit");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_SceneTree_wrapper_set_current_scene(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, WasGoState::WasGoID p_child_node){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_child_node = state->lookup_object(p_child_node);
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_child_node};
// 			Variant wasgo_ret = caller->call((String)"set_current_scene", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_current_scene");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_SceneTree_wrapper_set_debug_collisions_hint(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enable){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enable = (bool) p_enable;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enable};
// 			Variant wasgo_ret = caller->call((String)"set_debug_collisions_hint", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_debug_collisions_hint");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_SceneTree_wrapper_set_debug_navigation_hint(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enable){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enable = (bool) p_enable;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enable};
// 			Variant wasgo_ret = caller->call((String)"set_debug_navigation_hint", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_debug_navigation_hint");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_SceneTree_wrapper_set_edited_scene_root(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, WasGoState::WasGoID p_scene){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_scene = state->lookup_object(p_scene);
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_scene};
// 			Variant wasgo_ret = caller->call((String)"set_edited_scene_root", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_edited_scene_root");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_SceneTree_wrapper_set_group(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_group, int p_wasgo_buffer_size_group, const uint8_t * p_property, int p_wasgo_buffer_size_property, const uint8_t * p_value, int p_wasgo_buffer_size_value){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_group = String();
// 			decode_variant(v_group, p_group, p_wasgo_buffer_size_group);
// 			Variant * group = &v_group;
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Variant v_value = Variant();
// 			decode_variant(v_value, p_value, p_wasgo_buffer_size_value);
// 			Variant * value = &v_value;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_group, & v_property, & v_value};
// 			Variant wasgo_ret = caller->call((String)"set_group", varargs, 3, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_group");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_SceneTree_wrapper_set_group_flags(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, int p_call_flags, const uint8_t * p_group, int p_wasgo_buffer_size_group, const uint8_t * p_property, int p_wasgo_buffer_size_property, const uint8_t * p_value, int p_wasgo_buffer_size_value){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_call_flags = (int) p_call_flags;
// 			Variant v_group = String();
// 			decode_variant(v_group, p_group, p_wasgo_buffer_size_group);
// 			Variant * group = &v_group;
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Variant v_value = Variant();
// 			decode_variant(v_value, p_value, p_wasgo_buffer_size_value);
// 			Variant * value = &v_value;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_call_flags, & v_group, & v_property, & v_value};
// 			Variant wasgo_ret = caller->call((String)"set_group_flags", varargs, 4, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_group_flags");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_SceneTree_wrapper_set_input_as_handled(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"set_input_as_handled", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_input_as_handled");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_SceneTree_wrapper_set_multiplayer(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, WasGoState::WasGoID p_multiplayer){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_multiplayer = state->lookup_object(p_multiplayer);
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_multiplayer};
// 			Variant wasgo_ret = caller->call((String)"set_multiplayer", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_multiplayer");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_SceneTree_wrapper_set_multiplayer_poll_enabled(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enabled){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enabled = (bool) p_enabled;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enabled};
// 			Variant wasgo_ret = caller->call((String)"set_multiplayer_poll_enabled", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_multiplayer_poll_enabled");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_SceneTree_wrapper_set_network_peer(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, WasGoState::WasGoID p_peer){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_peer = state->lookup_object(p_peer);
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_peer};
// 			Variant wasgo_ret = caller->call((String)"set_network_peer", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_network_peer");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_SceneTree_wrapper_set_pause(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enable){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enable = (bool) p_enable;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enable};
// 			Variant wasgo_ret = caller->call((String)"set_pause", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_pause");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_SceneTree_wrapper_set_quit_on_go_back(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enabled){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enabled = (bool) p_enabled;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enabled};
// 			Variant wasgo_ret = caller->call((String)"set_quit_on_go_back", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_quit_on_go_back");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_SceneTree_wrapper_set_refuse_new_network_connections(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_refuse){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_refuse = (bool) p_refuse;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_refuse};
// 			Variant wasgo_ret = caller->call((String)"set_refuse_new_network_connections", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_refuse_new_network_connections");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_SceneTree_wrapper_set_screen_stretch(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, WasGoState::WasGoID p_mode, WasGoState::WasGoID p_aspect, const uint8_t * p_minsize, int p_wasgo_buffer_size_minsize, float p_shrink){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_mode = state->lookup_object(p_mode);
// 			Variant v_aspect = state->lookup_object(p_aspect);
// 			Variant v_minsize = Vector2();
// 			decode_variant(v_minsize, p_minsize, p_wasgo_buffer_size_minsize);
// 			Variant * minsize = &v_minsize;
// 			Variant v_shrink = (float) p_shrink;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_mode, & v_aspect, & v_minsize, & v_shrink};
// 			Variant wasgo_ret = caller->call((String)"set_screen_stretch", varargs, 4, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_screen_stretch");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_SceneTree_wrapper_set_use_font_oversampling(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enable){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		SceneTree *caller = (SceneTree *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enable = (bool) p_enable;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enable};
// 			Variant wasgo_ret = caller->call((String)"set_use_font_oversampling", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_use_font_oversampling");}
// 		}
// 	}
// 	return;
// }

//     WasGoState::WasGoID _wasgo_Node3D_constructor(){
//         return 0;
//     }
//     void _wasgo_Node3D_destructor(WasGoState::WasGoID p_wasgo_id){

//     }

// void _wasgo_Node3D_wrapper_force_update_transform(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"force_update_transform", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: force_update_transform");}
// 		}
// 	}
// 	return;
// }

// WasGoState::WasGoID _wasgo_Node3D_wrapper_get_gizmo(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"get_gizmo", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_gizmo");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// void _wasgo_Node3D_wrapper_get_global_transform(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_local_ret = caller->call((String)"get_global_transform", varargs, 0, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_global_transform");}
// 		}
// 	}}

// WasGoState::WasGoID _wasgo_Node3D_wrapper_get_parent_Node3D(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"get_parent_Node3D", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_parent_Node3D");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// void _wasgo_Node3D_wrapper_get_rotation(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_local_ret = caller->call((String)"get_rotation", varargs, 0, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_rotation");}
// 		}
// 	}}

// void _wasgo_Node3D_wrapper_get_rotation_degrees(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_local_ret = caller->call((String)"get_rotation_degrees", varargs, 0, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_rotation_degrees");}
// 		}
// 	}}

// void _wasgo_Node3D_wrapper_get_scale(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_local_ret = caller->call((String)"get_scale", varargs, 0, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_scale");}
// 		}
// 	}}

// void _wasgo_Node3D_wrapper_get_transform(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_local_ret = caller->call((String)"get_transform", varargs, 0, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_transform");}
// 		}
// 	}}

// void _wasgo_Node3D_wrapper_get_translation(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_local_ret = caller->call((String)"get_translation", varargs, 0, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_translation");}
// 		}
// 	}}

// WasGoState::WasGoID _wasgo_Node3D_wrapper_get_world(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"get_world", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_world");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// void _wasgo_Node3D_wrapper_global_rotate(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_axis, int p_wasgo_buffer_size_axis, float p_angle){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_axis = Vector3();
// 			decode_variant(v_axis, p_axis, p_wasgo_buffer_size_axis);
// 			Variant * axis = &v_axis;
// 			Variant v_angle = (float) p_angle;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_axis, & v_angle};
// 			Variant wasgo_ret = caller->call((String)"global_rotate", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: global_rotate");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_global_scale(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_scale, int p_wasgo_buffer_size_scale){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_scale = Vector3();
// 			decode_variant(v_scale, p_scale, p_wasgo_buffer_size_scale);
// 			Variant * scale = &v_scale;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_scale};
// 			Variant wasgo_ret = caller->call((String)"global_scale", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: global_scale");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_global_translate(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_offset, int p_wasgo_buffer_size_offset){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_offset = Vector3();
// 			decode_variant(v_offset, p_offset, p_wasgo_buffer_size_offset);
// 			Variant * offset = &v_offset;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_offset};
// 			Variant wasgo_ret = caller->call((String)"global_translate", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: global_translate");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_hide(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"hide", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: hide");}
// 		}
// 	}
// 	return;
// }

// int _wasgo_Node3D_wrapper_is_local_transform_notification_enabled(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_local_transform_notification_enabled", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_local_transform_notification_enabled");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Node3D_wrapper_is_scale_disabled(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_scale_disabled", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_scale_disabled");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Node3D_wrapper_is_set_as_toplevel(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_set_as_toplevel", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_set_as_toplevel");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Node3D_wrapper_is_transform_notification_enabled(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_transform_notification_enabled", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_transform_notification_enabled");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Node3D_wrapper_is_visible(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_visible", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_visible");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// int _wasgo_Node3D_wrapper_is_visible_in_tree(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_visible_in_tree", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_visible_in_tree");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// void _wasgo_Node3D_wrapper_look_at(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_target, int p_wasgo_buffer_size_target, const uint8_t * p_up, int p_wasgo_buffer_size_up){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_target = Vector3();
// 			decode_variant(v_target, p_target, p_wasgo_buffer_size_target);
// 			Variant * target = &v_target;
// 			Variant v_up = Vector3();
// 			decode_variant(v_up, p_up, p_wasgo_buffer_size_up);
// 			Variant * up = &v_up;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_target, & v_up};
// 			Variant wasgo_ret = caller->call((String)"look_at", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: look_at");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_look_at_from_position(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_position, int p_wasgo_buffer_size_position, const uint8_t * p_target, int p_wasgo_buffer_size_target, const uint8_t * p_up, int p_wasgo_buffer_size_up){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_position = Vector3();
// 			decode_variant(v_position, p_position, p_wasgo_buffer_size_position);
// 			Variant * position = &v_position;
// 			Variant v_target = Vector3();
// 			decode_variant(v_target, p_target, p_wasgo_buffer_size_target);
// 			Variant * target = &v_target;
// 			Variant v_up = Vector3();
// 			decode_variant(v_up, p_up, p_wasgo_buffer_size_up);
// 			Variant * up = &v_up;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_position, & v_target, & v_up};
// 			Variant wasgo_ret = caller->call((String)"look_at_from_position", varargs, 3, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: look_at_from_position");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_orthonormalize(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"orthonormalize", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: orthonormalize");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_rotate(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_axis, int p_wasgo_buffer_size_axis, float p_angle){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_axis = Vector3();
// 			decode_variant(v_axis, p_axis, p_wasgo_buffer_size_axis);
// 			Variant * axis = &v_axis;
// 			Variant v_angle = (float) p_angle;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_axis, & v_angle};
// 			Variant wasgo_ret = caller->call((String)"rotate", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: rotate");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_rotate_object_local(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_axis, int p_wasgo_buffer_size_axis, float p_angle){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_axis = Vector3();
// 			decode_variant(v_axis, p_axis, p_wasgo_buffer_size_axis);
// 			Variant * axis = &v_axis;
// 			Variant v_angle = (float) p_angle;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_axis, & v_angle};
// 			Variant wasgo_ret = caller->call((String)"rotate_object_local", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: rotate_object_local");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_rotate_x(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, float p_angle){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_angle = (float) p_angle;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_angle};
// 			Variant wasgo_ret = caller->call((String)"rotate_x", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: rotate_x");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_rotate_y(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, float p_angle){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_angle = (float) p_angle;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_angle};
// 			Variant wasgo_ret = caller->call((String)"rotate_y", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: rotate_y");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_rotate_z(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, float p_angle){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_angle = (float) p_angle;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_angle};
// 			Variant wasgo_ret = caller->call((String)"rotate_z", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: rotate_z");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_scale_object_local(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_scale, int p_wasgo_buffer_size_scale){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_scale = Vector3();
// 			decode_variant(v_scale, p_scale, p_wasgo_buffer_size_scale);
// 			Variant * scale = &v_scale;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_scale};
// 			Variant wasgo_ret = caller->call((String)"scale_object_local", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: scale_object_local");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_set_as_toplevel(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enable){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enable = (bool) p_enable;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enable};
// 			Variant wasgo_ret = caller->call((String)"set_as_toplevel", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_as_toplevel");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_set_disable_scale(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_disable){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_disable = (bool) p_disable;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_disable};
// 			Variant wasgo_ret = caller->call((String)"set_disable_scale", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_disable_scale");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_set_gizmo(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, WasGoState::WasGoID p_gizmo){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_gizmo = state->lookup_object(p_gizmo);
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_gizmo};
// 			Variant wasgo_ret = caller->call((String)"set_gizmo", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_gizmo");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_set_global_transform(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_global, int p_wasgo_buffer_size_global){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_global = Transform3D();
// 			decode_variant(v_global, p_global, p_wasgo_buffer_size_global);
// 			Variant * global = &v_global;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_global};
// 			Variant wasgo_ret = caller->call((String)"set_global_transform", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_global_transform");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_set_identity(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"set_identity", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_identity");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_set_ignore_transform_notification(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enabled){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enabled = (bool) p_enabled;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enabled};
// 			Variant wasgo_ret = caller->call((String)"set_ignore_transform_notification", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_ignore_transform_notification");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_set_notify_local_transform(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enable){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enable = (bool) p_enable;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enable};
// 			Variant wasgo_ret = caller->call((String)"set_notify_local_transform", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_notify_local_transform");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_set_notify_transform(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enable){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enable = (bool) p_enable;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enable};
// 			Variant wasgo_ret = caller->call((String)"set_notify_transform", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_notify_transform");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_set_rotation(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_euler, int p_wasgo_buffer_size_euler){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_euler = Vector3();
// 			decode_variant(v_euler, p_euler, p_wasgo_buffer_size_euler);
// 			Variant * euler = &v_euler;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_euler};
// 			Variant wasgo_ret = caller->call((String)"set_rotation", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_rotation");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_set_rotation_degrees(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_euler_degrees, int p_wasgo_buffer_size_euler_degrees){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_euler_degrees = Vector3();
// 			decode_variant(v_euler_degrees, p_euler_degrees, p_wasgo_buffer_size_euler_degrees);
// 			Variant * euler_degrees = &v_euler_degrees;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_euler_degrees};
// 			Variant wasgo_ret = caller->call((String)"set_rotation_degrees", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_rotation_degrees");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_set_scale(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_scale, int p_wasgo_buffer_size_scale){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_scale = Vector3();
// 			decode_variant(v_scale, p_scale, p_wasgo_buffer_size_scale);
// 			Variant * scale = &v_scale;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_scale};
// 			Variant wasgo_ret = caller->call((String)"set_scale", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_scale");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_set_transform(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_local, int p_wasgo_buffer_size_local){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_local = Transform3D();
// 			decode_variant(v_local, p_local, p_wasgo_buffer_size_local);
// 			Variant * local = &v_local;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_local};
// 			Variant wasgo_ret = caller->call((String)"set_transform", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_transform");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_set_translation(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_translation, int p_wasgo_buffer_size_translation){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_translation = Vector3();
// 			decode_variant(v_translation, p_translation, p_wasgo_buffer_size_translation);
// 			Variant * translation = &v_translation;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_translation};
// 			Variant wasgo_ret = caller->call((String)"set_translation", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_translation");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_set_visible(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_visible){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_visible = (bool) p_visible;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_visible};
// 			Variant wasgo_ret = caller->call((String)"set_visible", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_visible");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_show(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"show", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: show");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_to_global(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len, const uint8_t * p_local_point, int p_wasgo_buffer_size_local_point){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_local_point = Vector3();
// 			decode_variant(v_local_point, p_local_point, p_wasgo_buffer_size_local_point);
// 			Variant * local_point = &v_local_point;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_local_point};
// 			Variant wasgo_local_ret = caller->call((String)"to_global", varargs, 1, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: to_global");}
// 		}
// 	}}

// void _wasgo_Node3D_wrapper_to_local(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len, const uint8_t * p_global_point, int p_wasgo_buffer_size_global_point){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_global_point = Vector3();
// 			decode_variant(v_global_point, p_global_point, p_wasgo_buffer_size_global_point);
// 			Variant * global_point = &v_global_point;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_global_point};
// 			Variant wasgo_local_ret = caller->call((String)"to_local", varargs, 1, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: to_local");}
// 		}
// 	}}

// void _wasgo_Node3D_wrapper_translate(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_offset, int p_wasgo_buffer_size_offset){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_offset = Vector3();
// 			decode_variant(v_offset, p_offset, p_wasgo_buffer_size_offset);
// 			Variant * offset = &v_offset;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_offset};
// 			Variant wasgo_ret = caller->call((String)"translate", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: translate");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_translate_object_local(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_offset, int p_wasgo_buffer_size_offset){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_offset = Vector3();
// 			decode_variant(v_offset, p_offset, p_wasgo_buffer_size_offset);
// 			Variant * offset = &v_offset;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_offset};
// 			Variant wasgo_ret = caller->call((String)"translate_object_local", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: translate_object_local");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_Node3D_wrapper_update_gizmo(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		Node3D *caller = (Node3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"update_gizmo", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: update_gizmo");}
// 		}
// 	}
// 	return;
// }

//     WasGoState::WasGoID _wasgo_VelocityTracker3D_constructor(){
//         return 0;
//     }
//     void _wasgo_VelocityTracker3D_destructor(WasGoState::WasGoID p_wasgo_id){

//     }

// void _wasgo_VelocityTracker3D_wrapper_get_tracked_linear_velocity(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		VelocityTracker3D *caller = (VelocityTracker3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_local_ret = caller->call((String)"get_tracked_linear_velocity", varargs, 0, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_tracked_linear_velocity");}
// 		}
// 	}}

// int _wasgo_VelocityTracker3D_wrapper_is_tracking_physics_step(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		VelocityTracker3D *caller = (VelocityTracker3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"is_tracking_physics_step", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: is_tracking_physics_step");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// void _wasgo_VelocityTracker3D_wrapper_reset(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_position, int p_wasgo_buffer_size_position){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		VelocityTracker3D *caller = (VelocityTracker3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_position = Vector3();
// 			decode_variant(v_position, p_position, p_wasgo_buffer_size_position);
// 			Variant * position = &v_position;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_position};
// 			Variant wasgo_ret = caller->call((String)"reset", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: reset");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_VelocityTracker3D_wrapper_set_track_physics_step(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, bool p_enable){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		VelocityTracker3D *caller = (VelocityTracker3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_enable = (bool) p_enable;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_enable};
// 			Variant wasgo_ret = caller->call((String)"set_track_physics_step", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_track_physics_step");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_VelocityTracker3D_wrapper_update_position(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_position, int p_wasgo_buffer_size_position){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		VelocityTracker3D *caller = (VelocityTracker3D *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_position = Vector3();
// 			decode_variant(v_position, p_position, p_wasgo_buffer_size_position);
// 			Variant * position = &v_position;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_position};
// 			Variant wasgo_ret = caller->call((String)"update_position", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: update_position");}
// 		}
// 	}
// 	return;
// }

//     WasGoState::WasGoID _wasgo_WasGoState_constructor(){
//         return 0;
//     }
//     void _wasgo_WasGoState_destructor(WasGoState::WasGoID p_wasgo_id){

//     }

// int _wasgo_WasGoState_wrapper_get_heap_size(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"get_heap_size", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_heap_size");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// WasGoState::WasGoID _wasgo_WasGoState_wrapper_get_properties(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"get_properties", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_properties");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// void _wasgo_WasGoState_wrapper_get_property_aabb(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len, const uint8_t * p_property, int p_wasgo_buffer_size_property){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property};
// 			Variant wasgo_local_ret = caller->call((String)"get_property_aabb", varargs, 1, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_property_aabb");}
// 		}
// 	}}

// void _wasgo_WasGoState_wrapper_get_property_basis(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len, const uint8_t * p_property, int p_wasgo_buffer_size_property){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property};
// 			Variant wasgo_local_ret = caller->call((String)"get_property_basis", varargs, 1, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_property_basis");}
// 		}
// 	}}

// int _wasgo_WasGoState_wrapper_get_property_bool(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_property, int p_wasgo_buffer_size_property){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property};
// 			int wasgo_ret = (int) caller->call((String)"get_property_bool", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_property_bool");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// void _wasgo_WasGoState_wrapper_get_property_color(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len, const uint8_t * p_property, int p_wasgo_buffer_size_property){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property};
// 			Variant wasgo_local_ret = caller->call((String)"get_property_color", varargs, 1, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_property_color");}
// 		}
// 	}}

// float _wasgo_WasGoState_wrapper_get_property_float(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_property, int p_wasgo_buffer_size_property){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property};
// 			float wasgo_ret = (float) caller->call((String)"get_property_float", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_property_float");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return float();
// }

// int _wasgo_WasGoState_wrapper_get_property_int(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_property, int p_wasgo_buffer_size_property){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property};
// 			int wasgo_ret = (int) caller->call((String)"get_property_int", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_property_int");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// void _wasgo_WasGoState_wrapper_get_property_nodepath(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len, const uint8_t * p_property, int p_wasgo_buffer_size_property){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property};
// 			Variant wasgo_local_ret = caller->call((String)"get_property_nodepath", varargs, 1, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_property_nodepath");}
// 		}
// 	}}

// void _wasgo_WasGoState_wrapper_get_property_plane(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len, const uint8_t * p_property, int p_wasgo_buffer_size_property){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property};
// 			Variant wasgo_local_ret = caller->call((String)"get_property_plane", varargs, 1, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_property_plane");}
// 		}
// 	}}

// void _wasgo_WasGoState_wrapper_get_property_quat(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len, const uint8_t * p_property, int p_wasgo_buffer_size_property){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property};
// 			Variant wasgo_local_ret = caller->call((String)"get_property_quat", varargs, 1, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_property_quat");}
// 		}
// 	}}

// void _wasgo_WasGoState_wrapper_get_property_rect2(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len, const uint8_t * p_property, int p_wasgo_buffer_size_property){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property};
// 			Variant wasgo_local_ret = caller->call((String)"get_property_rect2", varargs, 1, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_property_rect2");}
// 		}
// 	}}

// void _wasgo_WasGoState_wrapper_get_property_string(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len, const uint8_t * p_property, int p_wasgo_buffer_size_property){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property};
// 			Variant wasgo_local_ret = caller->call((String)"get_property_string", varargs, 1, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_property_string");}
// 		}
// 	}}

// void _wasgo_WasGoState_wrapper_get_property_transform(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len, const uint8_t * p_property, int p_wasgo_buffer_size_property){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property};
// 			Variant wasgo_local_ret = caller->call((String)"get_property_transform", varargs, 1, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_property_transform");}
// 		}
// 	}}

// void _wasgo_WasGoState_wrapper_get_property_transform2d(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len, const uint8_t * p_property, int p_wasgo_buffer_size_property){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property};
// 			Variant wasgo_local_ret = caller->call((String)"get_property_transform2d", varargs, 1, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_property_transform2d");}
// 		}
// 	}}

// void _wasgo_WasGoState_wrapper_get_property_vector2(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len, const uint8_t * p_property, int p_wasgo_buffer_size_property){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property};
// 			Variant wasgo_local_ret = caller->call((String)"get_property_vector2", varargs, 1, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_property_vector2");}
// 		}
// 	}}

// void _wasgo_WasGoState_wrapper_get_property_vector3(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, uint8_t * wasgo_ret, int wasgo_ret_len, const uint8_t * p_property, int p_wasgo_buffer_size_property){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property};
// 			Variant wasgo_local_ret = caller->call((String)"get_property_vector3", varargs, 1, error);
// 			Error wasgo_encode_error = encode_variant(wasgo_local_ret, wasgo_ret, wasgo_ret_len);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_property_vector3");}
// 		}
// 	}}

// int _wasgo_WasGoState_wrapper_get_stack_size(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			int wasgo_ret = (int) caller->call((String)"get_stack_size", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_stack_size");}
// 			return wasgo_ret;
// 		}
// 	}
// 	return int();
// }

// WasGoState::WasGoID _wasgo_WasGoState_wrapper_get_wasm_script(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Callable::CallError error;
// 			const Variant ** varargs = nullptr;
// 			Variant wasgo_ret = caller->call((String)"get_wasm_script", varargs, 0, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: get_wasm_script");}
// 			return state->handle_return_variant(wasgo_ret);
// 		}
// 	}
// 	return WasGoState::WasGoID();
// }

// void _wasgo_WasGoState_wrapper_set_heap_size(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, int p_heap_size){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_heap_size = (int) p_heap_size;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_heap_size};
// 			Variant wasgo_ret = caller->call((String)"set_heap_size", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_heap_size");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_WasGoState_wrapper_set_int_property(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, int p_value, const uint8_t * p_key, int p_wasgo_buffer_size_key){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_value = (int) p_value;
// 			Variant v_key = String();
// 			decode_variant(v_key, p_key, p_wasgo_buffer_size_key);
// 			Variant * key = &v_key;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_value, & v_key};
// 			Variant wasgo_ret = caller->call((String)"set_int_property", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_int_property");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_WasGoState_wrapper_set_properties(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, WasGoState::WasGoID p_properties){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_properties = state->lookup_object(p_properties);
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_properties};
// 			Variant wasgo_ret = caller->call((String)"set_properties", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_properties");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_WasGoState_wrapper_set_property_aabb(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_property, int p_wasgo_buffer_size_property, const uint8_t * p_value, int p_wasgo_buffer_size_value){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Variant v_value = AABB();
// 			decode_variant(v_value, p_value, p_wasgo_buffer_size_value);
// 			Variant * value = &v_value;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property, & v_value};
// 			Variant wasgo_ret = caller->call((String)"set_property_aabb", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_property_aabb");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_WasGoState_wrapper_set_property_basis(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_property, int p_wasgo_buffer_size_property, const uint8_t * p_value, int p_wasgo_buffer_size_value){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Variant v_value = Basis();
// 			decode_variant(v_value, p_value, p_wasgo_buffer_size_value);
// 			Variant * value = &v_value;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property, & v_value};
// 			Variant wasgo_ret = caller->call((String)"set_property_basis", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_property_basis");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_WasGoState_wrapper_set_property_bool(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_property, int p_wasgo_buffer_size_property, bool p_value){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Variant v_value = (bool) p_value;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property, & v_value};
// 			Variant wasgo_ret = caller->call((String)"set_property_bool", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_property_bool");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_WasGoState_wrapper_set_property_color(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_property, int p_wasgo_buffer_size_property, const uint8_t * p_value, int p_wasgo_buffer_size_value){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Variant v_value = Color();
// 			decode_variant(v_value, p_value, p_wasgo_buffer_size_value);
// 			Variant * value = &v_value;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property, & v_value};
// 			Variant wasgo_ret = caller->call((String)"set_property_color", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_property_color");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_WasGoState_wrapper_set_property_float(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_property, int p_wasgo_buffer_size_property, float p_value){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Variant v_value = (float) p_value;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property, & v_value};
// 			Variant wasgo_ret = caller->call((String)"set_property_float", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_property_float");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_WasGoState_wrapper_set_property_int(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_property, int p_wasgo_buffer_size_property, int p_value){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Variant v_value = (int) p_value;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property, & v_value};
// 			Variant wasgo_ret = caller->call((String)"set_property_int", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_property_int");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_WasGoState_wrapper_set_property_nodepath(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_property, int p_wasgo_buffer_size_property, const uint8_t * p_value, int p_wasgo_buffer_size_value){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Variant v_value = NodePath();
// 			decode_variant(v_value, p_value, p_wasgo_buffer_size_value);
// 			Variant * value = &v_value;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property, & v_value};
// 			Variant wasgo_ret = caller->call((String)"set_property_nodepath", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_property_nodepath");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_WasGoState_wrapper_set_property_plane(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_property, int p_wasgo_buffer_size_property, const uint8_t * p_value, int p_wasgo_buffer_size_value){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Variant v_value = Plane();
// 			decode_variant(v_value, p_value, p_wasgo_buffer_size_value);
// 			Variant * value = &v_value;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property, & v_value};
// 			Variant wasgo_ret = caller->call((String)"set_property_plane", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_property_plane");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_WasGoState_wrapper_set_property_quat(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_property, int p_wasgo_buffer_size_property, const uint8_t * p_value, int p_wasgo_buffer_size_value){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Variant v_value = Quaternion();
// 			decode_variant(v_value, p_value, p_wasgo_buffer_size_value);
// 			Variant * value = &v_value;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property, & v_value};
// 			Variant wasgo_ret = caller->call((String)"set_property_quat", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_property_quat");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_WasGoState_wrapper_set_property_rect2(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_property, int p_wasgo_buffer_size_property, const uint8_t * p_value, int p_wasgo_buffer_size_value){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Variant v_value = Rect2();
// 			decode_variant(v_value, p_value, p_wasgo_buffer_size_value);
// 			Variant * value = &v_value;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property, & v_value};
// 			Variant wasgo_ret = caller->call((String)"set_property_rect2", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_property_rect2");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_WasGoState_wrapper_set_property_string(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_property, int p_wasgo_buffer_size_property, const uint8_t * p_value, int p_wasgo_buffer_size_value){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Variant v_value = String();
// 			decode_variant(v_value, p_value, p_wasgo_buffer_size_value);
// 			Variant * value = &v_value;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property, & v_value};
// 			Variant wasgo_ret = caller->call((String)"set_property_string", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_property_string");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_WasGoState_wrapper_set_property_transform(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_property, int p_wasgo_buffer_size_property, const uint8_t * p_value, int p_wasgo_buffer_size_value){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Variant v_value = Transform3D();
// 			decode_variant(v_value, p_value, p_wasgo_buffer_size_value);
// 			Variant * value = &v_value;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property, & v_value};
// 			Variant wasgo_ret = caller->call((String)"set_property_transform", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_property_transform");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_WasGoState_wrapper_set_property_transform2d(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_property, int p_wasgo_buffer_size_property, const uint8_t * p_value, int p_wasgo_buffer_size_value){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Variant v_value = Transform2D();
// 			decode_variant(v_value, p_value, p_wasgo_buffer_size_value);
// 			Variant * value = &v_value;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property, & v_value};
// 			Variant wasgo_ret = caller->call((String)"set_property_transform2d", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_property_transform2d");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_WasGoState_wrapper_set_property_vector2(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_property, int p_wasgo_buffer_size_property, const uint8_t * p_value, int p_wasgo_buffer_size_value){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Variant v_value = Vector2();
// 			decode_variant(v_value, p_value, p_wasgo_buffer_size_value);
// 			Variant * value = &v_value;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property, & v_value};
// 			Variant wasgo_ret = caller->call((String)"set_property_vector2", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_property_vector2");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_WasGoState_wrapper_set_property_vector3(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, const uint8_t * p_property, int p_wasgo_buffer_size_property, const uint8_t * p_value, int p_wasgo_buffer_size_value){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_property = String();
// 			decode_variant(v_property, p_property, p_wasgo_buffer_size_property);
// 			Variant * property = &v_property;
// 			Variant v_value = Vector3();
// 			decode_variant(v_value, p_value, p_wasgo_buffer_size_value);
// 			Variant * value = &v_value;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_property, & v_value};
// 			Variant wasgo_ret = caller->call((String)"set_property_vector3", varargs, 2, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_property_vector3");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_WasGoState_wrapper_set_stack_size(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, int p_stack_size){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_stack_size = (int) p_stack_size;
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_stack_size};
// 			Variant wasgo_ret = caller->call((String)"set_stack_size", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_stack_size");}
// 		}
// 	}
// 	return;
// }

// void _wasgo_WasGoState_wrapper_set_wasm_script(wasm_exec_env_t exec_env, WasGoState::WasGoID caller_id, WasGoState::WasGoID p_wasm_script){
// 	WasGoState *state = (WasGoState *)wasm_runtime_get_user_data(exec_env);
// 	if(state){
// 		WasGoState *caller = (WasGoState *) ((Object *) state->lookup_object(caller_id));
// 		if(caller){
// 			Variant v_wasm_script = state->lookup_object(p_wasm_script);
// 			Callable::CallError error;
// 			const Variant* varargs[] = {& v_wasm_script};
// 			Variant wasgo_ret = caller->call((String)"set_wasm_script", varargs, 1, error);
// 			if(error.error != Callable::CallError::CALL_OK){print_line("BAD WASGO CALL: set_wasm_script");}
// 		}
// 	}
// 	return;
// }
#endif // WASGO_NATIVE_WRAPPERS_H
