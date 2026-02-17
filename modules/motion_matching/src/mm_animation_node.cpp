/**************************************************************************/
/*  mm_animation_node.cpp                                                 */
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

#include "mm_animation_node.h"

#include "math/spring.hpp"
#include "mm_query.h"

#include "core/config/engine.h"
#include "scene/resources/animation.h"

#ifdef TOOLS_ENABLED
#include "editor/animation_tree_handler_plugin.h"
#endif

// Only play the matched animation if the matched time position
// is QUERY_TIME_ERROR away from the current time
constexpr float QUERY_TIME_ERROR = 0.05;

AnimationNode::NodeTimeInfo MMAnimationNode::_process(const AnimationMixer::PlaybackInfo p_playback_info, bool p_test_only) {
	AnimationNode::NodeTimeInfo default_result;
	default_result.length = 0.0;
	default_result.position = 0.0;
	default_result.delta = 0.0;
	default_result.loop_mode = Animation::LOOP_NONE;
	default_result.will_end = false;
	default_result.is_infinity = false;

	if (Engine::get_singleton()->is_editor_hint()) {
		return default_result;
	}

	if (library.is_empty()) {
		return default_result;
	}

	const double time = p_playback_info.time;
	const double delta_time = p_playback_info.delta;
	_current_animation_info.time = time;
	_current_animation_info.delta = delta_time;
	_current_animation_info.seeked = p_playback_info.seeked;
	_current_animation_info.is_external_seeking = p_playback_info.is_external_seeking;

	const bool is_about_to_end = false; // TODO: Implement this

	// We run queries periodically, or when the animation is about to end
	const bool has_current_animation = !_last_query_output.animation_match.is_empty();
	const bool should_query = (_time_since_last_query > (1.0 / query_frequency)) || is_about_to_end || !has_current_animation;

	if (!should_query) {
		_time_since_last_query += delta_time;
		return _update_current_animation(p_test_only);
	}

	MMQueryInput *query_input = Object::cast_to<MMQueryInput>(get_parameter("motion_matching_input"));

	if (!query_input || !query_input->is_valid()) {
		_time_since_last_query += delta_time;
		return _update_current_animation(p_test_only);
	}

	_time_since_last_query = 0.f;

	// Run query
	AnimationTree *animation_tree = Object::cast_to<AnimationTree>(ObjectDB::get_instance(get_processing_animation_tree_instance_id()));
	Ref<MMAnimationLibrary> animation_library = animation_tree->get_animation_library(library);
	ERR_FAIL_COND_V_MSG(animation_library.is_null(), default_result, "Library not found: " + library);
	ERR_FAIL_COND_V_MSG(
			animation_library->db_anim_index.is_empty() || animation_library->db_time_index.is_empty(),
			default_result,
			"Library not baked: " + library);
	const MMQueryOutput query_output = animation_library->query(*query_input);

	const bool is_same_animation = query_output.animation_match == _last_query_output.animation_match;
	const bool is_same_time = Math::abs(query_output.time_match - time) < QUERY_TIME_ERROR;

	// Play selected animation
	if (!is_same_animation || !is_same_time) {
		const String animation_match = query_output.animation_match;
		const float time_match = query_output.time_match;
		if (!p_test_only) {
			_start_transition(animation_match, time_match);
		}
		_last_query_output = query_output;
		query_input->on_query_result(query_output);
	}

	return _update_current_animation(p_test_only);
}

StringName MMAnimationNode::_animation_key(const StringName &p_animation) const {
	if (library.is_empty()) {
		return p_animation;
	}
	return StringName(String(library) + "/" + String(p_animation));
}

void MMAnimationNode::_start_transition(const StringName p_animation, float p_time) {
	AnimationTree *animation_tree = Object::cast_to<AnimationTree>(ObjectDB::get_instance(get_processing_animation_tree_instance_id()));
	const StringName key = _animation_key(p_animation);
	Ref<Animation> anim = animation_tree->get_animation(key);
	ERR_FAIL_COND_MSG(anim.is_null(), vformat("Animation not found: %s", p_animation));

	if (!_current_animation_info.name.is_empty() && blending_enabled) {
		_prev_animation_queue.push_front(_current_animation_info);
	}

	_current_animation_info.name = key;
	_current_animation_info.length = anim->get_length();
	_current_animation_info.time = p_time;
	_current_animation_info.weight = blending_enabled ? 0.f : 1.f;
}

AnimationNode::NodeTimeInfo MMAnimationNode::_update_current_animation(bool p_test_only) {
	Spring::_simple_spring_damper_exact(
			_current_animation_info.weight,
			_current_animation_info.blend_spring_speed,
			1.,
			transition_halflife,
			(real_t)_current_animation_info.delta);

	int pop_count = 0;
	for (AnimationInfo &prev_info : _prev_animation_queue) {
		Spring::_simple_spring_damper_exact(
				prev_info.weight,
				prev_info.blend_spring_speed,
				0.f,
				transition_halflife,
				_current_animation_info.delta);
		if (prev_info.weight <= SMALL_NUMBER) {
			pop_count++;
		}
	}

	for (int i = 0; i < pop_count; i++) {
		_prev_animation_queue.pop_back();
	}

	// Normalized blend weights in the queue
	const float inv_blend = 1.f - _current_animation_info.weight;
	float prev_blend_total = 0.f;
	for (AnimationInfo &prev_info : _prev_animation_queue) {
		prev_blend_total += prev_info.weight;
	}

	for (AnimationInfo &prev_info : _prev_animation_queue) {
		prev_info.weight *= inv_blend / prev_blend_total;
	}

	if (!p_test_only) {
		for (AnimationInfo &prev_info : _prev_animation_queue) {
			AnimationMixer::PlaybackInfo info;
			info.time = prev_info.time;
			info.delta = prev_info.delta;
			info.seeked = prev_info.seeked;
			info.is_external_seeking = prev_info.is_external_seeking;
			info.weight = prev_info.weight;
			blend_animation(prev_info.name, info);
		}
		AnimationMixer::PlaybackInfo info;
		info.time = _current_animation_info.time;
		info.delta = _current_animation_info.delta;
		info.seeked = _current_animation_info.seeked;
		info.is_external_seeking = _current_animation_info.is_external_seeking;
		info.weight = _current_animation_info.weight;
		blend_animation(_current_animation_info.name, info);
	}

	AnimationNode::NodeTimeInfo result;
	result.length = 0.0;
	result.position = _current_animation_info.time;
	result.delta = _current_animation_info.delta;
	result.loop_mode = Animation::LOOP_NONE;
	result.will_end = false;
	result.is_infinity = false;
	return result;
}

void MMAnimationNode::get_parameter_list(List<PropertyInfo> *r_list) const {
	r_list->push_back(PropertyInfo(Variant::OBJECT, "motion_matching_input", PROPERTY_HINT_RESOURCE_TYPE, "MMQueryInput", PROPERTY_USAGE_STORAGE));
}

Variant MMAnimationNode::get_parameter_default_value(const StringName &p_parameter) const {
	if (p_parameter == StringName("motion_matching_input")) {
		Ref<MMQueryInput> p;
		p.instantiate();
		return p;
	}
	return AnimationNode::get_parameter_default_value(p_parameter);
}

bool MMAnimationNode::is_parameter_read_only(const StringName &p_parameter) const {
	if (p_parameter == StringName("motion_matching_input")) {
		return false;
	}
	return AnimationNode::is_parameter_read_only(p_parameter);
}

String MMAnimationNode::get_caption() const {
	return "Motion Matching";
}

bool MMAnimationNode::has_filter() const {
	return true;
}

void MMAnimationNode::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == StringName("transition_halflife")) {
		if (!blending_enabled) {
			p_property.usage = PROPERTY_USAGE_NO_EDITOR;
		}
	}

#ifdef TOOLS_ENABLED
	if (p_property.name == StringName("library")) {
		AnimationTreeHandlerPlugin *plugin = AnimationTreeHandlerPlugin::get_singleton();
		if (plugin) {
			AnimationTree *tree = plugin->get_animation_tree();
			if (tree) {
				List<StringName> library_names;
				tree->get_animation_library_list(&library_names);
				String animations;
				for (const StringName &lib_name : library_names) {
					Ref<MMAnimationLibrary> lib = tree->get_animation_library(lib_name);
					if (lib.is_null()) {
						continue;
					}
					if (!animations.is_empty()) {
						animations += ",";
					}
					animations += (String)lib_name;
				}
				if (!animations.is_empty()) {
					p_property.hint = PROPERTY_HINT_ENUM;
					p_property.hint_string = animations;
				}
			}
		}
	}
#endif

	AnimationNode::_validate_property(p_property);
}

void MMAnimationNode::_bind_methods() {
	BINDER_PROPERTY_PARAMS(MMAnimationNode, Variant::STRING_NAME, library);
	BINDER_PROPERTY_PARAMS(MMAnimationNode, Variant::FLOAT, query_frequency);
	ClassDB::bind_method(D_METHOD("get_blending_enabled"), &MMAnimationNode::get_blending_enabled);
	ClassDB::bind_method(D_METHOD("set_blending_enabled", "value"), &MMAnimationNode::set_blending_enabled);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "blending_enabled"), "set_blending_enabled", "get_blending_enabled");

	BINDER_PROPERTY_PARAMS(MMAnimationNode, Variant::FLOAT, transition_halflife);
}

Dictionary MMAnimationNode::_output_to_dict(const MMQueryOutput &output) {
	Dictionary result;

	result.get_or_add("animation", output.animation_match);
	result.get_or_add("time", output.time_match);
	result.get_or_add("frame_data", output.matched_frame_data);
	result.merge(output.feature_costs);
	result.get_or_add("total_cost", output.cost);

	return result;
}

bool MMAnimationNode::get_blending_enabled() const {
	return blending_enabled;
}

void MMAnimationNode::set_blending_enabled(bool value) {
	blending_enabled = value;
	notify_property_list_changed();
}
