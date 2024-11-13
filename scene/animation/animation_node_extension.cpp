/**************************************************************************/
/*  animation_node_extension.cpp                                          */
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

#include "animation_node_extension.h"

void NodeTimeInfoRC::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_length"), &NodeTimeInfoRC::get_length);
	ClassDB::bind_method(D_METHOD("set_length", "length"), &NodeTimeInfoRC::set_length);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "length"), "set_length", "get_length");

	ClassDB::bind_method(D_METHOD("get_position"), &NodeTimeInfoRC::get_position);
	ClassDB::bind_method(D_METHOD("set_position", "position"), &NodeTimeInfoRC::set_position);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "position"), "set_position", "get_position");

	ClassDB::bind_method(D_METHOD("get_delta"), &NodeTimeInfoRC::get_delta);
	ClassDB::bind_method(D_METHOD("set_delta", "delta"), &NodeTimeInfoRC::set_delta);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "delta"), "set_delta", "get_delta");

	ClassDB::bind_method(D_METHOD("get_loop_mode"), &NodeTimeInfoRC::get_loop_mode);
	ClassDB::bind_method(D_METHOD("set_loop_mode", "loop_mode"), &NodeTimeInfoRC::set_loop_mode);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "loop_mode", PROPERTY_HINT_ENUM, "None,Linear,PingPong"), "set_loop_mode", "get_loop_mode");

	ClassDB::bind_method(D_METHOD("get_will_end"), &NodeTimeInfoRC::get_will_end);
	ClassDB::bind_method(D_METHOD("set_will_end", "will_end"), &NodeTimeInfoRC::set_will_end);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "will_end"), "set_will_end", "get_will_end");

	ClassDB::bind_method(D_METHOD("get_is_infinity"), &NodeTimeInfoRC::get_is_infinity);
	ClassDB::bind_method(D_METHOD("set_is_infinity", "is_infinity"), &NodeTimeInfoRC::set_is_infinity);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_infinity"), "set_is_infinity", "get_is_infinity");
}

void PlaybackInfoRC::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_time"), &PlaybackInfoRC::get_time);
	ClassDB::bind_method(D_METHOD("set_time", "time"), &PlaybackInfoRC::set_time);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "time"), "set_time", "get_time");

	ClassDB::bind_method(D_METHOD("get_delta"), &PlaybackInfoRC::get_delta);
	ClassDB::bind_method(D_METHOD("set_delta", "delta"), &PlaybackInfoRC::set_delta);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "delta"), "set_delta", "get_delta");

	ClassDB::bind_method(D_METHOD("get_start"), &PlaybackInfoRC::get_start);
	ClassDB::bind_method(D_METHOD("set_start", "start"), &PlaybackInfoRC::set_start);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "start"), "set_start", "get_start");

	ClassDB::bind_method(D_METHOD("get_end"), &PlaybackInfoRC::get_end);
	ClassDB::bind_method(D_METHOD("set_end", "end"), &PlaybackInfoRC::set_end);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "end"), "set_end", "get_end");

	ClassDB::bind_method(D_METHOD("get_seeked"), &PlaybackInfoRC::get_seeked);
	ClassDB::bind_method(D_METHOD("set_seeked", "seeked"), &PlaybackInfoRC::set_seeked);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "seeked"), "set_seeked", "get_seeked");

	ClassDB::bind_method(D_METHOD("get_is_external_seeking"), &PlaybackInfoRC::get_is_external_seeking);
	ClassDB::bind_method(D_METHOD("set_is_external_seeking", "is_external_seeking"), &PlaybackInfoRC::set_is_external_seeking);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_external_seeking"), "set_is_external_seeking", "get_is_external_seeking");

	ClassDB::bind_method(D_METHOD("get_looped_flag"), &PlaybackInfoRC::get_looped_flag);
	ClassDB::bind_method(D_METHOD("set_looped_flag", "looped_flag"), &PlaybackInfoRC::set_looped_flag);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "looped_flag", PROPERTY_HINT_ENUM, "None,End,Start"), "set_looped_flag", "get_looped_flag");

	ClassDB::bind_method(D_METHOD("get_weight"), &PlaybackInfoRC::get_weight);
	ClassDB::bind_method(D_METHOD("set_weight", "weight"), &PlaybackInfoRC::set_weight);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "weight"), "set_weight", "get_weight");
}

AnimationNodeExtension::AnimationNodeExtension() {
	_node_time_info.instantiate();
	_playback_info.instantiate();
}

AnimationNode::NodeTimeInfo AnimationNodeExtension::_process(const AnimationMixer::PlaybackInfo p_playback_info, bool p_test_only) {
	_node_time_info->_reset();
	_playback_info->_reset(p_playback_info);

	GDVIRTUAL_CALL(_process, _playback_info, _node_time_info, p_test_only);

	return _node_time_info->_get_node_time_info();
}

AnimationTree *AnimationNodeExtension::get_process_tree() const {
	ERR_FAIL_NULL_V(process_state, nullptr);
	return process_state->tree;
}

void AnimationNodeExtension::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_process_tree"), &AnimationNodeExtension::get_process_tree);
	GDVIRTUAL_BIND(_process, "playback_info", "node_time_info", "test_only");
}
