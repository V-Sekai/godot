/**************************************************************************/
/*  animation_node_extension.h                                            */
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

#ifndef ANIMATION_NODE_EXTENSION_H
#define ANIMATION_NODE_EXTENSION_H

#include "scene/animation/animation_tree.h"

class NodeTimeInfoRC : public RefCounted {
	GDCLASS(NodeTimeInfoRC, RefCounted);

public:
	double get_length() const {
		return node_time_info.length;
	}
	void set_length(double p_length) {
		node_time_info.length = p_length;
	}

	double get_position() const {
		return node_time_info.position;
	}
	void set_position(double p_position) {
		node_time_info.position = p_position;
	}

	double get_delta() const {
		return node_time_info.delta;
	}
	void set_delta(double p_delta) {
		node_time_info.delta = p_delta;
	}

	Animation::LoopMode get_loop_mode() const {
		return node_time_info.loop_mode;
	}
	void set_loop_mode(Animation::LoopMode p_loop_mode) {
		node_time_info.loop_mode = p_loop_mode;
	}

	bool get_will_end() const {
		return node_time_info.will_end;
	}
	void set_will_end(bool p_will_end) {
		node_time_info.will_end = p_will_end;
	}

	bool get_is_infinity() const {
		return node_time_info.is_infinity;
	}
	void set_is_infinity(bool p_is_infinity) {
		node_time_info.is_infinity = p_is_infinity;
	}

	void _reset() {
		node_time_info = AnimationNode::NodeTimeInfo();
	}

	const AnimationNode::NodeTimeInfo &_get_node_time_info() const {
		return node_time_info;
	}

protected:
	static void _bind_methods();

private:
	AnimationNode::NodeTimeInfo node_time_info;
};

class PlaybackInfoRC : public RefCounted {
	GDCLASS(PlaybackInfoRC, RefCounted);

public:
	double get_time() const {
		return playback_info.time;
	}
	void set_time(double p_time) {
		playback_info.time = p_time;
	}

	double get_delta() const {
		return playback_info.delta;
	}
	void set_delta(double p_delta) {
		playback_info.delta = p_delta;
	}

	double get_start() const {
		return playback_info.start;
	}
	void set_start(double p_start) {
		playback_info.start = p_start;
	}

	double get_end() const {
		return playback_info.end;
	}
	void set_end(double p_end) {
		playback_info.end = p_end;
	}

	bool get_seeked() const {
		return playback_info.seeked;
	}
	void set_seeked(bool p_seeked) {
		playback_info.seeked = p_seeked;
	}

	bool get_is_external_seeking() const {
		return playback_info.is_external_seeking;
	}
	void set_is_external_seeking(bool p_is_external_seeking) {
		playback_info.is_external_seeking = p_is_external_seeking;
	}

	Animation::LoopedFlag get_looped_flag() const {
		return playback_info.looped_flag;
	}
	void set_looped_flag(Animation::LoopedFlag p_looped_flag) {
		playback_info.looped_flag = p_looped_flag;
	}

	real_t get_weight() const {
		return playback_info.weight;
	}
	void set_weight(real_t p_weight) {
		playback_info.weight = p_weight;
	}

	void _reset(const AnimationMixer::PlaybackInfo &p_playback_info) {
		playback_info = p_playback_info;
	}

protected:
	static void _bind_methods();

private:
	AnimationMixer::PlaybackInfo playback_info;
};

class AnimationNodeExtension : public AnimationRootNode {
	GDCLASS(AnimationNodeExtension, AnimationRootNode);

public:
	AnimationNodeExtension();
	virtual ~AnimationNodeExtension() = default;

	virtual NodeTimeInfo _process(const AnimationMixer::PlaybackInfo p_playback_info, bool p_test_only = false) override;

	AnimationTree *get_process_tree() const;

protected:
	static void _bind_methods();

	GDVIRTUAL3(_process, Ref<PlaybackInfoRC>, Ref<NodeTimeInfoRC>, bool);

private:
	Ref<NodeTimeInfoRC> _node_time_info;
	Ref<PlaybackInfoRC> _playback_info;
};

#endif // ANIMATION_NODE_EXTENSION_H
