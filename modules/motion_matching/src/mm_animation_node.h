/**************************************************************************/
/*  mm_animation_node.h                                                   */
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

#pragma once

#include "common.h"
#include "mm_animation_library.h"

#include "scene/animation/animation_tree.h"

#include <queue>

class MMAnimationNode : public AnimationNode {
	GDCLASS(MMAnimationNode, AnimationNode);

public:
	GETSET(StringName, library);
	GETSET(real_t, query_frequency, 2.0f)
	GETSET(real_t, transition_halflife, 0.1f)

	bool blending_enabled{ true };
	bool get_blending_enabled() const;

	void set_blending_enabled(bool value);

	virtual AnimationNode::NodeTimeInfo _process(const AnimationMixer::PlaybackInfo p_playback_info, bool p_test_only = false) override;

	virtual void get_parameter_list(List<PropertyInfo> *r_list) const override;
	virtual Variant get_parameter_default_value(const StringName &p_parameter) const override;
	virtual bool is_parameter_read_only(const StringName &p_parameter) const override;
	virtual String get_caption() const override;
	virtual bool has_filter() const override;

protected:
	void _validate_property(PropertyInfo &p_property) const;
	static void _bind_methods();

private:
	static Dictionary _output_to_dict(const MMQueryOutput &output);

	struct AnimationInfo {
		StringName name;
		double length;
		double time;
		double delta;
		bool seeked;
		bool is_external_seeking;
		real_t weight;
		real_t blend_spring_speed;
	};

	std::deque<AnimationInfo> _prev_animation_queue;
	AnimationInfo _current_animation_info;

	StringName _animation_key(const StringName &p_animation) const;
	void _start_transition(const StringName p_animation, float p_time);
	AnimationNode::NodeTimeInfo _update_current_animation(bool p_test_only);

	MMQueryOutput _last_query_output;
	float _time_since_last_query{ 0.f };
};
