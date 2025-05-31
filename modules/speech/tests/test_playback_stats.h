/**************************************************************************/
/*  test_playback_stats.h                                                 */
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

#include "../playback_stats.h"
#include "core/object/ref_counted.h"
#include "core/variant/dictionary.h"
#include "scene/main/scene_tree.h" // Added for SceneTree
#include "scene/main/window.h" // Added for Window (root of SceneTree)
#include "tests/test_macros.h"

namespace TestPlaybackStats {

TEST_CASE("[SceneTree][PlaybackStats] Initialization and Getters") {
	SceneTree *tree = SceneTree::get_singleton();
	Window *original_root_window = tree->get_root();
	Window *temp_root_window = nullptr;

	if (!original_root_window) {
		temp_root_window = memnew(Window);
		tree->add_current_scene(temp_root_window);
	}

	Ref<PlaybackStats> stats;
	stats.instantiate();
	REQUIRE(stats.is_valid());

	CHECK(stats->playback_ring_current_size == 0);
	CHECK(stats->playback_ring_max_size == 0);
	CHECK(stats->playback_ring_size_sum == 0);
	CHECK(Math::is_zero_approx(stats->playback_get_frames));
	CHECK(stats->playback_pushed_calls == 0);
	CHECK(stats->playback_discarded_calls == 0);
	CHECK(stats->playback_push_buffer_calls == 0);
	CHECK(stats->playback_blank_push_calls == 0);
	CHECK(Math::is_zero_approx(stats->playback_position));
	CHECK(Math::is_zero_approx(stats->playback_skips));
	CHECK(Math::is_zero_approx(stats->jitter_buffer_size_sum));
	CHECK(stats->jitter_buffer_calls == 0);
	CHECK(stats->jitter_buffer_max_size == 0);
	CHECK(stats->jitter_buffer_current_size == 0);
	CHECK(stats->playback_ring_buffer_length == 0);
	CHECK(stats->buffer_frame_count == 0);
	if (temp_root_window) {
		tree->queue_delete(temp_root_window);
	}
}

TEST_CASE("[SceneTree][PlaybackStats] Stats Modification and Retrieval") {
	Ref<PlaybackStats> stats;
	stats.instantiate();
	REQUIRE(stats.is_valid());

	stats->playback_ring_current_size = 100;
	stats->jitter_buffer_max_size = 5;
	stats->playback_skips = 10.5;

	CHECK(stats->playback_ring_current_size == 100);
	CHECK(stats->jitter_buffer_max_size == 5);
	CHECK(Math::is_equal_approx(stats->playback_skips, 10.5));
}

} // namespace TestPlaybackStats
