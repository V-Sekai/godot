/**************************************************************************/
/*  playback_stats.h                                                      */
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

#ifndef PLAYBACK_STATS_H
#define PLAYBACK_STATS_H
#include "core/object/ref_counted.h"

class PlaybackStats : public RefCounted {
	GDCLASS(PlaybackStats, RefCounted);

protected:
	static void _bind_methods();

public:
	int64_t playback_ring_current_size = 0;
	int64_t playback_ring_max_size = 0;
	int64_t playback_ring_size_sum = 0;
	double playback_get_frames = 0.0;
	int64_t playback_pushed_calls = 0;
	int64_t playback_discarded_calls = 0;
	int64_t playback_push_buffer_calls = 0;
	int64_t playback_blank_push_calls = 0;
	double playback_position = 0.0;
	double playback_skips = 0.0;

	double jitter_buffer_size_sum = 0.0;
	int64_t jitter_buffer_calls = 0;
	int64_t jitter_buffer_max_size = 0;
	int64_t jitter_buffer_current_size = 0;

	int64_t playback_ring_buffer_length = 0;
	int64_t buffer_frame_count = 0;
	Dictionary get_playback_stats();
};

#endif // PLAYBACK_STATS_H
