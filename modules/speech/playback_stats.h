#pragma once
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
