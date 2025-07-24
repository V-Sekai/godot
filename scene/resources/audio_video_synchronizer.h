/**************************************************************************/
/*  audio_video_synchronizer.h                                            */
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

#include "core/object/ref_counted.h"
#include "scene/resources/one_euro_filter.h"
#include "scene/resources/texture.h"

class AudioVideoSynchronizer : public RefCounted {
	GDCLASS(AudioVideoSynchronizer, RefCounted);

public:
	enum SyncMode {
		SYNC_MODE_AUDIO_MASTER,  // Audio is master clock (default)
		SYNC_MODE_VIDEO_MASTER,  // Video is master clock
		SYNC_MODE_EXTERNAL       // External clock source
	};

private:
	struct FrameInfo {
		Ref<Texture2D> texture;
		double presentation_time = 0.0;
		uint64_t frame_number = 0;
		bool displayed = false;
	};

	// Synchronization state
	SyncMode sync_mode = SYNC_MODE_AUDIO_MASTER;
	double master_clock_time = 0.0;
	double video_clock_time = 0.0;
	double audio_clock_time = 0.0;
	
	// Frame management
	Vector<FrameInfo> frame_queue;
	int max_queue_size = 3;
	int current_frame_index = -1;
	
	// Timing filters
	Ref<OneEuroFilter> timing_filter;
	bool use_timing_filter = true;
	
	// Synchronization parameters
	double sync_threshold = 0.040; // 40ms threshold for sync correction
	double max_frame_drop_threshold = 0.100; // 100ms threshold for dropping frames
	double max_frame_repeat_threshold = 0.020; // 20ms threshold for repeating frames
	
	// Statistics
	uint64_t frames_dropped = 0;
	uint64_t frames_repeated = 0;
	double average_sync_error = 0.0;
	
	// Internal methods
	void _cleanup_old_frames();
	int _find_best_frame_for_time(double target_time) const;
	void _update_sync_statistics(double sync_error);

protected:
	static void _bind_methods();

public:
	AudioVideoSynchronizer();
	virtual ~AudioVideoSynchronizer();

	// Configuration
	void set_sync_mode(SyncMode p_mode);
	SyncMode get_sync_mode() const;
	
	void set_max_queue_size(int p_size);
	int get_max_queue_size() const;
	
	void set_use_timing_filter(bool p_enable);
	bool get_use_timing_filter() const;
	
	void set_sync_threshold(double p_threshold);
	double get_sync_threshold() const;
	
	// Clock management
	void update_master_clock(double p_time);
	void update_audio_clock(double p_time);
	void update_video_clock(double p_time);
	
	double get_master_clock_time() const;
	double get_sync_error() const;
	
	// Frame management
	void queue_video_frame(const Ref<Texture2D> &p_texture, double p_presentation_time, uint64_t p_frame_number = 0);
	Ref<Texture2D> get_current_frame();
	void clear_frame_queue();
	
	// Statistics
	uint64_t get_frames_dropped() const { return frames_dropped; }
	uint64_t get_frames_repeated() const { return frames_repeated; }
	double get_average_sync_error() const { return average_sync_error; }
	int get_queued_frame_count() const { return frame_queue.size(); }
	
	// Reset
	void reset();
};

VARIANT_ENUM_CAST(AudioVideoSynchronizer::SyncMode);
