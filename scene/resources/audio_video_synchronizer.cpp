/**************************************************************************/
/*  audio_video_synchronizer.cpp                                          */
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

#include "audio_video_synchronizer.h"

#include "core/math/math_funcs.h"

AudioVideoSynchronizer::AudioVideoSynchronizer() {
	timing_filter.instantiate();
	timing_filter->update_parameters(0.1, 5.0); // Default OneEuro parameters
}

AudioVideoSynchronizer::~AudioVideoSynchronizer() {
}

void AudioVideoSynchronizer::set_sync_mode(SyncMode p_mode) {
	sync_mode = p_mode;
}

AudioVideoSynchronizer::SyncMode AudioVideoSynchronizer::get_sync_mode() const {
	return sync_mode;
}

void AudioVideoSynchronizer::set_max_queue_size(int p_size) {
	max_queue_size = MAX(1, p_size);
	_cleanup_old_frames();
}

int AudioVideoSynchronizer::get_max_queue_size() const {
	return max_queue_size;
}

void AudioVideoSynchronizer::set_use_timing_filter(bool p_enable) {
	use_timing_filter = p_enable;
	if (!p_enable && timing_filter.is_valid()) {
		timing_filter->reset();
	}
}

bool AudioVideoSynchronizer::get_use_timing_filter() const {
	return use_timing_filter;
}

void AudioVideoSynchronizer::set_sync_threshold(double p_threshold) {
	sync_threshold = MAX(0.001, p_threshold);
}

double AudioVideoSynchronizer::get_sync_threshold() const {
	return sync_threshold;
}

void AudioVideoSynchronizer::update_master_clock(double p_time) {
	master_clock_time = p_time;
}

void AudioVideoSynchronizer::update_audio_clock(double p_time) {
	audio_clock_time = p_time;
	if (sync_mode == SYNC_MODE_AUDIO_MASTER) {
		master_clock_time = p_time;
	}
}

void AudioVideoSynchronizer::update_video_clock(double p_time) {
	video_clock_time = p_time;
	if (sync_mode == SYNC_MODE_VIDEO_MASTER) {
		master_clock_time = p_time;
	}
}

double AudioVideoSynchronizer::get_master_clock_time() const {
	return master_clock_time;
}

double AudioVideoSynchronizer::get_sync_error() const {
	return video_clock_time - master_clock_time;
}

void AudioVideoSynchronizer::queue_video_frame(const Ref<Texture2D> &p_texture, double p_presentation_time, uint64_t p_frame_number) {
	FrameInfo frame;
	frame.texture = p_texture;
	frame.presentation_time = p_presentation_time;
	frame.frame_number = p_frame_number;
	frame.displayed = false;

	frame_queue.push_back(frame);
	_cleanup_old_frames();
}

Ref<Texture2D> AudioVideoSynchronizer::get_current_frame() {
	if (frame_queue.is_empty()) {
		return Ref<Texture2D>();
	}

	double target_time = master_clock_time;
	
	// Apply timing filter if enabled
	if (use_timing_filter && timing_filter.is_valid()) {
		static double last_time = 0.0;
		double delta = target_time - last_time;
		if (delta > 0.0) {
			target_time = timing_filter->filter(target_time, delta);
		}
		last_time = target_time;
	}

	int best_frame_index = _find_best_frame_for_time(target_time);
	
	if (best_frame_index == -1) {
		// No suitable frame found
		return Ref<Texture2D>();
	}

	// Calculate sync error for statistics
	double sync_error = frame_queue[best_frame_index].presentation_time - target_time;
	_update_sync_statistics(sync_error);

	// Handle frame dropping/repeating logic
	if (Math::abs(sync_error) > max_frame_drop_threshold) {
		// Frame is too far off, drop it
		frames_dropped++;
		frame_queue.remove_at(best_frame_index);
		return get_current_frame(); // Recursively try next frame
	}

	// Mark frame as displayed and return texture
	frame_queue.write[best_frame_index].displayed = true;
	current_frame_index = best_frame_index;
	
	return frame_queue[best_frame_index].texture;
}

void AudioVideoSynchronizer::clear_frame_queue() {
	frame_queue.clear();
	current_frame_index = -1;
}

void AudioVideoSynchronizer::reset() {
	master_clock_time = 0.0;
	video_clock_time = 0.0;
	audio_clock_time = 0.0;
	
	clear_frame_queue();
	
	frames_dropped = 0;
	frames_repeated = 0;
	average_sync_error = 0.0;
	
	if (timing_filter.is_valid()) {
		timing_filter->reset();
	}
}

void AudioVideoSynchronizer::_cleanup_old_frames() {
	// Remove frames that exceed queue size
	while (frame_queue.size() > max_queue_size) {
		frame_queue.remove_at(0);
		if (current_frame_index > 0) {
			current_frame_index--;
		} else {
			current_frame_index = -1;
		}
	}

	// Remove old displayed frames
	for (int i = frame_queue.size() - 1; i >= 0; i--) {
		if (frame_queue[i].displayed && 
			frame_queue[i].presentation_time < master_clock_time - sync_threshold) {
			frame_queue.remove_at(i);
			if (current_frame_index >= i) {
				current_frame_index--;
			}
		}
	}
}

int AudioVideoSynchronizer::_find_best_frame_for_time(double target_time) const {
	if (frame_queue.is_empty()) {
		return -1;
	}

	int best_index = -1;
	double best_error = INFINITY;

	for (int i = 0; i < frame_queue.size(); i++) {
		if (frame_queue[i].displayed) {
			continue; // Skip already displayed frames
		}

		double error = Math::abs(frame_queue[i].presentation_time - target_time);
		if (error < best_error) {
			best_error = error;
			best_index = i;
		}
	}

	return best_index;
}

void AudioVideoSynchronizer::_update_sync_statistics(double sync_error) {
	// Simple moving average for sync error
	static const double alpha = 0.1;
	average_sync_error = alpha * Math::abs(sync_error) + (1.0 - alpha) * average_sync_error;
}

void AudioVideoSynchronizer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_sync_mode", "mode"), &AudioVideoSynchronizer::set_sync_mode);
	ClassDB::bind_method(D_METHOD("get_sync_mode"), &AudioVideoSynchronizer::get_sync_mode);
	
	ClassDB::bind_method(D_METHOD("set_max_queue_size", "size"), &AudioVideoSynchronizer::set_max_queue_size);
	ClassDB::bind_method(D_METHOD("get_max_queue_size"), &AudioVideoSynchronizer::get_max_queue_size);
	
	ClassDB::bind_method(D_METHOD("set_use_timing_filter", "enable"), &AudioVideoSynchronizer::set_use_timing_filter);
	ClassDB::bind_method(D_METHOD("get_use_timing_filter"), &AudioVideoSynchronizer::get_use_timing_filter);
	
	ClassDB::bind_method(D_METHOD("set_sync_threshold", "threshold"), &AudioVideoSynchronizer::set_sync_threshold);
	ClassDB::bind_method(D_METHOD("get_sync_threshold"), &AudioVideoSynchronizer::get_sync_threshold);
	
	ClassDB::bind_method(D_METHOD("update_master_clock", "time"), &AudioVideoSynchronizer::update_master_clock);
	ClassDB::bind_method(D_METHOD("update_audio_clock", "time"), &AudioVideoSynchronizer::update_audio_clock);
	ClassDB::bind_method(D_METHOD("update_video_clock", "time"), &AudioVideoSynchronizer::update_video_clock);
	
	ClassDB::bind_method(D_METHOD("get_master_clock_time"), &AudioVideoSynchronizer::get_master_clock_time);
	ClassDB::bind_method(D_METHOD("get_sync_error"), &AudioVideoSynchronizer::get_sync_error);
	
	ClassDB::bind_method(D_METHOD("queue_video_frame", "texture", "presentation_time", "frame_number"), &AudioVideoSynchronizer::queue_video_frame, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_current_frame"), &AudioVideoSynchronizer::get_current_frame);
	ClassDB::bind_method(D_METHOD("clear_frame_queue"), &AudioVideoSynchronizer::clear_frame_queue);
	
	ClassDB::bind_method(D_METHOD("get_frames_dropped"), &AudioVideoSynchronizer::get_frames_dropped);
	ClassDB::bind_method(D_METHOD("get_frames_repeated"), &AudioVideoSynchronizer::get_frames_repeated);
	ClassDB::bind_method(D_METHOD("get_average_sync_error"), &AudioVideoSynchronizer::get_average_sync_error);
	ClassDB::bind_method(D_METHOD("get_queued_frame_count"), &AudioVideoSynchronizer::get_queued_frame_count);
	
	ClassDB::bind_method(D_METHOD("reset"), &AudioVideoSynchronizer::reset);

	BIND_ENUM_CONSTANT(SYNC_MODE_AUDIO_MASTER);
	BIND_ENUM_CONSTANT(SYNC_MODE_VIDEO_MASTER);
	BIND_ENUM_CONSTANT(SYNC_MODE_EXTERNAL);
}
