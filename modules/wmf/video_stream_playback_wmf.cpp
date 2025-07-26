/**************************************************************************/
/*  video_stream_playback_wmf.cpp                                         */
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

#include "video_stream_playback_wmf.h"

#include "core/string/print_string.h"
#include "audio_stream_playback_wmf.h"

VideoStreamPlaybackWMF::VideoStreamPlaybackWMF() {
	static int counter = 0;
	id = counter++;

	texture = Ref<ImageTexture>(memnew(ImageTexture));
	cache_frames.resize(24);
	
	// Create synchronizer for audio/video sync
	synchronizer = Ref<AudioVideoSynchronizer>(memnew(AudioVideoSynchronizer));
	synchronizer->set_sync_mode(AudioVideoSynchronizer::SYNC_MODE_AUDIO_MASTER);
	synchronizer->set_max_queue_size(6); // Allow up to 6 frames in queue
}

VideoStreamPlaybackWMF::~VideoStreamPlaybackWMF() {
	stop_decode_worker();
	shutdown_stream();
}

void VideoStreamPlaybackWMF::play() {
	if (!video_decoder) {
		print_line("VideoStreamPlaybackWMF: No video decoder available");
		return;
	}
	
	Error err = video_decoder->start_decoding();
	if (err != OK) {
		print_line("VideoStreamPlaybackWMF: Failed to start video decoder");
		return;
	}
	
	is_video_playing = true;
	is_video_paused = false;
	
	start_decode_worker();
	
	print_line("VideoStreamPlaybackWMF: Started video playback");
}

void VideoStreamPlaybackWMF::stop() {
	if (!is_video_playing) {
		return;
	}
	
	stop_decode_worker();
	
	if (video_decoder) {
		video_decoder->stop_decoding();
	}
	
	is_video_playing = false;
	is_video_paused = false;
	time = 0.0;
	
	clear_decoded_queue();
	
	print_line("VideoStreamPlaybackWMF: Stopped video playback");
}

bool VideoStreamPlaybackWMF::is_playing() const {
	return is_video_playing && !is_video_paused;
}

void VideoStreamPlaybackWMF::set_paused(bool p_paused) {
	if (is_video_paused == p_paused) {
		return;
	}
	
	is_video_paused = p_paused;
	
	if (video_decoder) {
		video_decoder->pause_decoding(p_paused);
	}
	
	print_line("VideoStreamPlaybackWMF: " + String(p_paused ? "Paused" : "Resumed") + " video playback");
}

bool VideoStreamPlaybackWMF::is_paused() const {
	return is_video_paused;
}

double VideoStreamPlaybackWMF::get_length() const {
	return stream_info.duration;
}

String VideoStreamPlaybackWMF::get_stream_name() const {
	return String("WMF Video Stream");
}

int VideoStreamPlaybackWMF::get_loop_count() const {
	return 0;
}

double VideoStreamPlaybackWMF::get_playback_position() const {
	return time;
}

void VideoStreamPlaybackWMF::seek(double p_time) {
	if (!video_decoder) {
		return;
	}
	
	time = p_time;
	video_decoder->seek_to_time(p_time);
	
	// Clear decoded frame queue after seek
	clear_decoded_queue();
	
	print_line("VideoStreamPlaybackWMF: Seeked to " + rtos(p_time) + " seconds");
}

void VideoStreamPlaybackWMF::set_video_decoder(WMFVideoDecoder *p_decoder) {
	MutexLock lock(video_mutex);
	video_decoder = p_decoder;
	
	if (video_decoder) {
		stream_info.size.x = video_decoder->get_width();
		stream_info.size.y = video_decoder->get_height();
		stream_info.fps = video_decoder->get_fps();
		stream_info.duration = 0.0f; // Will be set by container
		print_line("VideoStreamPlaybackWMF: Set video decoder - " + itos(stream_info.size.x) + "x" + itos(stream_info.size.y) + " @ " + rtos(stream_info.fps) + " FPS");
	}
}

void VideoStreamPlaybackWMF::set_audio_decoder(WMFAudioDecoder *p_decoder) {
	// Audio decoder is handled by AudioStreamPlaybackWMF
	// This method exists for interface compatibility
	print_line("VideoStreamPlaybackWMF: Audio decoder set (handled by AudioStreamPlaybackWMF)");
}

void VideoStreamPlaybackWMF::set_synchronizer(const Ref<AudioVideoSynchronizer> &p_synchronizer) {
	synchronizer = p_synchronizer;
	if (synchronizer.is_valid()) {
		print_line("VideoStreamPlaybackWMF: Synchronizer set");
	}
}

Ref<AudioVideoSynchronizer> VideoStreamPlaybackWMF::get_synchronizer() const {
	return synchronizer;
}

void VideoStreamPlaybackWMF::set_file(const String &p_file) {
	// This method is deprecated - decoders are set directly
	print_line("VideoStreamPlaybackWMF: set_file() is deprecated, use set_video_decoder() instead");
}

Ref<Texture2D> VideoStreamPlaybackWMF::get_texture() const {
	return texture;
}

void VideoStreamPlaybackWMF::update(double p_delta) {
	if (!is_playing()) {
		return;
	}
	
	// Use frame-rate aware time advancement
	double frame_duration = 1.0 / stream_info.fps;
	time += p_delta;
	
	// Process decoded frames with proper timing
	_process_decode_queue();
}

FrameData *VideoStreamPlaybackWMF::get_next_writable_frame() {
	return &cache_frames.write[write_frame_idx];
}

void VideoStreamPlaybackWMF::write_frame_done() {
	MutexLock lock(video_mutex);
	int next_write_frame_idx = (write_frame_idx + 1) % cache_frames.size();
	
	if (read_frame_idx == next_write_frame_idx) {
		read_frame_idx = (read_frame_idx + 1) % cache_frames.size();
	}
	
	write_frame_idx = next_write_frame_idx;
}

void VideoStreamPlaybackWMF::present() {
	// Update texture with current frame data
	if (!frame_data.is_empty() && texture.is_valid()) {
		// Convert frame data to image and update texture
		// This would need proper implementation based on the frame format
	}
}

int64_t VideoStreamPlaybackWMF::next_sample_time() {
	MutexLock lock(video_mutex);
	int64_t sample_time = INT64_MAX;
	if (!cache_frames.is_empty()) {
		sample_time = cache_frames[read_frame_idx].sample_time;
	}
	return sample_time;
}

void VideoStreamPlaybackWMF::shutdown_stream() {
	// Clean up any remaining resources
}

void VideoStreamPlaybackWMF::_decode_worker_thread(void *userdata) {
	VideoStreamPlaybackWMF *playback = static_cast<VideoStreamPlaybackWMF *>(userdata);
	playback->decode_thread_running = true;
	
	// Calculate frame timing for proper pacing
	double frame_duration_us = (1.0 / playback->stream_info.fps) * 1000000.0; // microseconds
	uint64_t last_frame_time = OS::get_singleton()->get_ticks_usec();
	
	while (playback->should_decode) {
		// Check if queue is full
		{
			MutexLock lock(playback->decoded_queue_mutex);
			if (playback->decoded_frame_queue.size() >= MAX_DECODED_FRAMES) {
				// Queue is full, wait for frames to be consumed
				OS::get_singleton()->delay_usec(frame_duration_us / 4); // Wait 1/4 frame duration
				continue;
			}
		}
		
		if (!playback->video_decoder || !playback->video_decoder->has_frame_available()) {
			// No frames available, wait based on frame rate
			OS::get_singleton()->delay_usec(frame_duration_us / 8); // Wait 1/8 frame duration
			continue;
		}
		
		// Get frame from decoder
		VideoFrame frame = playback->video_decoder->get_next_frame();
		if (frame.data.is_empty()) {
			continue;
		}
		
		// Convert to decoded frame
		DecodedFrame decoded_frame;
		decoded_frame.sample_time = frame.timestamp;
		decoded_frame.presentation_time = playback->video_decoder->get_frame_time(frame.timestamp);
		
		// Create texture from frame data
		if (frame.texture.is_valid()) {
			decoded_frame.texture = frame.texture;
		} else if (!frame.data.is_empty()) {
			// Create texture from raw frame data
			Ref<Image> img = Image::create_from_data(frame.width, frame.height, false, Image::FORMAT_RGBA8, frame.data);
			if (img.is_valid()) {
				Ref<ImageTexture> tex = memnew(ImageTexture);
				tex->set_image(img);
				decoded_frame.texture = tex;
			}
		}
		
		// Add to queue
		{
			MutexLock lock(playback->decoded_queue_mutex);
			playback->decoded_frame_queue.push_back(decoded_frame);
		}
		
		// Frame rate limiting - ensure we don't decode too fast
		uint64_t current_time = OS::get_singleton()->get_ticks_usec();
		uint64_t elapsed = current_time - last_frame_time;
		if (elapsed < frame_duration_us) {
			OS::get_singleton()->delay_usec(frame_duration_us - elapsed);
		}
		last_frame_time = OS::get_singleton()->get_ticks_usec();
	}
	
	playback->decode_thread_running = false;
}

void VideoStreamPlaybackWMF::_process_decode_queue() {
	if (!synchronizer.is_valid()) {
		return;
	}
	
	// Update video clock in synchronizer
	synchronizer->update_video_clock(time);
	
	// Feed decoded frames to synchronizer
	{
		MutexLock lock(decoded_queue_mutex);
		while (!decoded_frame_queue.empty()) {
			const DecodedFrame &frame = decoded_frame_queue.front();
			
			// Queue frame in synchronizer with proper presentation time and frame number
			synchronizer->queue_video_frame(frame.texture, frame.presentation_time, frame_counter++);
			
			decoded_frame_queue.pop_front();
		}
	}
	
	// Get current frame from synchronizer (handles all timing logic)
	Ref<Texture2D> current_frame = synchronizer->get_current_frame();
	if (current_frame.is_valid()) {
		texture = current_frame;
		frame_ready = true;
	}
}

void VideoStreamPlaybackWMF::start_decode_worker() {
	if (decode_thread_running) {
		return;
	}
	
	should_decode = true;
	decode_task_id = WorkerThreadPool::get_singleton()->add_native_task(&_decode_worker_thread, this, false, "WMF Video Decode");
}

void VideoStreamPlaybackWMF::stop_decode_worker() {
	if (!decode_thread_running) {
		return;
	}
	
	should_decode = false;
	
	// Wait for thread to finish
	if (decode_task_id != WorkerThreadPool::INVALID_TASK_ID) {
		WorkerThreadPool::get_singleton()->wait_for_task_completion(decode_task_id);
		decode_task_id = WorkerThreadPool::INVALID_TASK_ID;
	}
}

void VideoStreamPlaybackWMF::clear_decoded_queue() {
	MutexLock lock(decoded_queue_mutex);
	decoded_frame_queue.clear();
}
