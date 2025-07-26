/**************************************************************************/
/*  video_stream_playback_wmf.h                                           */
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

#include "scene/resources/video_stream.h"
#include "core/object/worker_thread_pool.h"
#include "core/os/mutex.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/audio_video_synchronizer.h"

#include <atomic>
#include <deque>

// Forward declarations
class WMFVideoDecoder;
class WMFAudioDecoder;
class SampleGrabberCallback;

struct FrameData {
	int64_t sample_time = 0;
	Vector<uint8_t> data;
};

struct DecodedFrame {
	int64_t sample_time = 0;
	Ref<ImageTexture> texture;
	double presentation_time = 0.0;
};

class VideoStreamPlaybackWMF : public VideoStreamPlayback {
	GDCLASS(VideoStreamPlaybackWMF, VideoStreamPlayback);

private:
	WMFVideoDecoder *video_decoder = nullptr;
	Ref<AudioVideoSynchronizer> synchronizer;
	
	Vector<FrameData> cache_frames;
	int read_frame_idx = 0;
	int write_frame_idx = 0;

	Vector<uint8_t> frame_data;
	Ref<ImageTexture> texture;
	
	// Video processing mutex
	Mutex video_mutex;

	bool is_video_playing = false;
	bool is_video_paused = false;
	bool is_video_seekable = false;

	double time = 0.0;
	double next_frame_time = 0.0;
	double current_frame_time = -1.0;
	bool frame_ready = false;
	uint64_t frame_counter = 0;

	int id = 0;

	// Worker thread infrastructure
	WorkerThreadPool::TaskID decode_task_id = WorkerThreadPool::INVALID_TASK_ID;
	std::atomic<bool> should_decode{false};
	std::atomic<bool> decode_thread_running{false};
	std::atomic<double> target_decode_time{0.0};
	
	// Decoded frame queue for async processing (now feeds synchronizer)
	Mutex decoded_queue_mutex;
	std::deque<DecodedFrame> decoded_frame_queue;
	static constexpr int MAX_DECODED_FRAMES = 6;
	static constexpr double DECODE_LOOKAHEAD = 0.2; // 200ms lookahead

	void shutdown_stream();
	
	// Worker thread methods
	static void _decode_worker_thread(void *userdata);
	void _process_decode_queue();
	void start_decode_worker();
	void stop_decode_worker();
	void clear_decoded_queue();

public:
	struct StreamInfo {
		Point2i size;
		float fps = 0.0f;
		float duration = 0.0f;
	};
	StreamInfo stream_info;

	// VideoStreamPlayback interface
	virtual void play() override;
	virtual void stop() override;
	virtual bool is_playing() const override;

	virtual void set_paused(bool p_paused) override;
	virtual bool is_paused() const override;

	virtual double get_length() const override;
	virtual String get_stream_name() const;

	virtual int get_loop_count() const;

	virtual double get_playback_position() const override;
	virtual void seek(double p_time) override;

	virtual Ref<Texture2D> get_texture() const override;
	virtual void update(double p_delta) override;

	// WMF specific
	void set_video_decoder(WMFVideoDecoder *p_decoder);
	void set_audio_decoder(WMFAudioDecoder *p_decoder);
	void set_file(const String &p_file);

	// Synchronizer integration
	void set_synchronizer(const Ref<AudioVideoSynchronizer> &p_synchronizer);
	Ref<AudioVideoSynchronizer> get_synchronizer() const;

	FrameData *get_next_writable_frame();
	void write_frame_done();
	void present();

	int64_t next_sample_time();

	VideoStreamPlaybackWMF();
	~VideoStreamPlaybackWMF();
};
