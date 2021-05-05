/**************************************************************************/
/*  video_stream_wmf.h                                                    */
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

#ifndef WINDOWS_MEDIA_FOUNDATION_VIDEO_H
#define WINDOWS_MEDIA_FOUNDATION_VIDEO_H

#include "core/io/resource_loader.h"
#include "core/os/mutex.h"
#include "scene/resources/video_stream.h"

#include <deque>

class SampleGrabberCallback;
struct IMFMediaSession;
struct IMFMediaSource;
struct IMFTopology;
struct IMFPresentationClock;

struct FrameData {
	int64_t sample_time = 0;
	PoolVector<uint8_t> data;
};

class VideoStreamPlaybackWMF : public VideoStreamPlayback {
	GDCLASS(VideoStreamPlaybackWMF, VideoStreamPlayback);

	IMFMediaSession *media_session;
	IMFMediaSource *media_source;
	IMFTopology *topology;
	IMFPresentationClock *presentation_clock;
	SampleGrabberCallback *sample_grabber_callback;

	Vector<FrameData> cache_frames;
	int read_frame_idx;
	int write_frame_idx;

	PoolVector<uint8_t> frame_data;
	Ref<ImageTexture> texture;
	Mutex mtx;

	bool is_video_playing;
	bool is_video_paused;
	bool is_video_seekable;

	int id = 0;
	bool is_video_sync_enabled = false;

	void shutdown_stream();

public:
	struct StreamInfo {
		Point2i size;
		float fps;
		float duration;
	};
	StreamInfo stream_info;

	virtual void play();
	virtual void stop();
	virtual bool is_playing() const;

	virtual void set_paused(bool p_paused);
	virtual bool is_paused() const;

	virtual void set_loop(bool p_enable);
	virtual bool has_loop() const;

	virtual float get_length() const;
	virtual String get_stream_name() const;

	virtual int get_loop_count() const;

	virtual float get_playback_position() const;
	virtual void seek(float p_time);

	void set_file(const String &p_file);

	virtual Ref<Texture> get_texture() const;
	virtual void update(float p_delta);

	virtual void set_mix_callback(AudioMixCallback p_callback, void *p_userdata);
	virtual int get_channels() const;
	virtual int get_mix_rate() const;

	virtual void set_audio_track(int p_idx);

	FrameData *get_next_writable_frame();
	void write_frame_done();
	void present();

	void enable_sync(bool b) { is_video_sync_enabled = b; }
	int64_t next_sample_time();

	VideoStreamPlaybackWMF();
	~VideoStreamPlaybackWMF();
};

class VideoStreamWMF : public VideoStream {
	GDCLASS(VideoStreamWMF, VideoStream);
	RES_BASE_EXTENSION("wmfvstr");

	String file;
	int audio_track;

protected:
	static void _bind_methods();

public:
	Ref<VideoStreamPlayback> instance_playback();

	void set_file(const String &p_file);

	String get_file() {
		return file;
	}

	void set_audio_track(int p_track) {
		audio_track = p_track;
	}

	VideoStreamWMF();
	~VideoStreamWMF();
};

#endif
