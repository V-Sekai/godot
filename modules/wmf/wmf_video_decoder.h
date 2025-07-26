/**************************************************************************/
/*  wmf_video_decoder.h                                                   */
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

#include "wmf_container_decoder.h"
#include "core/os/mutex.h"
#include "core/templates/vector.h"
#include "scene/resources/image_texture.h"

#include <mfapi.h>
#include <mfidl.h>

class SampleGrabberCallback;
struct IMFMediaSession;
struct IMFTopology;
struct IMFPresentationClock;

struct VideoFrame {
	int64_t timestamp = 0;
	Vector<uint8_t> data;
	int width = 0;
	int height = 0;
	Ref<ImageTexture> texture;
	double presentation_time = 0.0;
};

class WMFVideoDecoder {
private:
	WMFContainerDecoder *container_decoder = nullptr;
	
	IMFMediaSession *media_session = nullptr;
	IMFTopology *topology = nullptr;
	IMFPresentationClock *presentation_clock = nullptr;
	SampleGrabberCallback *sample_grabber_callback = nullptr;
	
	Vector<VideoFrame> frame_buffer;
	int read_frame_idx = 0;
	int write_frame_idx = 0;
	Mutex video_mutex;
	
	bool is_initialized = false;
	bool is_playing = false;
	bool is_paused = false;
	
	// Video properties
	int video_width = 0;
	int video_height = 0;
	float video_fps = 0.0f;
	int video_stream_index = -1;
	
	HRESULT create_video_topology();
	HRESULT create_video_topology_internal(IMFMediaSource *media_source);
	HRESULT create_sample_grabber_sink(IMFActivate **sink_activate);
	void cleanup();

public:
	WMFVideoDecoder();
	~WMFVideoDecoder();
	
	// Initialization
	Error initialize(WMFContainerDecoder *p_container, int p_video_stream_index);
	void shutdown();
	
	// Playback control
	Error start_decoding();
	Error stop_decoding();
	Error pause_decoding(bool p_paused);
	Error seek_to_time(double time_seconds);
	
	// Frame access
	bool has_frame_available() const;
	VideoFrame get_next_frame();
	
	// Properties
	int get_width() const { return video_width; }
	int get_height() const { return video_height; }
	float get_fps() const { return video_fps; }
	bool is_decoder_playing() const { return is_playing; }
	bool is_decoder_paused() const { return is_paused; }
	
	// Frame buffer management (called by SampleGrabberCallback)
	VideoFrame *get_next_writable_frame();
	void write_frame_done();
	
	// Utility
	double get_frame_time(int64_t timestamp) const;
};
