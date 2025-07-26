/**************************************************************************/
/*  wmf_audio_decoder.h                                                   */
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

#include <mfapi.h>
#include <mfidl.h>

class AudioSampleGrabberCallback;
struct IMFMediaSession;
struct IMFTopology;
struct IMFPresentationClock;

struct WMFAudioSample {
	int64_t timestamp = 0;
	Vector<float> data;
	int channels = 0;
	int sample_rate = 0;
};

class WMFAudioDecoder {
private:
	WMFContainerDecoder *container_decoder = nullptr;
	
	IMFMediaSession *media_session = nullptr;
	IMFTopology *topology = nullptr;
	IMFPresentationClock *presentation_clock = nullptr;
	AudioSampleGrabberCallback *sample_grabber_callback = nullptr;
	
	Vector<WMFAudioSample> sample_buffer;
	int read_sample_idx = 0;
	int write_sample_idx = 0;
	Mutex audio_mutex;
	
	bool is_initialized = false;
	bool is_playing = false;
	bool is_paused = false;
	
	// Audio properties
	int audio_channels = 0;
	int audio_sample_rate = 0;
	int audio_bits_per_sample = 0;
	int audio_stream_index = -1;
	
	HRESULT create_audio_topology();
	void cleanup();

public:
	WMFAudioDecoder();
	~WMFAudioDecoder();
	
	// Initialization
	Error initialize(WMFContainerDecoder *p_container, int p_audio_stream_index);
	void shutdown();
	
	// Playback control
	Error start_decoding();
	Error stop_decoding();
	Error pause_decoding(bool p_paused);
	Error seek_to_time(double time_seconds);
	
	// Sample access
	bool has_sample_available() const;
	WMFAudioSample get_next_sample();
	
	// Properties
	int get_channels() const { return audio_channels; }
	int get_sample_rate() const { return audio_sample_rate; }
	int get_bits_per_sample() const { return audio_bits_per_sample; }
	bool is_decoder_playing() const { return is_playing; }
	bool is_decoder_paused() const { return is_paused; }
	
	// Sample buffer management (called by AudioSampleGrabberCallback)
	WMFAudioSample *get_next_writable_sample();
	void write_sample_done();
};
