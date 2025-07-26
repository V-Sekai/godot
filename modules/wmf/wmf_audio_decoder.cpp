/**************************************************************************/
/*  wmf_audio_decoder.cpp                                                 */
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

#include "wmf_audio_decoder.h"

#include "core/error/error_macros.h"
#include "core/string/print_string.h"
#include <mfapi.h>
#include <mferror.h>

// Undefine Windows macros that conflict with Godot
#ifdef CONNECT_DEFERRED
#undef CONNECT_DEFERRED
#endif
#ifdef CONNECT_ONESHOT
#undef CONNECT_ONESHOT
#endif

#define CHECK_HR(func)                                                           \
	if (SUCCEEDED(hr)) {                                                         \
		hr = (func);                                                             \
		if (FAILED(hr)) {                                                        \
			print_line("WMF Audio Decoder function failed, HRESULT: " + itos(hr)); \
		}                                                                        \
	}

#define SafeRelease(p)      \
	{                       \
		if (p) {            \
			(p)->Release(); \
			(p) = nullptr;  \
		}                   \
	}

WMFAudioDecoder::WMFAudioDecoder() {
	sample_buffer.resize(24); // Buffer for 24 audio samples
}

WMFAudioDecoder::~WMFAudioDecoder() {
	shutdown();
}

Error WMFAudioDecoder::initialize(WMFContainerDecoder *p_container, int p_audio_stream_index) {
	ERR_FAIL_COND_V(!p_container, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(!p_container->is_valid(), ERR_UNCONFIGURED);
	
	container_decoder = p_container;
	audio_stream_index = p_audio_stream_index;
	
	// Get audio stream information
	StreamInfo stream_info = container_decoder->get_stream_info(audio_stream_index);
	audio_channels = stream_info.channels;
	audio_sample_rate = stream_info.sample_rate;
	audio_bits_per_sample = stream_info.bits_per_sample;
	
	print_line("WMF Audio Decoder: Initializing for stream " + itos(audio_stream_index));
	print_line("WMF Audio Decoder: " + itos(audio_channels) + " channels, " + itos(audio_sample_rate) + "Hz, " + itos(audio_bits_per_sample) + " bits");
	
	HRESULT hr = create_audio_topology();
	if (FAILED(hr)) {
		cleanup();
		return ERR_CANT_CREATE;
	}
	
	is_initialized = true;
	return OK;
}

void WMFAudioDecoder::shutdown() {
	MutexLock lock(audio_mutex);
	
	if (is_playing) {
		stop_decoding();
	}
	
	cleanup();
	
	container_decoder = nullptr;
	audio_stream_index = -1;
	is_initialized = false;
}

HRESULT WMFAudioDecoder::create_audio_topology() {
	// TODO: Implement audio topology creation
	// This will create a WMF topology specifically for audio decoding
	// using the container decoder's media source and the selected audio stream
	
	HRESULT hr = S_OK;
	
	// For now, return success - full implementation needed
	print_line("WMF Audio Decoder: Audio topology creation - implementation needed");
	
	return hr;
}

void WMFAudioDecoder::cleanup() {
	SafeRelease(presentation_clock);
	SafeRelease(topology);
	SafeRelease(media_session);
	
	sample_buffer.clear();
	sample_buffer.resize(24);
	read_sample_idx = 0;
	write_sample_idx = 0;
	
	is_playing = false;
	is_paused = false;
}

Error WMFAudioDecoder::start_decoding() {
	if (!is_initialized) {
		return ERR_UNCONFIGURED;
	}
	
	if (is_playing) {
		return OK; // Already playing
	}
	
	// TODO: Start audio decoding session
	print_line("WMF Audio Decoder: Starting audio decoding");
	
	is_playing = true;
	is_paused = false;
	
	return OK;
}

Error WMFAudioDecoder::stop_decoding() {
	if (!is_playing) {
		return OK; // Already stopped
	}
	
	// TODO: Stop audio decoding session
	print_line("WMF Audio Decoder: Stopping audio decoding");
	
	is_playing = false;
	is_paused = false;
	
	return OK;
}

Error WMFAudioDecoder::pause_decoding(bool p_paused) {
	if (!is_initialized) {
		return ERR_UNCONFIGURED;
	}
	
	// TODO: Pause/resume audio decoding session
	print_line("WMF Audio Decoder: " + String(p_paused ? "Pausing" : "Resuming") + " audio decoding");
	
	is_paused = p_paused;
	
	return OK;
}

Error WMFAudioDecoder::seek_to_time(double time_seconds) {
	if (!is_initialized) {
		return ERR_UNCONFIGURED;
	}
	
	// TODO: Seek audio decoder to specific time
	print_line("WMF Audio Decoder: Seeking to " + rtos(time_seconds) + " seconds");
	
	// Clear audio buffer after seek
	MutexLock lock(audio_mutex);
	read_sample_idx = write_sample_idx = 0;
	
	return OK;
}

bool WMFAudioDecoder::has_sample_available() const {
	MutexLock lock(audio_mutex);
	return read_sample_idx != write_sample_idx;
}

WMFAudioSample WMFAudioDecoder::get_next_sample() {
	MutexLock lock(audio_mutex);
	
	if (read_sample_idx == write_sample_idx) {
		// No samples available
		return WMFAudioSample();
	}
	
	WMFAudioSample sample = sample_buffer[read_sample_idx];
	read_sample_idx = (read_sample_idx + 1) % sample_buffer.size();
	
	return sample;
}

WMFAudioSample *WMFAudioDecoder::get_next_writable_sample() {
	return &sample_buffer.write[write_sample_idx];
}

void WMFAudioDecoder::write_sample_done() {
	MutexLock lock(audio_mutex);
	int next_write_idx = (write_sample_idx + 1) % sample_buffer.size();
	
	// Handle buffer overflow by advancing read index if needed
	if (read_sample_idx == next_write_idx) {
		read_sample_idx = (read_sample_idx + 1) % sample_buffer.size();
	}
	
	write_sample_idx = next_write_idx;
}
