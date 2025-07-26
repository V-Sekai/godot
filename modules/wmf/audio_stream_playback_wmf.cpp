/**************************************************************************/
/*  audio_stream_playback_wmf.cpp                                         */
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

#include "audio_stream_playback_wmf.h"

#include "core/string/print_string.h"

// AudioStreamPlaybackWMF Implementation

AudioStreamPlaybackWMF::AudioStreamPlaybackWMF() {
	mix_buffer.resize(4096); // Default buffer size
}

AudioStreamPlaybackWMF::~AudioStreamPlaybackWMF() {
	stop();
}

void AudioStreamPlaybackWMF::start(double p_from_pos) {
	MutexLock lock(audio_mutex);
	
	if (!audio_decoder) {
		print_line("AudioStreamPlaybackWMF: No audio decoder available");
		return;
	}
	
	playback_position = p_from_pos;
	
	if (p_from_pos > 0.0) {
		audio_decoder->seek_to_time(p_from_pos);
	}
	
	Error err = audio_decoder->start_decoding();
	if (err != OK) {
		print_line("AudioStreamPlaybackWMF: Failed to start audio decoder");
		return;
	}
	
	playing_state = true;
	paused_state = false;
	
	print_line("AudioStreamPlaybackWMF: Started playback from " + rtos(p_from_pos) + " seconds");
}

void AudioStreamPlaybackWMF::stop() {
	MutexLock lock(audio_mutex);
	
	if (!playing_state) {
		return;
	}
	
	if (audio_decoder) {
		audio_decoder->stop_decoding();
	}
	
	playing_state = false;
	paused_state = false;
	playback_position = 0.0;
	
	print_line("AudioStreamPlaybackWMF: Stopped playback");
}

bool AudioStreamPlaybackWMF::is_playing() const {
	return playing_state && !paused_state;
}

int AudioStreamPlaybackWMF::get_loop_count() const {
	return 0; // No looping for video audio
}

double AudioStreamPlaybackWMF::get_playback_position() const {
	return playback_position;
}

void AudioStreamPlaybackWMF::seek(double p_time) {
	MutexLock lock(audio_mutex);
	
	if (!audio_decoder) {
		return;
	}
	
	playback_position = p_time;
	audio_decoder->seek_to_time(p_time);
	
	print_line("AudioStreamPlaybackWMF: Seeked to " + rtos(p_time) + " seconds");
}

int AudioStreamPlaybackWMF::mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) {
	if (!playing_state || paused_state || !audio_decoder) {
		// Fill with silence
		for (int i = 0; i < p_frames; i++) {
			p_buffer[i] = AudioFrame(0.0f, 0.0f);
		}
		return p_frames;
	}
	
	MutexLock lock(audio_mutex);
	
	int frames_mixed = 0;
	double frame_duration = 1.0 / static_cast<double>(mix_rate);
	
	// Get audio samples from decoder and mix them with proper timing
	while (frames_mixed < p_frames) {
		if (!audio_decoder->has_sample_available()) {
			// No more samples available, fill rest with silence
			break;
		}
		
		WMFAudioSample sample = audio_decoder->get_next_sample();
		
		if (sample.data.is_empty()) {
			break;
		}
		
		// Calculate expected sample time vs actual playback position
		double sample_time = static_cast<double>(sample.timestamp) / 10000000.0; // Convert from 100ns units
		double time_diff = sample_time - playback_position;
		
		// Skip samples that are too far behind (more than 1 frame duration)
		if (time_diff < -frame_duration) {
			continue;
		}
		
		// Convert sample data to AudioFrames with proper rate scaling
		int sample_frames = sample.data.size() / sample.channels;
		int frames_to_copy = MIN(sample_frames, p_frames - frames_mixed);
		
		for (int i = 0; i < frames_to_copy; i++) {
			float left = 0.0f;
			float right = 0.0f;
			
			if (sample.channels == 1) {
				// Mono - duplicate to both channels
				left = right = sample.data[i] * p_rate_scale;
			} else if (sample.channels >= 2) {
				// Stereo or more - take first two channels
				left = sample.data[i * sample.channels] * p_rate_scale;
				right = sample.data[i * sample.channels + 1] * p_rate_scale;
			}
			
			p_buffer[frames_mixed] = AudioFrame(left, right);
			frames_mixed++;
		}
		
		// Update playback position more accurately
		playback_position += static_cast<double>(frames_to_copy) * frame_duration;
	}
	
	// Update audio clock in synchronizer (audio is master clock)
	if (synchronizer.is_valid()) {
		synchronizer->update_audio_clock(playback_position);
	}
	
	// Fill remaining frames with silence if needed
	for (int i = frames_mixed; i < p_frames; i++) {
		p_buffer[i] = AudioFrame(0.0f, 0.0f);
	}
	
	return p_frames;
}

void AudioStreamPlaybackWMF::set_audio_decoder(WMFAudioDecoder *p_decoder) {
	MutexLock lock(audio_mutex);
	audio_decoder = p_decoder;
	
	if (audio_decoder) {
		channels = audio_decoder->get_channels();
		mix_rate = audio_decoder->get_sample_rate();
		print_line("AudioStreamPlaybackWMF: Set audio decoder - " + itos(channels) + " channels, " + itos(mix_rate) + "Hz");
	}
}

void AudioStreamPlaybackWMF::set_synchronizer(const Ref<AudioVideoSynchronizer> &p_synchronizer) {
	MutexLock lock(audio_mutex);
	synchronizer = p_synchronizer;
	if (synchronizer.is_valid()) {
		print_line("AudioStreamPlaybackWMF: Synchronizer set");
	}
}

Ref<AudioVideoSynchronizer> AudioStreamPlaybackWMF::get_synchronizer() const {
	return synchronizer;
}

void AudioStreamPlaybackWMF::set_paused(bool p_paused) {
	MutexLock lock(audio_mutex);
	
	if (paused_state == p_paused) {
		return;
	}
	
	paused_state = p_paused;
	
	if (audio_decoder) {
		audio_decoder->pause_decoding(p_paused);
	}
	
	print_line("AudioStreamPlaybackWMF: " + String(p_paused ? "Paused" : "Resumed") + " playback");
}

// AudioStreamWMF Implementation

AudioStreamWMF::AudioStreamWMF() {
}

AudioStreamWMF::~AudioStreamWMF() {
}

Ref<AudioStreamPlayback> AudioStreamWMF::instantiate_playback() {
	Ref<AudioStreamPlaybackWMF> playback = memnew(AudioStreamPlaybackWMF);
	playback->set_audio_decoder(audio_decoder);
	
	// Set synchronizer if available (shared from video stream)
	if (shared_synchronizer.is_valid()) {
		playback->set_synchronizer(shared_synchronizer);
	}
	
	return playback;
}

String AudioStreamWMF::get_stream_name() const {
	return "WMF Audio Stream";
}

double AudioStreamWMF::get_length() const {
	return length;
}

void AudioStreamWMF::set_audio_decoder(WMFAudioDecoder *p_decoder) {
	audio_decoder = p_decoder;
	// Duration will be set by the container decoder
}

void AudioStreamWMF::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_length"), &AudioStreamWMF::get_length);
	ClassDB::bind_method(D_METHOD("get_stream_name"), &AudioStreamWMF::get_stream_name);
}
