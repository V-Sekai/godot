/**************************************************************************/
/*  playback_stats.cpp                                                    */
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

#include "playback_stats.h"
#include "speech_processor.h"

Dictionary PlaybackStats::get_playback_stats() {
	double playback_pushed_frames = playback_pushed_calls * (buffer_frame_count * 1.0);
	double playback_discarded_frames = playback_discarded_calls * (buffer_frame_count * 1.0);
	Dictionary dict;
	dict["playback_ring_limit_s"] = playback_ring_buffer_length / double(SpeechProcessor::SPEECH_SETTING_VOICE_PACKET_SAMPLE_RATE);
	dict["playback_ring_current_size_s"] = playback_ring_current_size / double(SpeechProcessor::SPEECH_SETTING_VOICE_PACKET_SAMPLE_RATE);
	dict["playback_ring_max_size_s"] = playback_ring_max_size / double(SpeechProcessor::SPEECH_SETTING_VOICE_PACKET_SAMPLE_RATE);
	dict["playback_ring_mean_size_s"] = 0;
	if (playback_push_buffer_calls > 0) {
		dict["playback_ring_mean_size_s"] = playback_ring_size_sum / playback_push_buffer_calls / double(SpeechProcessor::SPEECH_SETTING_VOICE_PACKET_SAMPLE_RATE);
	} else {
		dict["playback_ring_mean_size_s"] = 0;
	}
	dict["jitter_buffer_current_size_s"] = float(jitter_buffer_current_size) * SpeechProcessor::SPEECH_SETTING_PACKET_DELTA_TIME;
	dict["jitter_buffer_max_size_s"] = float(jitter_buffer_max_size) * SpeechProcessor::SPEECH_SETTING_PACKET_DELTA_TIME;
	dict["jitter_buffer_mean_size_s"] = 0;
	if (jitter_buffer_calls > 0) {
		dict["jitter_buffer_mean_size_s"] = float(jitter_buffer_size_sum) / jitter_buffer_calls * SpeechProcessor::SPEECH_SETTING_PACKET_DELTA_TIME;
	}
	dict["jitter_buffer_calls"] = jitter_buffer_calls;
	dict["playback_position_s"] = playback_position;
	dict["playback_get_percent"] = 0;
	dict["playback_discard_percent"] = 0;
	if (playback_pushed_frames > 0) {
		dict["playback_get_percent"] = 100.0 * playback_get_frames / playback_pushed_frames;
		dict["playback_discard_percent"] = 100.0 * playback_discarded_frames / playback_pushed_frames;
	}
	dict["playback_get_s"] = playback_get_frames / double(SpeechProcessor::SPEECH_SETTING_VOICE_PACKET_SAMPLE_RATE);
	dict["playback_pushed_s"] = playback_pushed_frames / double(SpeechProcessor::SPEECH_SETTING_VOICE_PACKET_SAMPLE_RATE);
	dict["playback_discarded_s"] = playback_discarded_frames / double(SpeechProcessor::SPEECH_SETTING_VOICE_PACKET_SAMPLE_RATE);
	dict["playback_push_buffer_calls"] = floor(playback_push_buffer_calls);
	dict["playback_blank_s"] = playback_blank_push_calls * SpeechProcessor::SPEECH_SETTING_PACKET_DELTA_TIME;
	dict["playback_blank_percent"] = 0;
	if (playback_push_buffer_calls > 0) {
		dict["playback_blank_percent"] = 100.0 * playback_blank_push_calls / playback_push_buffer_calls;
	}
	dict["playback_skips"] = floor(playback_skips);
	return dict;
}

void PlaybackStats::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_playback_stats"),
			&PlaybackStats::get_playback_stats);
}
