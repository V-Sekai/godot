/**************************************************************************/
/*  speech.h                                                              */
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

#include "core/error/error_macros.h"
#include "thirdparty/libsamplerate/src/samplerate.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/os/mutex.h"
#include "core/variant/array.h"
#include "core/variant/dictionary.h"
#include "scene/main/node.h"
#include "servers/audio_server.h"

#include "servers/audio/effects/audio_stream_generator.h"
#include "speech_processor.h"

class PlaybackStats : public RefCounted {
	GDCLASS(PlaybackStats, RefCounted);

protected:
	static void _bind_methods();

public:
	int64_t playback_ring_current_size = 0;
	int64_t playback_ring_max_size = 0;
	int64_t playback_ring_size_sum = 0;
	double playback_get_frames = 0.0;
	int64_t playback_pushed_calls = 0;
	int64_t playback_discarded_calls = 0;
	int64_t playback_push_buffer_calls = 0;
	int64_t playback_blank_push_calls = 0;
	double playback_position = 0.0;
	double playback_skips = 0.0;

	double jitter_buffer_size_sum = 0.0;
	int64_t jitter_buffer_calls = 0;
	int64_t jitter_buffer_max_size = 0;
	int64_t jitter_buffer_current_size = 0;

	int64_t playback_ring_buffer_length = 0;
	int64_t buffer_frame_count = 0;
	Dictionary get_playback_stats();
};

class Speech : public Node {
	GDCLASS(Speech, Node);

	static const int MAX_AUDIO_BUFFER_ARRAY_SIZE = 10;

	PackedByteArray input_byte_array;
	float volume = 0.0;

	Mutex audio_mutex;

	int skipped_audio_packets = 0;

	SpeechProcessor *speech_processor = nullptr;

	struct InputPacket {
		PackedByteArray compressed_byte_array;
		int buffer_size = 0;
		float loudness = 0.0;
	};

	int current_input_size = 0;
	PackedByteArray compression_output_byte_array;
	InputPacket input_audio_buffer_array[MAX_AUDIO_BUFFER_ARRAY_SIZE];
	//
private:
	// Assigns the memory to the fixed audio buffer arrays
	void preallocate_buffers();

	// Assigns a callback from the speech_processor to this object.
	void setup_connections();

	// Returns a pointer to the first valid input packet
	// If the current_input_size has exceeded MAX_AUDIO_BUFFER_ARRAY_SIZE,
	// The front packet will be popped from the queue back recursively
	// copying from the back.
	InputPacket *get_next_valid_input_packet();

	// Is responsible for receiving packets from the SpeechProcessor and then
	// compressing them
	void speech_processed(SpeechProcessor::SpeechInput *p_mic_input);

private:
	float BUFFER_DELAY_THRESHOLD = 0.1;
	float STREAM_STANDARD_PITCH = 1.0;
	float STREAM_SPEEDUP_PITCH = 1.5;
	int MAX_JITTER_BUFFER_SIZE = 16;
	int JITTER_BUFFER_SPEEDUP = 12;
	int JITTER_BUFFER_SLOWDOWN = 6;

	bool DEBUG = false;
	PackedVector2Array uncompressed_audio;

	int packets_received_this_frame = 0;
	int playback_ring_buffer_length = 0;

	PackedVector2Array blank_packet;
	Dictionary player_audio;
	int nearest_shift(int p_number);

public:
	int get_jitter_buffer_speedup() const;
	void set_jitter_buffer_speedup(int p_jitter_buffer_speedup);
	int get_jitter_buffer_slowdown() const;
	void set_jitter_buffer_slowdown(int p_jitter_buffer_slowdown);
	float get_stream_speedup_pitch() const;
	void set_stream_speedup_pitch(float p_stream_speedup_pitch);
	int get_max_jitter_buffer_size() const;
	void set_max_jitter_buffer_size(int p_max_jitter_buffer_size);
	float get_buffer_delay_threshold() const;
	void set_buffer_delay_threshold(float p_buffer_delay_threshold);
	float get_stream_standard_pitch() const;
	void set_stream_standard_pitch(float p_stream_standard_pitch);
	bool get_debug() const;
	void set_debug(bool val);
	PackedVector2Array get_uncompressed_audio() const;
	void set_uncompressed_audio(PackedVector2Array val);
	int get_packets_received_this_frame() const;
	void set_packets_received_this_frame(int val);
	int get_playback_ring_buffer_length() const;
	void set_playback_ring_buffer_length(int val);
	PackedVector2Array get_blank_packet() const;
	void set_blank_packet(PackedVector2Array val);
	Dictionary get_player_audio();
	void set_player_audio(Dictionary val);
	int calc_playback_ring_buffer_length(Ref<AudioStreamGenerator> audio_stream_generator);

protected:
	static void _bind_methods();

	int get_skipped_audio_packets();

	void clear_skipped_audio_packets();

	virtual PackedVector2Array
	decompress_buffer(Ref<SpeechDecoder> p_speech_decoder,
			PackedByteArray p_read_byte_array, const int p_read_size,
			PackedVector2Array p_write_vec2_array);
	// Copies all the input buffers to the output buffers
	// Returns the amount of buffers
	Array copy_and_clear_buffers();
	Ref<SpeechDecoder> get_speech_decoder();
	bool start_recording();
	bool end_recording();
	void _notification(int p_what);
	void set_streaming_bus(const String &p_name);
	void set_error_cancellation_bus(const String &p_name);
	bool set_audio_input_stream_player(Node *p_audio_stream);
	Dictionary get_stats();
	Speech();
	~Speech();

public:
	void add_player_audio(int p_player_id, Node *p_audio_stream_player);
	void vc_debug_print(String p_str) const;
	void vc_debug_printerr(String p_str) const;
	void on_received_audio_packet(int p_peer_id, int p_sequence_id, PackedByteArray p_packet);
	Dictionary get_playback_stats(Dictionary speech_stat_dict);
	void remove_player_audio(int p_player_id);
	void clear_all_player_audio();
	void attempt_to_feed_stream(int p_skip_count, Ref<SpeechDecoder> p_decoder, Node *p_audio_stream_player, Array p_jitter_buffer, Ref<PlaybackStats> p_playback_stats, Dictionary p_player_dict);
};
