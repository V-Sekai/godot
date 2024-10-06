/**************************************************************************/
/*  speech.cpp                                                            */
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

#include "scene/2d/audio_stream_player_2d.h"
#include "scene/3d/audio_stream_player_3d.h"

#include "speech.h"
#include "speech_processor.h"

void Speech::preallocate_buffers() {
	input_byte_array.resize(SpeechProcessor::SPEECH_SETTING_PCM_BUFFER_SIZE);
	input_byte_array.fill(0);
	compression_output_byte_array.resize(
			SpeechProcessor::SPEECH_SETTING_PCM_BUFFER_SIZE);
	compression_output_byte_array.fill(0);
	for (int i = 0; i < MAX_AUDIO_BUFFER_ARRAY_SIZE; i++) {
		input_audio_buffer_array[i].compressed_byte_array.resize(
				SpeechProcessor::SPEECH_SETTING_PCM_BUFFER_SIZE);
		input_audio_buffer_array[i].compressed_byte_array.fill(0);
	}
}

void Speech::setup_connections() {
	if (speech_processor) {
		speech_processor->register_speech_processed(
				std::function<void(SpeechProcessor::SpeechInput *)>(std::bind(
						&Speech::speech_processed, this, std::placeholders::_1)));
	}
}

Speech::InputPacket *Speech::get_next_valid_input_packet() {
	if (current_input_size < MAX_AUDIO_BUFFER_ARRAY_SIZE) {
		InputPacket *input_packet = &input_audio_buffer_array[current_input_size];
		current_input_size++;
		return input_packet;
	} else {
		for (int i = MAX_AUDIO_BUFFER_ARRAY_SIZE - 1; i > 0; i--) {
			memcpy(input_audio_buffer_array[i - 1].compressed_byte_array.ptrw(),
					input_audio_buffer_array[i].compressed_byte_array.ptr(),
					SpeechProcessor::SPEECH_SETTING_PCM_BUFFER_SIZE);

			input_audio_buffer_array[i - 1].buffer_size =
					input_audio_buffer_array[i].buffer_size;
			input_audio_buffer_array[i - 1].loudness =
					input_audio_buffer_array[i].loudness;
		}
		skipped_audio_packets++;
		return &input_audio_buffer_array[MAX_AUDIO_BUFFER_ARRAY_SIZE - 1];
	}
}

void Speech::speech_processed(SpeechProcessor::SpeechInput *p_mic_input) {
	// Copy the raw PCM data from the SpeechInput packet to the input byte array
	PackedByteArray *mic_input_byte_array = p_mic_input->pcm_byte_array;
	memcpy(input_byte_array.ptrw(), mic_input_byte_array->ptr(),
			SpeechProcessor::SPEECH_SETTING_PCM_BUFFER_SIZE);

	// Create a new SpeechProcessor::CompressedBufferInput to be passed into the
	// compressor and assign it the compressed_byte_array from the input packet
	SpeechProcessor::CompressedSpeechBuffer compressed_buffer_input;
	compressed_buffer_input.compressed_byte_array =
			&compression_output_byte_array;

	// Compress the packet
	speech_processor->compress_buffer_internal(&input_byte_array,
			&compressed_buffer_input);
	{
		// Lock
		MutexLock mutex_lock(audio_mutex);

		int64_t size = compressed_buffer_input.buffer_size;
		ERR_FAIL_COND(size > SpeechProcessor::SPEECH_SETTING_PCM_BUFFER_SIZE);
		// Find the next valid input packet in the queue
		InputPacket *input_packet = get_next_valid_input_packet();
		// Copy the buffer size from the compressed_buffer_input back into the
		// input packet
		memcpy(input_packet->compressed_byte_array.ptrw(),
				compressed_buffer_input.compressed_byte_array->ptr(), size);

		input_packet->buffer_size = size;
		input_packet->loudness = p_mic_input->volume;
	}
}

int Speech::get_jitter_buffer_speedup() const {
	return JITTER_BUFFER_SPEEDUP;
}

void Speech::set_jitter_buffer_speedup(int p_jitter_buffer_speedup) {
	JITTER_BUFFER_SPEEDUP = p_jitter_buffer_speedup;
}

int Speech::get_jitter_buffer_slowdown() const {
	return JITTER_BUFFER_SLOWDOWN;
}

void Speech::set_jitter_buffer_slowdown(int p_jitter_buffer_slowdown) {
	JITTER_BUFFER_SLOWDOWN = p_jitter_buffer_slowdown;
}

float Speech::get_stream_speedup_pitch() const {
	return STREAM_SPEEDUP_PITCH;
}

void Speech::set_stream_speedup_pitch(float p_stream_speedup_pitch) {
	STREAM_SPEEDUP_PITCH = p_stream_speedup_pitch;
}

int Speech::get_max_jitter_buffer_size() const {
	return MAX_JITTER_BUFFER_SIZE;
}

void Speech::set_max_jitter_buffer_size(int p_max_jitter_buffer_size) {
	MAX_JITTER_BUFFER_SIZE = p_max_jitter_buffer_size;
}

float Speech::get_buffer_delay_threshold() const {
	return BUFFER_DELAY_THRESHOLD;
}

void Speech::set_buffer_delay_threshold(float p_buffer_delay_threshold) {
	BUFFER_DELAY_THRESHOLD = p_buffer_delay_threshold;
}

float Speech::get_stream_standard_pitch() const {
	return STREAM_STANDARD_PITCH;
}

void Speech::set_stream_standard_pitch(float p_stream_standard_pitch) {
	STREAM_STANDARD_PITCH = p_stream_standard_pitch;
}

bool Speech::get_debug() const {
	return DEBUG;
}

void Speech::set_debug(bool val) {
	DEBUG = val;
}

PackedVector2Array Speech::get_uncompressed_audio() const {
	return uncompressed_audio;
}

void Speech::set_uncompressed_audio(PackedVector2Array val) {
	uncompressed_audio = val;
}

int Speech::get_packets_received_this_frame() const {
	return packets_received_this_frame;
}

void Speech::set_packets_received_this_frame(int val) {
	packets_received_this_frame = val;
}

int Speech::get_playback_ring_buffer_length() const {
	return playback_ring_buffer_length;
}

void Speech::set_playback_ring_buffer_length(int val) {
	playback_ring_buffer_length = val;
}

PackedVector2Array Speech::get_blank_packet() const {
	return blank_packet;
}

void Speech::set_blank_packet(PackedVector2Array val) {
	blank_packet = val;
}

Dictionary Speech::get_player_audio() {
	return player_audio;
}

void Speech::set_player_audio(Dictionary val) {
	player_audio = val;
}

int Speech::nearest_shift(int p_number) {
	for (int32_t i = 30; i-- > 0;) {
		if (p_number & (1 << i)) {
			return i + 1;
		}
	}
	return 0;
}

int Speech::calc_playback_ring_buffer_length(Ref<AudioStreamGenerator> audio_stream_generator) {
	int target_buffer_size = int(audio_stream_generator->get_mix_rate() * audio_stream_generator->get_buffer_length());
	return (1 << nearest_shift(target_buffer_size));
}

void Speech::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_skipped_audio_packets"),
			&Speech::get_skipped_audio_packets);
	ClassDB::bind_method(D_METHOD("clear_skipped_audio_packets"),
			&Speech::clear_skipped_audio_packets);

	ClassDB::bind_method(D_METHOD("decompress_buffer", "decoder", "read_array",
								 "read_size", "write_array"),
			&Speech::decompress_buffer);

	ClassDB::bind_method(D_METHOD("copy_and_clear_buffers"),
			&Speech::copy_and_clear_buffers);
	ClassDB::bind_method(D_METHOD("get_speech_decoder"),
			&Speech::get_speech_decoder);
	ClassDB::bind_method(D_METHOD("get_stats"), &Speech::get_stats);

	ClassDB::bind_method(D_METHOD("start_recording"), &Speech::start_recording);
	ClassDB::bind_method(D_METHOD("end_recording"), &Speech::end_recording);

	ClassDB::bind_method(D_METHOD("set_streaming_bus", "bus"),
			&Speech::set_streaming_bus);
	ClassDB::bind_method(D_METHOD("set_audio_input_stream_player", "player"),
			&Speech::set_audio_input_stream_player);
	ClassDB::bind_method(D_METHOD("set_buffer_delay_threshold", "buffer_delay_threshold"),
			&Speech::set_buffer_delay_threshold);
	ClassDB::bind_method(D_METHOD("get_buffer_delay_threshold"),
			&Speech::get_buffer_delay_threshold);
	ClassDB::bind_method(D_METHOD("get_stream_standard_pitch"),
			&Speech::get_stream_standard_pitch);
	ClassDB::bind_method(D_METHOD("set_stream_standard_pitch", "stream_standard_pitch"),
			&Speech::set_stream_standard_pitch);
	ClassDB::bind_method(D_METHOD("get_stream_speedup_pitch"),
			&Speech::get_stream_standard_pitch);
	ClassDB::bind_method(D_METHOD("set_stream_speedup_pitch", "stream_speedup_pitch"),
			&Speech::set_stream_standard_pitch);
	ClassDB::bind_method(D_METHOD("get_max_jitter_buffer_size"),
			&Speech::get_max_jitter_buffer_size);
	ClassDB::bind_method(D_METHOD("set_max_jitter_buffer_size", "max_jitter_buffer_size"),
			&Speech::set_max_jitter_buffer_size);
	ClassDB::bind_method(D_METHOD("get_jitter_buffer_speedup"),
			&Speech::get_jitter_buffer_speedup);
	ClassDB::bind_method(D_METHOD("set_jitter_buffer_speedup", "jitter_buffer_speedup"),
			&Speech::set_jitter_buffer_speedup);
	ClassDB::bind_method(D_METHOD("get_jitter_buffer_slowdown"),
			&Speech::get_jitter_buffer_slowdown);
	ClassDB::bind_method(D_METHOD("set_jitter_buffer_slowdown", "jitter_buffer_slowdown"),
			&Speech::set_jitter_buffer_slowdown);
	ClassDB::bind_method(D_METHOD("get_debug"),
			&Speech::get_debug);
	ClassDB::bind_method(D_METHOD("set_debug", "debug"),
			&Speech::set_debug);
	ClassDB::bind_method(D_METHOD("get_uncompressed_audio"),
			&Speech::get_uncompressed_audio);
	ClassDB::bind_method(D_METHOD("set_uncompressed_audio", "uncompressed_audio"),
			&Speech::set_uncompressed_audio);
	ClassDB::bind_method(D_METHOD("get_packets_received_this_frame"),
			&Speech::get_packets_received_this_frame);
	ClassDB::bind_method(D_METHOD("set_packets_received_this_frame", "packets_received_this_frame"),
			&Speech::set_packets_received_this_frame);
	ClassDB::bind_method(D_METHOD("get_playback_ring_buffer_length"),
			&Speech::get_playback_ring_buffer_length);
	ClassDB::bind_method(D_METHOD("set_playback_ring_buffer_length", "playback_ring_buffer_length"),
			&Speech::set_playback_ring_buffer_length);
	ClassDB::bind_method(D_METHOD("get_blank_packet"),
			&Speech::get_blank_packet);
	ClassDB::bind_method(D_METHOD("set_blank_packet", "blank_packet"),
			&Speech::set_blank_packet);
	ClassDB::bind_method(D_METHOD("get_player_audio"),
			&Speech::get_player_audio);
	ClassDB::bind_method(D_METHOD("set_player_audio", "player_audio"),
			&Speech::set_player_audio);
	ClassDB::bind_method(D_METHOD("calc_playback_ring_buffer_length", "generator"),
			&Speech::calc_playback_ring_buffer_length);
	ClassDB::bind_method(D_METHOD("add_player_audio", "player_id", "audio_stream_player"),
			&Speech::add_player_audio);
	ClassDB::bind_method(D_METHOD("on_received_audio_packet", "peer_id", "sequence_id", "packet"),
			&Speech::on_received_audio_packet);
	ClassDB::bind_method(D_METHOD("get_playback_stats", "speech_stat"),
			&Speech::get_playback_stats);
	ClassDB::bind_method(D_METHOD("remove_player_audio", "player_id"),
			&Speech::remove_player_audio);
	ClassDB::bind_method(D_METHOD("clear_all_player_audio"),
			&Speech::clear_all_player_audio);
	ClassDB::bind_method(D_METHOD("attempt_to_feed_stream", "skip_count", "decoder", "audio_stream_player", "jitter_buffer", "playback_stats", "player_dict"),
			&Speech::attempt_to_feed_stream);
	ClassDB::bind_method(D_METHOD("set_error_cancellation_bus", "name"),
			&Speech::set_error_cancellation_bus);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "BUFFER_DELAY_THRESHOLD"), "set_buffer_delay_threshold",
			"get_buffer_delay_threshold");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "STREAM_STANDARD_PITCH"), "set_stream_standard_pitch",
			"get_stream_standard_pitch");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "MAX_JITTER_BUFFER_SIZE"), "set_max_jitter_buffer_size",
			"get_max_jitter_buffer_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "STREAM_SPEEDUP_PITCH"), "set_stream_speedup_pitch",
			"get_stream_speedup_pitch");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "JITTER_BUFFER_SLOWDOWN"), "set_jitter_buffer_slowdown",
			"get_jitter_buffer_slowdown");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "JITTER_BUFFER_SPEEDUP"), "set_jitter_buffer_speedup",
			"get_jitter_buffer_speedup");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "DEBUG"), "set_debug",
			"get_debug");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_VECTOR2_ARRAY, "uncompressed_audio", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "set_uncompressed_audio",
			"get_uncompressed_audio");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "packets_received_this_frame"), "set_packets_received_this_frame",
			"get_packets_received_this_frame");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "playback_ring_buffer_length"), "set_playback_ring_buffer_length",
			"get_playback_ring_buffer_length");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_VECTOR2_ARRAY, "blank_packet", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "set_blank_packet",
			"get_blank_packet");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "player_audio", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "set_player_audio",
			"get_player_audio");
}

int Speech::get_skipped_audio_packets() {
	return skipped_audio_packets;
}

void Speech::clear_skipped_audio_packets() {
	skipped_audio_packets = 0;
}

PackedVector2Array Speech::decompress_buffer(Ref<SpeechDecoder> p_speech_decoder, PackedByteArray p_read_byte_array, const int p_read_size, PackedVector2Array p_write_vec2_array) {
	if (p_read_byte_array.size() < p_read_size) {
		ERR_PRINT("SpeechDecoder: read byte_array size!");
		return PackedVector2Array();
	}

	if (speech_processor->decompress_buffer_internal(
				p_speech_decoder.ptr(), &p_read_byte_array, p_read_size,
				&p_write_vec2_array)) {
		return p_write_vec2_array;
	}

	return PackedVector2Array();
}

Array Speech::copy_and_clear_buffers() {
	MutexLock mutex_lock(audio_mutex);

	Array output_array;
	output_array.resize(current_input_size);

	for (int i = 0; i < current_input_size; i++) {
		Dictionary dict;

		dict["byte_array"] = input_audio_buffer_array[i].compressed_byte_array;
		dict["buffer_size"] = input_audio_buffer_array[i].buffer_size;
		dict["loudness"] = input_audio_buffer_array[i].loudness;

		output_array[i] = dict;
	}
	current_input_size = 0;

	return output_array;
}

Ref<SpeechDecoder> Speech::get_speech_decoder() {
	if (speech_processor) {
		return speech_processor->get_speech_decoder();
	} else {
		return nullptr;
	}
}

bool Speech::start_recording() {
	if (speech_processor) {
		speech_processor->start();
		skipped_audio_packets = 0;
		return true;
	}

	return false;
}

bool Speech::end_recording() {
	bool result = true;
	if (speech_processor) {
		speech_processor->stop();
	} else {
		result = false;
	}
	if (has_method("clear_all_player_audio")) {
		call("clear_all_player_audio");
	}
	return result;
}

void Speech::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			setup_connections();
			if (speech_processor) {
				add_child(speech_processor, true);
				speech_processor->set_owner(get_owner());
			}
			uncompressed_audio.resize(
					SpeechProcessor::SPEECH_SETTING_BUFFER_FRAME_COUNT);
			uncompressed_audio.fill(Vector2());
			set_process_internal(true);
			break;
		}
		case NOTIFICATION_EXIT_TREE: {
			if (speech_processor) {
				remove_child(speech_processor);
			}
			break;
		}
		case NOTIFICATION_POSTINITIALIZE: {
			blank_packet.resize(SpeechProcessor::SPEECH_SETTING_BUFFER_FRAME_COUNT);
			blank_packet.fill(Vector2());
			for (int32_t i = 0; i < SpeechProcessor::SPEECH_SETTING_BUFFER_FRAME_COUNT; i++) {
				blank_packet.write[i] = Vector2();
			}
			break;
		}
		case NOTIFICATION_INTERNAL_PROCESS: {
			Array keys = player_audio.keys();
			for (int32_t i = 0; i < keys.size(); i++) {
				Variant key = keys[i];
				if (!player_audio.has(key)) {
					continue;
				}
				Dictionary elem = player_audio[key];
				if (!elem.has("speech_decoder")) {
					continue;
				}
				Ref<SpeechDecoder> speech_decoder = elem["speech_decoder"];
				if (!elem.has("audio_stream_player")) {
					continue;
				}
				Node *audio_stream_player = cast_to<Node>(elem["audio_stream_player"]);
				if (!elem.has("jitter_buffer")) {
					continue;
				}
				Array jitter_buffer = elem["jitter_buffer"];
				if (!elem.has("playback_stats")) {
					continue;
				}
				Ref<PlaybackStats> playback_stats = elem["playback_stats"];
				attempt_to_feed_stream(
						0,
						speech_decoder,
						audio_stream_player,
						jitter_buffer,
						playback_stats,
						elem);
				Dictionary dict = player_audio[key];
				dict["packets_received_this_frame"] = 0;
				player_audio[key] = dict;
			}
			packets_received_this_frame = 0;
			break;
		}
		default: {
			break;
		}
	}
}

void Speech::set_streaming_bus(const String &p_name) {
	if (speech_processor) {
		speech_processor->set_streaming_bus(p_name);
	}
}

void Speech::set_error_cancellation_bus(const String &p_name) {
	if (speech_processor) {
		speech_processor->set_error_cancellation_bus(p_name);
	}
}

bool Speech::set_audio_input_stream_player(Node *p_audio_stream) {
	AudioStreamPlayer *player = cast_to<AudioStreamPlayer>(p_audio_stream);
	ERR_FAIL_NULL_V(player, false);
	if (!speech_processor) {
		return false;
	}
	speech_processor->set_audio_input_stream_player(player);
	return true;
}

Dictionary Speech::get_stats() {
	if (speech_processor) {
		return speech_processor->get_stats();
	}
	return Dictionary();
}

Speech::Speech() {
	speech_processor = memnew(SpeechProcessor);
	preallocate_buffers();
}

Speech::~Speech() {
	memdelete(speech_processor);
}

void Speech::add_player_audio(int p_player_id, Node *p_audio_stream_player) {
	if (cast_to<AudioStreamPlayer>(p_audio_stream_player) || cast_to<AudioStreamPlayer2D>(p_audio_stream_player) || cast_to<AudioStreamPlayer3D>(p_audio_stream_player)) {
		if (!player_audio.has(p_player_id)) {
			Ref<AudioStreamGenerator> new_generator;
			new_generator.instantiate();
			new_generator->set_mix_rate(SpeechProcessor::SPEECH_SETTING_VOICE_PACKET_SAMPLE_RATE);
			new_generator->set_buffer_length(BUFFER_DELAY_THRESHOLD);
			playback_ring_buffer_length = calc_playback_ring_buffer_length(new_generator);
			p_audio_stream_player->call("set_stream", new_generator);
			p_audio_stream_player->call("set_bus", "VoiceOutput");
			p_audio_stream_player->call("set_autoplay", true);
			p_audio_stream_player->call("play");
			Ref<SpeechDecoder> speech_decoder = get_speech_decoder();
			Ref<PlaybackStats> pstats = memnew(PlaybackStats);
			pstats->playback_ring_buffer_length = playback_ring_buffer_length;
			pstats->buffer_frame_count = SpeechProcessor::SPEECH_SETTING_BUFFER_FRAME_COUNT;
			Dictionary dict;
			dict["playback_last_skips"] = 0;
			dict["audio_stream_player"] = p_audio_stream_player;
			dict["jitter_buffer"] = Array();
			dict["sequence_id"] = -1;
			dict["last_update"] = OS::get_singleton()->get_ticks_msec();
			dict["packets_received_this_frame"] = 0;
			dict["excess_packets"] = 0;
			dict["speech_decoder"] = speech_decoder;
			dict["playback_stats"] = pstats;
			dict["playback_start_time"] = 0;
			dict["playback_prev_time"] = -1;
			player_audio[p_player_id] = dict;
		} else {
			print_error(vformat("Attempted to duplicate player_audio entry (%s)!", p_player_id));
		}
	}
}

void Speech::vc_debug_print(String p_str) const {
	if (!DEBUG) {
		return;
	}
	print_line(p_str);
}

void Speech::vc_debug_printerr(String p_str) const {
	if (!DEBUG) {
		return;
	}
	print_error(p_str);
}

void Speech::on_received_audio_packet(int p_peer_id, int p_sequence_id, PackedByteArray p_packet) {
	vc_debug_print(
			vformat("Received_audio_packet: peer_id: {%s} sequence_id: {%s}", itos(p_peer_id), itos(p_sequence_id)));
	if (!player_audio.has(p_peer_id)) {
		return;
	}
	Dictionary elem = player_audio[p_peer_id];
	// Detects if no audio packets have been received from this player yet.
	if (int64_t(elem["sequence_id"]) == -1) {
		elem["sequence_id"] = p_sequence_id - 1;
	}

	elem["packets_received_this_frame"] = int64_t(elem["packets_received_this_frame"]) + 1;
	packets_received_this_frame += 1;
	int64_t current_sequence_id = elem["sequence_id"];
	Array jitter_buffer = elem["jitter_buffer"];
	int64_t sequence_id_offset = p_sequence_id - current_sequence_id;
	if (sequence_id_offset > 0) {
		// For skipped buffers, add empty packets.
		int64_t skipped_packets = sequence_id_offset - 1;
		if (skipped_packets) {
			Variant fill_packets;
			for (int32_t _i = 0; _i < skipped_packets; _i++) {
				Dictionary dict;
				dict["packet"] = fill_packets;
				dict["valid"] = false;
				jitter_buffer.push_back(dict);
			}
		}
		{
			// Add the new valid buffer.
			Dictionary dict;
			dict["packet"] = p_packet;
			dict["valid"] = true;
			jitter_buffer.push_back(dict);
		}
		int64_t excess_packet_count = jitter_buffer.size() - MAX_JITTER_BUFFER_SIZE;
		if (excess_packet_count > 0) {
			for (int32_t _i = 0; _i < excess_packet_count; _i++) {
				elem["excess_packets"] = (int64_t)elem["excess_packets"] + 1;
				jitter_buffer.pop_front();
			}
		}
		elem["sequence_id"] = int64_t(elem["sequence_id"]) + sequence_id_offset;
	} else {
		int64_t sequence_id = jitter_buffer.size() - 1 + sequence_id_offset;
		vc_debug_print(vformat("Updating existing sequence_id: %s", itos(sequence_id)));
		if (sequence_id >= 0) {
			// Update the existing buffer.
			Dictionary dict;
			dict["packet"] = p_packet;
			dict["valid"] = true;
			jitter_buffer[sequence_id] = dict;
		} else {
			vc_debug_printerr("Invalid repair sequence_id.");
		}
	}
	elem["jitter_buffer"] = jitter_buffer;
	player_audio[p_peer_id] = elem;
}

Dictionary Speech::get_playback_stats(Dictionary speech_stat_dict) {
	Dictionary stat_dict = speech_stat_dict.duplicate(true);
	stat_dict["capture_get_percent"] = 0;
	stat_dict["capture_discard_percent"] = 0;
	if (double(stat_dict["capture_pushed_s"]) > 0) {
		stat_dict["capture_get_percent"] = 100.0 * double(stat_dict["capture_get_s"]) / double(stat_dict["capture_pushed_s"]);
		stat_dict["capture_discard_percent"] = 100.0 * double(stat_dict["capture_discarded_s"]) / double(stat_dict["capture_pushed_s"]);
	}

	Array keys = player_audio.keys();
	for (int32_t key_i = 0; key_i < keys.size(); key_i++) {
		Variant key = keys[key_i];
		Dictionary elem = player_audio[key];
		Ref<PlaybackStats> playback_stats = elem["playback_stats"];
		if (playback_stats.is_null()) {
			continue;
		}
		Dictionary stats = playback_stats->get_playback_stats();
		stats["playback_total_time"] = (OS::get_singleton()->get_ticks_msec() - int64_t(elem["playback_start_time"])) / double(SpeechProcessor::SPEECH_SETTING_MILLISECONDS_PER_SECOND);
		stats["excess_packets"] = elem["excess_packets"];
		stats["excess_s"] = int64_t(elem["excess_packets"]) * SpeechProcessor::SPEECH_SETTING_PACKET_DELTA_TIME;
		stat_dict[key] = stats;
	}
	return stat_dict;
}

void Speech::remove_player_audio(int p_player_id) {
	if (player_audio.has(p_player_id)) {
		if (player_audio.erase(p_player_id)) {
			return;
		}
	}
	print_error(vformat("Attempted to remove a non-existant player_audio entry (%s)", p_player_id));
}

void Speech::clear_all_player_audio() {
	Array keys = player_audio.keys();
	for (int32_t i = 0; i < keys.size(); i++) {
		Variant key = keys[i];
		Variant element = player_audio[key];
		if (element.get_type() != Variant::DICTIONARY) {
			continue;
		}
		Dictionary elem = element;
		if (!elem.has("audio_stream_player")) {
			continue;
		}
		Dictionary dict = player_audio[key];
		Node *node = cast_to<Node>(dict["audio_stream_player"]);
		if (!node) {
			continue;
		}
		node->queue_free();
	}

	player_audio = Dictionary();
}

void Speech::attempt_to_feed_stream(int p_skip_count, Ref<SpeechDecoder> p_decoder, Node *p_audio_stream_player, Array p_jitter_buffer, Ref<PlaybackStats> p_playback_stats, Dictionary p_player_dict) {
	if (!p_audio_stream_player) {
		return;
	}

	for (int32_t skip_i = 0; skip_i < p_skip_count; skip_i++) {
		p_jitter_buffer.pop_front();
	}
	if (!p_audio_stream_player->has_method("get_stream_playback")) {
		return;
	}

	p_audio_stream_player->call("play", p_audio_stream_player->call("get_playback_position"));

	Ref<AudioStreamGeneratorPlayback> playback = p_audio_stream_player->call("get_stream_playback");
	if (playback.is_null()) {
		return;
	}
	if (int64_t(p_player_dict["playback_last_skips"]) != playback->get_skips()) {
		p_player_dict["playback_prev_time"] = double(p_player_dict["playback_prev_time"]) - SpeechProcessor::SPEECH_SETTING_MILLISECONDS_PER_PACKET;
		p_player_dict["playback_last_skips"] = playback->get_skips();
	}
	int64_t to_fill = playback->get_frames_available();
	int64_t required_packets = 0;
	while (to_fill >= SpeechProcessor::SPEECH_SETTING_BUFFER_FRAME_COUNT) {
		to_fill -= SpeechProcessor::SPEECH_SETTING_BUFFER_FRAME_COUNT;
		required_packets += 1;
	}

	Dictionary last_packet;
	if (p_jitter_buffer.size() > 0) {
		Dictionary jitter_buffer = p_jitter_buffer.back();
		last_packet = jitter_buffer["packet"];
	}
	while (p_jitter_buffer.size() < required_packets) {
		Variant fill_packets;
		Dictionary dict;
		dict["packet"] = fill_packets;
		dict["valid"] = false;
		p_jitter_buffer.push_back(dict);
	}

	for (int32_t _i = 0; _i < required_packets; _i++) {
		Dictionary packet = p_jitter_buffer.pop_front();
		bool packet_pushed = false;
		bool push_result = false;
		PackedByteArray buffer = packet["packet"];
		uncompressed_audio = decompress_buffer(p_decoder, buffer, buffer.size(), uncompressed_audio);
		if (uncompressed_audio.size()) {
			if (uncompressed_audio.size() == SpeechProcessor::SPEECH_SETTING_BUFFER_FRAME_COUNT) {
				push_result = playback->push_buffer(uncompressed_audio);
			}
		}
		packet_pushed = true;
		if (!packet_pushed) {
			push_result = playback->push_buffer(blank_packet);
		}
		if (p_playback_stats.is_null()) {
			continue;
		}
		p_playback_stats->playback_ring_current_size = playback_ring_buffer_length - playback->get_frames_available();
		p_playback_stats->playback_ring_max_size = p_playback_stats->playback_ring_current_size ? p_playback_stats->playback_ring_current_size > p_playback_stats->playback_ring_max_size : p_playback_stats->playback_ring_max_size;
		p_playback_stats->playback_ring_size_sum += 1.0 * p_playback_stats->playback_ring_current_size;
		p_playback_stats->playback_push_buffer_calls += 1;
		if (!packet_pushed) {
			p_playback_stats->playback_blank_push_calls += 1;
		}
		if (push_result) {
			p_playback_stats->playback_pushed_calls += 1;
		} else {
			p_playback_stats->playback_discarded_calls += 1;
		}
		p_playback_stats->playback_skips = 1.0 * double(playback->get_skips());
	}
	if (p_playback_stats.is_valid()) {
		p_playback_stats->jitter_buffer_size_sum += p_jitter_buffer.size();
		p_playback_stats->jitter_buffer_calls += 1;
		p_playback_stats->jitter_buffer_max_size = p_jitter_buffer.size() ? p_jitter_buffer.size() > p_playback_stats->jitter_buffer_max_size : p_playback_stats->jitter_buffer_max_size;
		p_playback_stats->jitter_buffer_current_size = p_jitter_buffer.size();
	}
	// Speed up or slow down the audio stream to mitigate skipping
	if (p_jitter_buffer.size() > JITTER_BUFFER_SPEEDUP) {
		p_audio_stream_player->set_physics_process(STREAM_SPEEDUP_PITCH);
	} else if (p_jitter_buffer.size() < JITTER_BUFFER_SLOWDOWN) {
		p_audio_stream_player->call("set_pitch_scale", STREAM_STANDARD_PITCH);
	}
}

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
