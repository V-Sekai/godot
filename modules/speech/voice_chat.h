/**************************************************************************/
/*  voice_chat.h                                                          */
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

#include "core/object/class_db.h"
#include "core/variant/array.h"
#include "core/variant/dictionary.h"
#include "core/variant/packed_array.h"
#include "scene/main/node.h"

#include "speech.h"
#include "speech_processor.h"
#include "speech_decoder.h"
#include "servers/audio/effects/audio_stream_generator.h"

/**
 * VoiceChat - Unified API for V-Sekai VOIP
 * 
 * Replaces the complex Speech/SpeechProcessor/SpeechDecoder system with a single,
 * clean interface. Integrates OneEuroFilter timing for stable audio without artifacts.
 * 
 * Key Features:
 * - Single-class API for all VOIP operations
 * - OneEuroFilter-based timing (no pitch manipulation)
 * - Testing interface for synthetic audio injection
 * - Automatic peer management
 * - Clean connection lifecycle
 */
class VoiceChat : public Node {
	GDCLASS(VoiceChat, Node);

private:
	Speech *speech_system;
	bool microphone_enabled;
	float voice_volume;
	bool voice_filter_enabled;
	
	// Testing infrastructure using Godot's native AudioStreamGenerator
	Ref<AudioStreamGenerator> test_audio_generator;
	bool testing_mode;

protected:
	static void _bind_methods();

public:
	VoiceChat();
	~VoiceChat();

	// Connection Management
	bool connect_to_peer(int peer_id);
	void disconnect_from_peer(int peer_id);
	void disconnect_all();

	// Audio Control
	void set_microphone_enabled(bool enabled);
	bool get_microphone_enabled() const;
	
	void set_voice_volume(float volume);
	float get_voice_volume() const;
	
	void set_voice_filter_enabled(bool enabled);
	bool get_voice_filter_enabled() const;

	// Testing Interface - Direct frame injection for synthetic testing using AudioStreamGenerator
	void inject_audio_frames(PackedVector2Array frames);
	PackedVector2Array get_processed_frames();
	
	// Enable/disable testing mode with AudioStreamGenerator
	void set_testing_mode(bool enabled);
	bool get_testing_mode() const;
	
	// Generate synthetic test audio using Godot's proven patterns
	PackedVector2Array generate_sine_wave(float frequency, float duration);
	PackedVector2Array generate_test_audio(float frequency, float duration, bool stereo = false);
	void set_network_simulation_delay(float delay_ms);

	// Status and Statistics
	Dictionary get_connection_stats();
	Array get_connected_peers();
	
	// Advanced configuration (OneEuroFilter parameters)
	void set_timing_filter_config(float cutoff, float beta);
	Dictionary get_timing_filter_config();

	// Internal methods
	void _notification(int p_what);
	void _on_audio_packet_received(int peer_id, int sequence_id, PackedByteArray packet);
};
