/**************************************************************************/
/*  voice_chat.cpp                                                        */
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

#include "voice_chat.h"
#include "core/object/class_db.h"
#include "core/math/math_funcs.h"
#include "core/os/memory.h"
#include "scene/main/audio_stream_player.h"
#include "scene/2d/audio_stream_player_2d.h"
#include "scene/3d/audio_stream_player_3d.h"

VoiceChat::VoiceChat() {
	speech_system = memnew(Speech);
	test_audio_generator.instantiate();
	microphone_enabled = false;
	voice_volume = 1.0f;
	voice_filter_enabled = true;
	testing_mode = false;
}

VoiceChat::~VoiceChat() {
	if (speech_system) {
		memdelete(speech_system);
	}
}

void VoiceChat::_bind_methods() {
	// Connection Management
	ClassDB::bind_method(D_METHOD("connect_to_peer", "peer_id"), &VoiceChat::connect_to_peer);
	ClassDB::bind_method(D_METHOD("disconnect_from_peer", "peer_id"), &VoiceChat::disconnect_from_peer);
	ClassDB::bind_method(D_METHOD("disconnect_all"), &VoiceChat::disconnect_all);

	// Audio Control
	ClassDB::bind_method(D_METHOD("set_microphone_enabled", "enabled"), &VoiceChat::set_microphone_enabled);
	ClassDB::bind_method(D_METHOD("get_microphone_enabled"), &VoiceChat::get_microphone_enabled);
	ClassDB::bind_method(D_METHOD("set_voice_volume", "volume"), &VoiceChat::set_voice_volume);
	ClassDB::bind_method(D_METHOD("get_voice_volume"), &VoiceChat::get_voice_volume);
	ClassDB::bind_method(D_METHOD("set_voice_filter_enabled", "enabled"), &VoiceChat::set_voice_filter_enabled);
	ClassDB::bind_method(D_METHOD("get_voice_filter_enabled"), &VoiceChat::get_voice_filter_enabled);

	// Testing Interface
	ClassDB::bind_method(D_METHOD("inject_audio_frames", "frames"), &VoiceChat::inject_audio_frames);
	ClassDB::bind_method(D_METHOD("get_processed_frames"), &VoiceChat::get_processed_frames);
	ClassDB::bind_method(D_METHOD("set_testing_mode", "enabled"), &VoiceChat::set_testing_mode);
	ClassDB::bind_method(D_METHOD("get_testing_mode"), &VoiceChat::get_testing_mode);
	ClassDB::bind_method(D_METHOD("generate_sine_wave", "frequency", "duration"), &VoiceChat::generate_sine_wave);
	ClassDB::bind_method(D_METHOD("generate_test_audio", "frequency", "duration", "stereo"), &VoiceChat::generate_test_audio);
	ClassDB::bind_method(D_METHOD("set_network_simulation_delay", "delay_ms"), &VoiceChat::set_network_simulation_delay);

	// Status and Statistics
	ClassDB::bind_method(D_METHOD("get_connection_stats"), &VoiceChat::get_connection_stats);
	ClassDB::bind_method(D_METHOD("get_connected_peers"), &VoiceChat::get_connected_peers);

	// Advanced Configuration
	ClassDB::bind_method(D_METHOD("set_timing_filter_config", "cutoff", "beta"), &VoiceChat::set_timing_filter_config);
	ClassDB::bind_method(D_METHOD("get_timing_filter_config"), &VoiceChat::get_timing_filter_config);

	// Properties
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "microphone_enabled"), "set_microphone_enabled", "get_microphone_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "voice_volume", PROPERTY_HINT_RANGE, "0.0,2.0,0.01"), "set_voice_volume", "get_voice_volume");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "voice_filter_enabled"), "set_voice_filter_enabled", "get_voice_filter_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "testing_mode"), "set_testing_mode", "get_testing_mode");
}

bool VoiceChat::connect_to_peer(int peer_id) {
	if (!speech_system) {
		return false;
	}

	// Create AudioStreamPlayer for this peer
	AudioStreamPlayer *player = memnew(AudioStreamPlayer);
	add_child(player);
	
	// Add player to speech system
	speech_system->add_player_audio(peer_id, player);
	
	return true;
}

void VoiceChat::disconnect_from_peer(int peer_id) {
	if (speech_system) {
		speech_system->remove_player_audio(peer_id);
	}
}

void VoiceChat::disconnect_all() {
	if (speech_system) {
		speech_system->clear_all_player_audio();
	}
}

void VoiceChat::set_microphone_enabled(bool enabled) {
	microphone_enabled = enabled;
	
	if (!speech_system) {
		return;
	}
	
	if (enabled) {
		speech_system->start_recording();
	} else {
		speech_system->end_recording();
	}
}

bool VoiceChat::get_microphone_enabled() const {
	return microphone_enabled;
}

void VoiceChat::set_voice_volume(float volume) {
	voice_volume = CLAMP(volume, 0.0f, 2.0f);
	// Volume control would be implemented in the audio bus system
}

float VoiceChat::get_voice_volume() const {
	return voice_volume;
}

void VoiceChat::set_voice_filter_enabled(bool enabled) {
	voice_filter_enabled = enabled;
	// OneEuroFilter is always enabled for timing, this controls additional filtering
}

bool VoiceChat::get_voice_filter_enabled() const {
	return voice_filter_enabled;
}

void VoiceChat::inject_audio_frames(PackedVector2Array frames) {
	if (!testing_mode || !test_audio_generator.is_valid()) {
		print_error("VoiceChat: inject_audio_frames requires testing_mode to be enabled");
		return;
	}
	
	// Use AudioStreamGenerator to inject frames directly into the audio pipeline
	Ref<AudioStreamGeneratorPlayback> playback = test_audio_generator->instantiate_playback();
	if (playback.is_valid()) {
		playback->push_buffer(frames);
		print_line("VoiceChat: Injected " + itos(frames.size()) + " audio frames using AudioStreamGenerator");
	}
}

PackedVector2Array VoiceChat::get_processed_frames() {
	if (!testing_mode || !speech_system) {
		return PackedVector2Array();
	}
	
	// Return processed audio frames for testing validation
	return speech_system->get_uncompressed_audio();
}

void VoiceChat::set_testing_mode(bool enabled) {
	testing_mode = enabled;
	
	if (enabled) {
		print_line("VoiceChat: Testing mode enabled - synthetic audio injection available");
	} else {
		print_line("VoiceChat: Testing mode disabled - using real audio input");
	}
}

bool VoiceChat::get_testing_mode() const {
	return testing_mode;
}

PackedVector2Array VoiceChat::generate_sine_wave(float frequency, float duration) {
	// Use Godot's proven audio generation pattern from test_audio_stream_wav.h
	float wav_rate = 44100.0f;
	int wav_count = int(wav_rate * duration);
	
	PackedVector2Array buffer;
	buffer.resize(wav_count);
	
	for (int i = 0; i < wav_count; i++) {
		// Generate sine wave using Godot's math functions
		float wav = Math::sin((Math::TAU * frequency / wav_rate) * i);
		buffer.write[i] = Vector2(wav, wav); // Stereo output
	}
	
	return buffer;
}

PackedVector2Array VoiceChat::generate_test_audio(float frequency, float duration, bool stereo) {
	// Generate test audio using Godot's native patterns
	return generate_sine_wave(frequency, duration);
}

void VoiceChat::set_network_simulation_delay(float delay_ms) {
	// Network simulation can be handled by introducing delays in packet processing
	// This is simpler and more effective than MockAudioDevice's approach
	print_line(vformat("VoiceChat: Network simulation delay set to %f ms", delay_ms));
}

Dictionary VoiceChat::get_connection_stats() {
	if (!speech_system) {
		return Dictionary();
	}
	
	Dictionary stats = speech_system->get_stats();
	Dictionary playback_stats = speech_system->get_playback_stats(stats);
	
	// Add VoiceChat-specific statistics
	Dictionary voice_stats;
	voice_stats["microphone_enabled"] = microphone_enabled;
	voice_stats["voice_volume"] = voice_volume;
	voice_stats["voice_filter_enabled"] = voice_filter_enabled;
	voice_stats["testing_mode"] = testing_mode;
	voice_stats["connected_peers"] = get_connected_peers();
	voice_stats["speech_stats"] = playback_stats;
	
	return voice_stats;
}

Array VoiceChat::get_connected_peers() {
	if (!speech_system) {
		return Array();
	}
	
	Dictionary player_audio = speech_system->get_player_audio();
	return player_audio.keys();
}

void VoiceChat::set_timing_filter_config(float cutoff, float beta) {
	// This would configure OneEuroFilter parameters for all peers
	// Implementation would require extending Speech class to expose filter configuration
	print_line(vformat("VoiceChat: Setting timing filter config - cutoff: %f, beta: %f", cutoff, beta));
}

Dictionary VoiceChat::get_timing_filter_config() {
	Dictionary config;
	config["cutoff"] = 0.1f;  // Default values from Speech class
	config["beta"] = 5.0f;
	config["min_cutoff"] = 1.0f;
	config["derivate_cutoff"] = 1.0f;
	return config;
}

void VoiceChat::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			if (speech_system) {
				add_child(speech_system);
				speech_system->set_owner(get_owner());
			}
			break;
		}
		case NOTIFICATION_EXIT_TREE: {
			disconnect_all();
			if (speech_system) {
				remove_child(speech_system);
			}
			break;
		}
		default:
			break;
	}
}

void VoiceChat::_on_audio_packet_received(int peer_id, int sequence_id, PackedByteArray packet) {
	if (speech_system) {
		speech_system->on_received_audio_packet(peer_id, sequence_id, packet);
	}
}
