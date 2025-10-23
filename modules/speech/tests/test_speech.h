/**************************************************************************/
/*  test_speech.h                                                         */
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

#include "../playback_stats.h"
#include "../speech.h"
#include "../speech_decoder.h"
#include "../speech_processor.h"
#include "core/math/math_funcs.h"
#include "core/object/ref_counted.h"
#include "core/os/memory.h"
#include "core/variant/array.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"
#include "scene/3d/audio_stream_player_3d.h"
#include "scene/audio/audio_stream_player.h"
#include "scene/main/node.h"
#include "tests/test_macros.h"

namespace TestSpeech {

TEST_CASE("[SceneTree][Speech] Initialization and Default Properties") {
	Speech *speech_node = memnew(Speech);
	REQUIRE(speech_node != nullptr);
	CHECK(Math::is_equal_approx(speech_node->get_buffer_delay_threshold(), 0.1f));
	CHECK(Math::is_equal_approx(speech_node->get_stream_standard_pitch(), 1.0f));
	CHECK(Math::is_equal_approx(speech_node->get_stream_speedup_pitch(), 1.5f));
	CHECK(speech_node->get_max_jitter_buffer_size() == 16);
	CHECK(speech_node->get_jitter_buffer_speedup() == 12);
	CHECK(speech_node->get_jitter_buffer_slowdown() == 6);
	CHECK(speech_node->get_debug() == false);
	REQUIRE(speech_node->get_player_audio().is_empty());

	Dictionary stats = speech_node->get_stats();
	REQUIRE(stats.has("capture_mix_rate"));
	memdelete(speech_node);
}

TEST_CASE("[SceneTree][Speech] Property Getters and Setters") {
	Speech *speech_node = memnew(Speech);

	speech_node->set_buffer_delay_threshold(0.25f);
	CHECK(Math::is_equal_approx(speech_node->get_buffer_delay_threshold(), 0.25f));

	speech_node->set_stream_standard_pitch(1.1f);
	CHECK(Math::is_equal_approx(speech_node->get_stream_standard_pitch(), 1.1f));

	speech_node->set_stream_speedup_pitch(1.75f);
	CHECK(Math::is_equal_approx(speech_node->get_stream_speedup_pitch(), 1.75f));

	speech_node->set_max_jitter_buffer_size(24);
	CHECK(speech_node->get_max_jitter_buffer_size() == 24);

	speech_node->set_jitter_buffer_speedup(10);
	CHECK(speech_node->get_jitter_buffer_speedup() == 10);

	speech_node->set_jitter_buffer_slowdown(5);
	CHECK(speech_node->get_jitter_buffer_slowdown() == 5);

	speech_node->set_debug(true);
	CHECK(speech_node->get_debug() == true);

	memdelete(speech_node);
}

TEST_CASE("[SceneTree][Speech] Decoder Retrieval") {
	Speech *speech_node = memnew(Speech);
	Ref<SpeechDecoder> decoder = speech_node->get_speech_decoder();
	REQUIRE(decoder.is_valid());
	memdelete(speech_node);
}

TEST_CASE("[SceneTree][Speech] Player Audio Management") {
	Speech *speech_node = memnew(Speech);
	AudioStreamPlayer3D *player_node1 = memnew(AudioStreamPlayer3D);
	AudioStreamPlayer3D *player_node2 = memnew(AudioStreamPlayer3D);

	speech_node->add_player_audio(1, player_node1);
	Dictionary player_audio_dict = speech_node->get_player_audio();
	REQUIRE(player_audio_dict.has(1));
	Dictionary player1_data = player_audio_dict[1];
	REQUIRE(player1_data.has("audio_stream_player"));

	speech_node->add_player_audio(2, player_node2);
	player_audio_dict = speech_node->get_player_audio();
	REQUIRE(player_audio_dict.has(1));
	REQUIRE(player_audio_dict.has(2));

	speech_node->remove_player_audio(1);
	player_audio_dict = speech_node->get_player_audio();
	CHECK_FALSE(player_audio_dict.has(1));
	REQUIRE(player_audio_dict.has(2));

	speech_node->clear_all_player_audio();
	player_audio_dict = speech_node->get_player_audio();
	CHECK(player_audio_dict.is_empty());

	memdelete(player_node1);
	memdelete(player_node2);
	memdelete(speech_node);
}

TEST_CASE("[SceneTree][Speech] on_received_audio_packet (Jitter Buffer Basic)") {
	Speech *speech_node = memnew(Speech);
	AudioStreamPlayer3D *dummy_player_node = memnew(AudioStreamPlayer3D);
	speech_node->add_player_audio(123, dummy_player_node);

	PackedByteArray packet_data;
	packet_data.resize(30);
	for (int i = 0; i < packet_data.size(); ++i) {
		packet_data.write[i] = static_cast<uint8_t>(i);
	}

	speech_node->on_received_audio_packet(123, 1, packet_data);

	Dictionary player_audio = speech_node->get_player_audio();
	REQUIRE(player_audio.has(123));
	Dictionary player_data = player_audio[123];
	REQUIRE(player_data.has("jitter_buffer"));
	Array jitter_buffer = player_data["jitter_buffer"];
	REQUIRE_FALSE(jitter_buffer.is_empty());
	CHECK(speech_node->get_packets_received_this_frame() == 2);

	memdelete(dummy_player_node);
	memdelete(speech_node);
}

TEST_CASE("[SceneTree][Speech] copy_and_clear_buffers (Basic Call)") {
	Speech *speech_node = memnew(Speech);
	Array buffers = speech_node->copy_and_clear_buffers();
	CHECK(buffers.is_empty());
	memdelete(speech_node);
}

TEST_CASE("[SceneTree][Speech] Recording Control and Stream Player (No Crash)") {
	Speech *speech_node = memnew(Speech);
	speech_node->start_recording();
	speech_node->end_recording();
	speech_node->set_audio_input_stream_player(nullptr);
	memdelete(speech_node);
}

} // namespace TestSpeech
