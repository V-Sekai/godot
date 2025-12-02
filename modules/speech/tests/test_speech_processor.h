/**************************************************************************/
/*  test_speech_processor.h                                               */
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

#include "../speech_decoder.h"
#include "../speech_processor.h"
#include "core/math/math_funcs.h"
#include "core/object/ref_counted.h"
#include "core/os/memory.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"
#include "scene/main/node.h"
#include "tests/test_macros.h"

namespace TestSpeechProcessor {

TEST_CASE("[SceneTree][SpeechProcessor] Initialization and Basic Getters") {
	SpeechProcessor *processor = memnew(SpeechProcessor);
	REQUIRE(processor != nullptr);
	Dictionary stats = processor->get_stats();
	REQUIRE(stats.has("capture_mix_rate"));

	Ref<SpeechDecoder> decoder = processor->get_speech_decoder();
	REQUIRE(decoder.is_valid());

	memdelete(processor);
}

TEST_CASE("[SceneTree][SpeechProcessor] Encode and Compress Operations") {
	SpeechProcessor *processor = memnew(SpeechProcessor);
	PackedByteArray pcm_buffer;
	pcm_buffer.resize(SpeechProcessor::SPEECH_SETTING_PCM_BUFFER_SIZE);
	pcm_buffer.fill(0);

	PackedByteArray internal_output_compressed_buffer;
	internal_output_compressed_buffer.resize(SpeechProcessor::SPEECH_SETTING_INTERNAL_BUFFER_SIZE);
	int encoded_bytes = processor->encode_buffer(&pcm_buffer, &internal_output_compressed_buffer);
	CHECK(encoded_bytes > 0);
	CHECK(encoded_bytes <= SpeechProcessor::SPEECH_SETTING_INTERNAL_BUFFER_SIZE);

	Dictionary output_dict;
	PackedByteArray dict_output_array;
	dict_output_array.resize(SpeechProcessor::SPEECH_SETTING_INTERNAL_BUFFER_SIZE);
	output_dict["byte_array"] = dict_output_array;

	Dictionary result_dict = processor->compress_buffer(pcm_buffer, output_dict);
	REQUIRE(result_dict.has("byte_array"));
	PackedByteArray compressed_result_from_dict = result_dict["byte_array"];
	SpeechProcessor::CompressedSpeechBuffer csb_check;
	csb_check.compressed_byte_array = &compressed_result_from_dict;
	bool success_internal = processor->compress_buffer_internal(&pcm_buffer, &csb_check);
	REQUIRE(success_internal);
	CHECK(csb_check.buffer_size == 67); // FIXME: Size should be consistent

	memdelete(processor);
}

TEST_CASE("[SceneTree][SpeechProcessor] Decode and Decompress Operations") {
	SpeechProcessor *processor = memnew(SpeechProcessor);
	Ref<SpeechDecoder> decoder = processor->get_speech_decoder();
	REQUIRE(decoder.is_valid());

	PackedByteArray pcm_silence_buffer;
	pcm_silence_buffer.resize(SpeechProcessor::SPEECH_SETTING_PCM_BUFFER_SIZE);
	pcm_silence_buffer.fill(0);

	PackedByteArray compressed_packet;
	compressed_packet.resize(SpeechProcessor::SPEECH_SETTING_INTERNAL_BUFFER_SIZE);
	int compressed_size = processor->encode_buffer(&pcm_silence_buffer, &compressed_packet);
	REQUIRE(compressed_size > 0);
	compressed_packet.resize(compressed_size);

	PackedVector2Array internal_output_vec2_array;
	internal_output_vec2_array.resize(SpeechProcessor::SPEECH_SETTING_BUFFER_FRAME_COUNT);
	bool success_internal_decompress = processor->decompress_buffer_internal(decoder.ptr(), &compressed_packet, compressed_size, &internal_output_vec2_array);
	REQUIRE(success_internal_decompress);
	REQUIRE(internal_output_vec2_array.size() == SpeechProcessor::SPEECH_SETTING_BUFFER_FRAME_COUNT);
	// bool all_zero = true;
	// for (int i = 0; i < internal_output_vec2_array.size(); ++i) {
	// 	if (!Math::is_zero_approx(internal_output_vec2_array[i].x) || !Math::is_zero_approx(internal_output_vec2_array[i].y)) {
	// 		all_zero = false;
	// 		break;
	// 	}
	// }
	// FIXME: 2025-05-17 CHECK_MESSAGE(all_zero, "Decompressed silence should be (near) zero.");

	memdelete(processor);
}

TEST_CASE("[SceneTree][SpeechProcessor] _16_pcm_mono_to_real_stereo Conversion") {
	SpeechProcessor processor_instance;
	PackedByteArray pcm_mono_buffer;
	const int frame_count = 10;
	pcm_mono_buffer.resize(frame_count * sizeof(int16_t));
	int16_t *pcm_ptr = reinterpret_cast<int16_t *>(pcm_mono_buffer.ptrw());
	for (int i = 0; i < frame_count; ++i) {
		pcm_ptr[i] = static_cast<int16_t>(i * 1000 - 5000);
	}

	PackedVector2Array stereo_output_buffer;
	stereo_output_buffer.resize(frame_count);

	bool conversion_success = processor_instance._16_pcm_mono_to_real_stereo(&pcm_mono_buffer, &stereo_output_buffer);
	REQUIRE(conversion_success);
	REQUIRE(stereo_output_buffer.size() == frame_count);
	for (int i = 0; i < frame_count; ++i) {
		float expected_float_val = static_cast<float>(pcm_ptr[i]) / 32768.0f;
		CHECK_MESSAGE(Math::is_equal_approx(static_cast<float>(stereo_output_buffer[i].x), expected_float_val), vformat("Mono to stereo L channel mismatch at index %s", itos(i)));
		CHECK_MESSAGE(Math::is_equal_approx(static_cast<float>(stereo_output_buffer[i].y), expected_float_val), vformat("Mono to stereo R channel mismatch at index %s", itos(i)));
	}
}

TEST_CASE("[SceneTree][SpeechProcessor] Set Audio Input Stream Player (Error Handling)") {
	SpeechProcessor *processor = memnew(SpeechProcessor);
	CHECK_FALSE(processor->set_audio_input_stream_player(nullptr));
	Node *generic_node = memnew(Node);
	CHECK_FALSE(processor->set_audio_input_stream_player(generic_node));
	memdelete(generic_node);
	memdelete(processor);
}

TEST_CASE("[SpeechProcessor] Test Direct Audio Processing via test_process_mono_audio_frames") {
	SpeechProcessor *processor = memnew(SpeechProcessor);
	REQUIRE(processor != nullptr);

	PackedFloat32Array mono_input_frames;
	const uint32_t input_sample_rate = 16000;
	const float duration_seconds = 0.1f;
	const int num_input_frames = static_cast<int>(input_sample_rate * duration_seconds);
	mono_input_frames.resize(num_input_frames);
	for (int i = 0; i < num_input_frames; ++i) {
		mono_input_frames.write[i] = Math::sin(2.0 * Math::PI * 440.0 * static_cast<double>(i) / input_sample_rate) * 0.5f; // A4 note
	}

	Vector<PackedByteArray> received_packets;

	processor->register_speech_processed([&](SpeechProcessor::SpeechInput *p_input) {
		REQUIRE(p_input != nullptr);
		REQUIRE(p_input->pcm_byte_array != nullptr);
		received_packets.push_back(*p_input->pcm_byte_array);
	});

	processor->test_process_mono_audio_frames(mono_input_frames, input_sample_rate);

	float expected_total_resampled_frames = static_cast<float>(num_input_frames) * (static_cast<float>(SpeechProcessor::SPEECH_SETTING_VOICE_SAMPLE_RATE) / input_sample_rate);
	int expected_num_packets = static_cast<int>(Math::floor(expected_total_resampled_frames / static_cast<float>(SpeechProcessor::SPEECH_SETTING_BUFFER_FRAME_COUNT)));

	bool is_packet_count_acceptable = Math::abs((int)received_packets.size() - expected_num_packets) <= 1;
	CHECK_MESSAGE(is_packet_count_acceptable,
			vformat("Expected around %d packets (+/-1), but received %d.", expected_num_packets, received_packets.size()));

	if (!received_packets.is_empty()) {
		for (const PackedByteArray &packet : received_packets) {
			CHECK(packet.size() == SpeechProcessor::SPEECH_SETTING_PCM_BUFFER_SIZE);
		}
	} else if (expected_num_packets > 0) {
		FAIL_CHECK(vformat("Expected %d packets, but received none.", expected_num_packets));
	}

	memdelete(processor);
}

} // namespace TestSpeechProcessor
