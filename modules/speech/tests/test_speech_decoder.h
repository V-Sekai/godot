/**************************************************************************/
/*  test_speech_decoder.h                                                 */
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
#include "core/object/ref_counted.h"
#include "core/os/memory.h"
#include "core/variant/variant.h"
#include "tests/test_macros.h"

namespace TestSpeechDecoder {

TEST_CASE("[SpeechDecoder] Initialization") {
	Ref<SpeechDecoder> decoder = memnew(SpeechDecoder);
	REQUIRE(decoder.is_valid());
}

TEST_CASE("[SpeechDecoder] Process Empty or Invalid Input") {
	Ref<SpeechDecoder> decoder = memnew(SpeechDecoder);
	REQUIRE(decoder.is_valid());

	PackedByteArray compressed_buffer;
	PackedByteArray pcm_output_buffer;
	pcm_output_buffer.resize(SpeechProcessor::SPEECH_SETTING_PCM_BUFFER_SIZE);
	pcm_output_buffer.fill(0);

	int32_t result = decoder->process(
			&compressed_buffer,
			&pcm_output_buffer,
			0,
			SpeechProcessor::SPEECH_SETTING_PCM_BUFFER_SIZE,
			SpeechProcessor::SPEECH_SETTING_BUFFER_FRAME_COUNT);
	PackedByteArray small_pcm_output_buffer;
	small_pcm_output_buffer.resize(SpeechProcessor::SPEECH_SETTING_PCM_BUFFER_SIZE / 4);
	compressed_buffer.resize(10);
	compressed_buffer.fill(1);

	result = decoder->process(
			&compressed_buffer,
			&small_pcm_output_buffer,
			compressed_buffer.size(),
			small_pcm_output_buffer.size(),
			SpeechProcessor::SPEECH_SETTING_BUFFER_FRAME_COUNT);
	CHECK_MESSAGE(result == OPUS_INVALID_PACKET, "Processing with too small output buffer should return OPUS_INVALID_PACKET.");
}

} // namespace TestSpeechDecoder
