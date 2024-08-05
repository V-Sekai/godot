/**************************************************************************/
/*  speech_decoder.cpp                                                    */
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

#include "speech_decoder.h"
#include "speech_processor.h"

int32_t SpeechDecoder::process(const PackedByteArray *p_compressed_buffer,
		PackedByteArray *p_pcm_output_buffer,
		const int p_compressed_buffer_size,
		const int p_pcm_output_buffer_size,
		const int p_buffer_frame_count) {
	*p_pcm_output_buffer->ptrw() = 0;
	if (!decoder) {
		return OPUS_INVALID_STATE;
	}
	opus_int16 *output_buffer_pointer =
			reinterpret_cast<opus_int16 *>(p_pcm_output_buffer->ptrw());
	const unsigned char *opus_buffer_pointer =
			reinterpret_cast<const unsigned char *>(p_compressed_buffer->ptr());

	opus_int32 ret_value =
			opus_decode(decoder, opus_buffer_pointer, p_compressed_buffer_size,
					output_buffer_pointer, p_buffer_frame_count, 0);
	return ret_value;
}

SpeechDecoder::SpeechDecoder() {
	int error = OPUS_INVALID_STATE;
	decoder = opus_decoder_create(
			SpeechProcessor::SPEECH_SETTING_SAMPLE_RATE, SpeechProcessor::SPEECH_SETTING_CHANNEL_COUNT, &error);
	if (error != OPUS_OK) {
		ERR_PRINT("OpusCodec: could not create Opus decoder!");
	}
}

SpeechDecoder::~SpeechDecoder() {
	opus_decoder_destroy(decoder);
}
