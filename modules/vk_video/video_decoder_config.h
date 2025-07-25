/**************************************************************************/
/*  video_decoder_config.h                                               */
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

#ifndef VIDEO_DECODER_CONFIG_H
#define VIDEO_DECODER_CONFIG_H

#include "core/string/ustring.h"
#include "servers/rendering/rendering_device.h"

// Video codec operation flags (extracted from Vulkan Video spec)
enum VideoCodecOperation {
	VIDEO_CODEC_OPERATION_NONE = 0,
	VIDEO_CODEC_OPERATION_DECODE_H264 = 1,
	VIDEO_CODEC_OPERATION_DECODE_H265 = 2,
	VIDEO_CODEC_OPERATION_DECODE_AV1 = 4,
	VIDEO_CODEC_OPERATION_DECODE_VP9 = 8,
	VIDEO_CODEC_OPERATION_ENCODE_H264 = 16,
	VIDEO_CODEC_OPERATION_ENCODE_H265 = 32,
	VIDEO_CODEC_OPERATION_ENCODE_AV1 = 64,
};

// Video codec profiles
enum VkVideoCodecProfile {
	VK_VIDEO_CODEC_PROFILE_UNKNOWN = 0,
	VK_VIDEO_CODEC_PROFILE_H264_BASELINE = 1,
	VK_VIDEO_CODEC_PROFILE_H264_MAIN = 2,
	VK_VIDEO_CODEC_PROFILE_H264_HIGH = 3,
	VK_VIDEO_CODEC_PROFILE_H265_MAIN = 10,
	VK_VIDEO_CODEC_PROFILE_H265_MAIN_10 = 11,
	VK_VIDEO_CODEC_PROFILE_AV1_MAIN = 20,
	VK_VIDEO_CODEC_PROFILE_AV1_HIGH = 21,
	VK_VIDEO_CODEC_PROFILE_AV1_PROFESSIONAL = 22,
	VK_VIDEO_CODEC_PROFILE_VP9_PROFILE_0 = 30,
	VK_VIDEO_CODEC_PROFILE_VP9_PROFILE_1 = 31,
	VK_VIDEO_CODEC_PROFILE_VP9_PROFILE_2 = 32,
	VK_VIDEO_CODEC_PROFILE_VP9_PROFILE_3 = 33,
};

// Video operation type
enum VkVideoOperation {
	VK_VIDEO_OPERATION_DECODE = 0,
	VK_VIDEO_OPERATION_ENCODE = 1,
};

struct VideoDecoderConfig {
	// Basic video properties
	int initial_width = 1920;
	int initial_height = 1080;
	int initial_bitdepth = 8;
	
	// Decoder settings
	VideoCodecOperation codec_operation = VIDEO_CODEC_OPERATION_DECODE_AV1;
	VkVideoCodecProfile codec_profile = VK_VIDEO_CODEC_PROFILE_AV1_MAIN;
	int num_decode_images_in_flight = 8;
	int num_decode_images_to_preallocate = -1; // -1 means allocate maximum
	int num_bitstream_buffers_to_preallocate = 8;
	int decoder_queue_size = 5;
	
	// Hardware settings
	uint32_t device_id = UINT32_MAX; // Use default device
	int queue_id = 0;
	bool enable_hw_load_balancing = false;
	bool select_video_with_compute_queue = false;
	
	// Debug/validation
	bool validate = false;
	bool validate_verbose = false;
	bool verbose = false;
	
	VideoDecoderConfig() = default;
	
	// Helper methods
	bool is_decode_operation() const {
		return (codec_operation & (VIDEO_CODEC_OPERATION_DECODE_H264 | 
								  VIDEO_CODEC_OPERATION_DECODE_H265 | 
								  VIDEO_CODEC_OPERATION_DECODE_AV1 | 
								  VIDEO_CODEC_OPERATION_DECODE_VP9)) != 0;
	}
	
	bool is_encode_operation() const {
		return (codec_operation & (VIDEO_CODEC_OPERATION_ENCODE_H264 | 
								  VIDEO_CODEC_OPERATION_ENCODE_H265 | 
								  VIDEO_CODEC_OPERATION_ENCODE_AV1)) != 0;
	}
	
	bool supports_av1() const {
		return (codec_operation & VIDEO_CODEC_OPERATION_DECODE_AV1) != 0;
	}
	
	String get_codec_name() const {
		switch (codec_operation) {
			case VIDEO_CODEC_OPERATION_DECODE_H264:
				return "H.264";
			case VIDEO_CODEC_OPERATION_DECODE_H265:
				return "H.265/HEVC";
			case VIDEO_CODEC_OPERATION_DECODE_AV1:
				return "AV1";
			case VIDEO_CODEC_OPERATION_DECODE_VP9:
				return "VP9";
			default:
				return "Unknown";
		}
	}
};

#endif // VIDEO_DECODER_CONFIG_H
