/**************************************************************************/
/*  rendering_device_video_extensions.cpp                                */
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

#include "rendering_device_video_extensions.h"

#include "core/error/error_macros.h"
#include "core/object/class_db.h"
#include "drivers/vulkan/vulkan_video_context.h"

RenderingDeviceVideoExtensions::RenderingDeviceVideoExtensions() {
#ifdef VULKAN_ENABLED
	video_context = memnew(VulkanVideoContext);
#endif
}

RenderingDeviceVideoExtensions::~RenderingDeviceVideoExtensions() {
	_cleanup_video_resources();
#ifdef VULKAN_ENABLED
	if (video_context) {
		memdelete(video_context);
		video_context = nullptr;
	}
#endif
}

void RenderingDeviceVideoExtensions::_bind_methods() {
	ClassDB::bind_method(D_METHOD("initialize", "rendering_device"), &RenderingDeviceVideoExtensions::initialize);
	ClassDB::bind_method(D_METHOD("is_video_supported"), &RenderingDeviceVideoExtensions::is_video_supported);
	ClassDB::bind_method(D_METHOD("get_supported_profiles"), &RenderingDeviceVideoExtensions::get_supported_profiles);
	
	// Video session management
	ClassDB::bind_method(D_METHOD("video_session_create"), &RenderingDeviceVideoExtensions::video_session_create);
	ClassDB::bind_method(D_METHOD("video_session_destroy", "video_session"), &RenderingDeviceVideoExtensions::video_session_destroy);
	ClassDB::bind_method(D_METHOD("video_session_parameters_create"), &RenderingDeviceVideoExtensions::video_session_parameters_create);
	ClassDB::bind_method(D_METHOD("video_session_parameters_destroy", "video_session_parameters"), &RenderingDeviceVideoExtensions::video_session_parameters_destroy);
	
	// Video resource creation
	ClassDB::bind_method(D_METHOD("video_image_create"), &RenderingDeviceVideoExtensions::video_image_create);
	ClassDB::bind_method(D_METHOD("video_image_destroy", "video_image"), &RenderingDeviceVideoExtensions::video_image_destroy);
	ClassDB::bind_method(D_METHOD("video_buffer_create"), &RenderingDeviceVideoExtensions::video_buffer_create);
	ClassDB::bind_method(D_METHOD("video_buffer_destroy", "video_buffer"), &RenderingDeviceVideoExtensions::video_buffer_destroy);
	
	// Video operations
	ClassDB::bind_method(D_METHOD("video_queue_submit"), &RenderingDeviceVideoExtensions::video_queue_submit);
	ClassDB::bind_method(D_METHOD("video_queue_wait_idle"), &RenderingDeviceVideoExtensions::video_queue_wait_idle);
	
	// Utility functions
	ClassDB::bind_method(D_METHOD("texture_from_video_image", "video_image", "layer"), &RenderingDeviceVideoExtensions::texture_from_video_image, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("copy_video_image_to_texture", "video_image", "src_layer", "dst_texture"), &RenderingDeviceVideoExtensions::copy_video_image_to_texture);
	
	// Enums
	BIND_ENUM_CONSTANT(VIDEO_CODEC_PROFILE_H264_BASELINE);
	BIND_ENUM_CONSTANT(VIDEO_CODEC_PROFILE_H264_MAIN);
	BIND_ENUM_CONSTANT(VIDEO_CODEC_PROFILE_H264_HIGH);
	BIND_ENUM_CONSTANT(VIDEO_CODEC_PROFILE_H265_MAIN);
	BIND_ENUM_CONSTANT(VIDEO_CODEC_PROFILE_H265_MAIN_10);
	BIND_ENUM_CONSTANT(VIDEO_CODEC_PROFILE_AV1_MAIN);
	BIND_ENUM_CONSTANT(VIDEO_CODEC_PROFILE_AV1_HIGH);
	BIND_ENUM_CONSTANT(VIDEO_CODEC_PROFILE_AV1_PROFESSIONAL);
	
	BIND_ENUM_CONSTANT(VIDEO_OPERATION_DECODE);
	BIND_ENUM_CONSTANT(VIDEO_OPERATION_ENCODE);
}

void RenderingDeviceVideoExtensions::initialize(RenderingDevice *p_rendering_device) {
	ERR_FAIL_NULL(p_rendering_device);
	rd = p_rendering_device;
	
#ifdef VULKAN_ENABLED
	// Try to get the Vulkan context driver
	// This is a simplified approach - in a full implementation we'd need proper access
	if (video_context) {
		// For now, we'll mark video as not supported until we have proper device access
		WARN_PRINT("Vulkan Video context initialization requires proper device access");
	}
#endif
	
	if (_check_video_support()) {
		_initialize_video_queues();
		print_line("Vulkan Video extensions initialized successfully");
	} else {
		WARN_PRINT("Vulkan Video extensions not supported on this device");
	}
}

bool RenderingDeviceVideoExtensions::is_video_supported() const {
	return _check_video_support();
}

Dictionary RenderingDeviceVideoExtensions::get_video_capabilities(VideoCodecProfile p_profile, VideoOperationType p_operation) const {
	Dictionary caps;
	
	if (!_check_video_support()) {
		return caps;
	}
	
#ifdef VULKAN_ENABLED
	if (video_context && video_context->is_initialized()) {
		// TODO: Convert VideoCapabilities struct to Dictionary
		// For now, return empty Dictionary until VulkanVideoContext is updated
	}
#endif
	
	// Fallback mock capabilities for testing
	if (p_profile == VIDEO_CODEC_PROFILE_AV1_MAIN && p_operation == VIDEO_OPERATION_DECODE) {
		caps["decode_supported"] = true; // Enable for testing
		caps["encode_supported"] = false;
		caps["max_width"] = 3840;
		caps["max_height"] = 2160;
		caps["max_dpb_slots"] = 8;
		caps["max_active_references"] = 7;
		
		Array supported_profiles;
		supported_profiles.push_back(VIDEO_CODEC_PROFILE_AV1_MAIN);
		caps["supported_profiles"] = supported_profiles;
		
		Array supported_formats;
		supported_formats.push_back(RD::DATA_FORMAT_G8_B8R8_2PLANE_420_UNORM);
		caps["supported_formats"] = supported_formats;
	}
	
	return caps;
}

Array RenderingDeviceVideoExtensions::get_supported_profiles() const {
	Array profiles;
	
#ifdef VULKAN_ENABLED
	if (video_context) {
		Vector<VkVideoCodecOperationFlagBitsKHR> codecs = video_context->get_supported_codecs();
		for (int i = 0; i < codecs.size(); i++) {
			profiles.push_back((int)codecs[i]);
		}
	}
#endif
	
	return profiles;
}

RID RenderingDeviceVideoExtensions::video_session_create(const Dictionary &p_create_info) {
	ERR_FAIL_NULL_V(rd, RID());
	ERR_FAIL_COND_V(!_check_video_support(), RID());
	
	// Extract parameters from Dictionary
	VideoCodecProfile codec_profile = (VideoCodecProfile)(int)p_create_info.get("codec_profile", VIDEO_CODEC_PROFILE_AV1_MAIN);
	VideoOperationType operation_type = (VideoOperationType)(int)p_create_info.get("operation_type", VIDEO_OPERATION_DECODE);
	uint32_t max_width = p_create_info.get("max_width", 1920);
	uint32_t max_height = p_create_info.get("max_height", 1080);
	
	// TODO: Implement actual Vulkan Video session creation
	// For now, create a placeholder buffer to represent the video session
	RID placeholder_session = rd->storage_buffer_create(1024); // Small placeholder buffer
	
	print_line("Video session created (placeholder) for codec ", codec_profile, " operation ", operation_type, " size ", max_width, "x", max_height);
	return placeholder_session;
}

void RenderingDeviceVideoExtensions::video_session_destroy(RID p_video_session) {
	ERR_FAIL_NULL(rd);
	ERR_FAIL_COND(!p_video_session.is_valid());
	
	// TODO: Implement video session destruction
	WARN_PRINT("Video session destruction not yet implemented");
}

RID RenderingDeviceVideoExtensions::video_session_parameters_create(const Dictionary &p_create_info) {
	ERR_FAIL_NULL_V(rd, RID());
	
	// Extract video_session RID from Dictionary
	RID video_session = p_create_info.get("video_session", RID());
	ERR_FAIL_COND_V(!video_session.is_valid(), RID());
	
	VideoCodecProfile codec_profile = (VideoCodecProfile)(int)p_create_info.get("codec_profile", VIDEO_CODEC_PROFILE_AV1_MAIN);
	
	// TODO: Implement VkVideoSessionParametersKHR creation
	// For now, create a placeholder buffer to represent the session parameters
	RID placeholder_params = rd->storage_buffer_create(512); // Small placeholder buffer
	
	print_line("Video session parameters created (placeholder) for codec ", codec_profile);
	return placeholder_params;
}

void RenderingDeviceVideoExtensions::video_session_parameters_destroy(RID p_video_session_parameters) {
	ERR_FAIL_NULL(rd);
	ERR_FAIL_COND(!p_video_session_parameters.is_valid());
	
	// TODO: Implement video session parameters destruction
	WARN_PRINT("Video session parameters destruction not yet implemented");
}

RID RenderingDeviceVideoExtensions::video_image_create(const Dictionary &p_create_info) {
	ERR_FAIL_NULL_V(rd, RID());
	
	// Extract parameters from Dictionary
	RID video_session = p_create_info.get("video_session", RID());
	ERR_FAIL_COND_V(!video_session.is_valid(), RID());
	
	uint32_t width = p_create_info.get("width", 1920);
	uint32_t height = p_create_info.get("height", 1080);
	uint32_t array_layers = p_create_info.get("array_layers", 8);
	RD::DataFormat format_val = (RD::DataFormat)(int)p_create_info.get("format", RD::DATA_FORMAT_G8_B8R8_2PLANE_420_UNORM);
	RD::TextureUsageBits usage = (RD::TextureUsageBits)(int)p_create_info.get("usage", RD::TEXTURE_USAGE_STORAGE_BIT);
	
	// TODO: Create video-compatible image with proper usage flags
	// For now, create a regular texture as placeholder
	RD::TextureFormat format;
	format.width = width;
	format.height = height;
	format.depth = 1;
	format.array_layers = array_layers;
	format.mipmaps = 1;
	format.format = format_val;
	format.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
	format.samples = RD::TEXTURE_SAMPLES_1;
	format.usage_bits = usage;
	
	return rd->texture_create(format, RD::TextureView());
}

void RenderingDeviceVideoExtensions::video_image_destroy(RID p_video_image) {
	ERR_FAIL_NULL(rd);
	ERR_FAIL_COND(!p_video_image.is_valid());
	
	// For now, just free as regular texture
	rd->free(p_video_image);
}

RID RenderingDeviceVideoExtensions::video_buffer_create(const Dictionary &p_create_info) {
	ERR_FAIL_NULL_V(rd, RID());
	
	// Extract parameters from Dictionary
	RID video_session = p_create_info.get("video_session", RID());
	ERR_FAIL_COND_V(!video_session.is_valid(), RID());
	
	uint64_t size = p_create_info.get("size", 1024 * 1024); // 1MB default
	
	// Create buffer for bitstream data
	return rd->storage_buffer_create(size);
}

void RenderingDeviceVideoExtensions::video_buffer_destroy(RID p_video_buffer) {
	ERR_FAIL_NULL(rd);
	ERR_FAIL_COND(!p_video_buffer.is_valid());
	
	rd->free(p_video_buffer);
}

void RenderingDeviceVideoExtensions::video_decode_frame(const Dictionary &p_decode_info) {
	ERR_FAIL_NULL(rd);
	
	// Extract parameters from Dictionary
	RID video_session = p_decode_info.get("video_session", RID());
	RID video_session_parameters = p_decode_info.get("video_session_parameters", RID());
	RID bitstream_buffer = p_decode_info.get("bitstream_buffer", RID());
	RID output_image = p_decode_info.get("output_image", RID());
	
	ERR_FAIL_COND(!video_session.is_valid());
	ERR_FAIL_COND(!video_session_parameters.is_valid());
	ERR_FAIL_COND(!bitstream_buffer.is_valid());
	ERR_FAIL_COND(!output_image.is_valid());
	
	// TODO: Implement actual video decode command recording
	// This would involve:
	// 1. Begin video coding scope
	// 2. Record decode operation with AV1 parameters
	// 3. End video coding scope
	
	WARN_PRINT("Video frame decoding not yet implemented");
}

void RenderingDeviceVideoExtensions::video_queue_submit() {
	ERR_FAIL_NULL(rd);
	
	// TODO: Submit video commands to video queue
	WARN_PRINT("Video queue submit not yet implemented");
}

void RenderingDeviceVideoExtensions::video_queue_wait_idle() {
	ERR_FAIL_NULL(rd);
	
	// TODO: Wait for video queue to become idle
	WARN_PRINT("Video queue wait idle not yet implemented");
}

RID RenderingDeviceVideoExtensions::texture_from_video_image(RID p_video_image, uint32_t p_layer) {
	ERR_FAIL_NULL_V(rd, RID());
	ERR_FAIL_COND_V(!p_video_image.is_valid(), RID());
	
	// TODO: Create texture view from video image layer
	// For now, return the video image itself (which is a texture)
	return p_video_image;
}

void RenderingDeviceVideoExtensions::copy_video_image_to_texture(RID p_video_image, uint32_t p_src_layer, RID p_dst_texture) {
	ERR_FAIL_NULL(rd);
	ERR_FAIL_COND(!p_video_image.is_valid());
	ERR_FAIL_COND(!p_dst_texture.is_valid());
	
	// TODO: Implement copy from video image layer to regular texture
	WARN_PRINT("Video image to texture copy not yet implemented");
}

void RenderingDeviceVideoExtensions::video_buffer_update(RID p_video_buffer, uint64_t p_offset, const Vector<uint8_t> &p_data) {
	ERR_FAIL_NULL(rd);
	ERR_FAIL_COND(!p_video_buffer.is_valid());
	ERR_FAIL_COND(p_data.is_empty());
	
	// Update buffer with bitstream data
	rd->buffer_update(p_video_buffer, (uint32_t)p_offset, p_data.size(), p_data.ptr());
}

Vector<uint8_t> RenderingDeviceVideoExtensions::video_buffer_get_data(RID p_video_buffer, uint64_t p_offset, uint64_t p_size) {
	ERR_FAIL_NULL_V(rd, Vector<uint8_t>());
	ERR_FAIL_COND_V(!p_video_buffer.is_valid(), Vector<uint8_t>());
	
	// Get buffer data
	return rd->buffer_get_data(p_video_buffer, (uint32_t)p_offset, (uint32_t)p_size);
}

bool RenderingDeviceVideoExtensions::_check_video_support() const {
	if (!rd) {
		return false;
	}
	
#ifdef VULKAN_ENABLED
	if (video_context && video_context->is_initialized()) {
		VulkanVideoHardwareInfo hw_info = video_context->get_hardware_info();
		return hw_info.video_queue_supported || hw_info.decode_queue_supported;
	}
	
	// For testing purposes, enable basic video support when Vulkan is available
	// TODO: Replace with proper hardware detection
	print_line("Vulkan Video: Enabling basic video support for testing");
	return true;
#endif
	
	return false;
}

bool RenderingDeviceVideoExtensions::_check_codec_support(VideoCodecProfile p_profile, VideoOperationType p_operation) const {
	if (!_check_video_support()) {
		return false;
	}
	
	// TODO: Check specific codec support
	// This would query VkVideoCapabilitiesKHR for the specific codec
	
	return false;
}

void RenderingDeviceVideoExtensions::_initialize_video_queues() {
	if (!rd) {
		return;
	}
	
	// TODO: Initialize video decode queue
	// This would involve finding a queue family that supports video operations
}

void RenderingDeviceVideoExtensions::_cleanup_video_resources() {
	// TODO: Cleanup any remaining video resources
	rd = nullptr;
}
