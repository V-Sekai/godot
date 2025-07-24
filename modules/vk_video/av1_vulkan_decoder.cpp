/**************************************************************************/
/*  av1_vulkan_decoder.cpp                                               */
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

#include "av1_vulkan_decoder.h"

#include "core/error/error_macros.h"
#include "core/os/os.h"
#include "servers/rendering_server.h"
#include "rendering_device_video_extensions.h"

// libsimplewebm
#include <WebMDemuxer.hpp>

#ifdef VULKAN_ENABLED
#include "drivers/vulkan/rendering_device_driver_vulkan.h"
#include "drivers/vulkan/vulkan_video_decoder.h"
#endif

AV1VulkanDecoder::AV1VulkanDecoder() {
}

AV1VulkanDecoder::~AV1VulkanDecoder() {
	cleanup();
}

bool AV1VulkanDecoder::initialize(int width, int height) {
	ERR_FAIL_COND_V(width <= 0 || height <= 0, false);
	ERR_FAIL_COND_V(initialized, false);

	frame_width = width;
	frame_height = height;

	// Get RenderingDevice
	rendering_device = RenderingServer::get_singleton()->get_rendering_device();
	ERR_FAIL_NULL_V(rendering_device, false);

	// Check hardware support
	if (!check_hardware_support()) {
		ERR_PRINT("AV1 hardware decoding not supported on this device");
		return false;
	}

	// Create video session
	if (!create_video_session()) {
		ERR_PRINT("Failed to create AV1 video session");
		return false;
	}

	// Create session parameters
	if (!create_session_parameters()) {
		ERR_PRINT("Failed to create AV1 session parameters");
		cleanup();
		return false;
	}

	initialized = true;
	print_line("AV1VulkanDecoder initialized successfully for ", width, "x", height);
	return true;
}

bool AV1VulkanDecoder::check_hardware_support() {
	ERR_FAIL_NULL_V(rendering_device, false);

#ifdef VULKAN_ENABLED
	// Create video extensions instance to check support
	Ref<RenderingDeviceVideoExtensions> video_ext;
	video_ext.instantiate();
	video_ext->initialize(rendering_device);

	// Check if video is supported
	if (!video_ext->is_video_supported()) {
		WARN_PRINT("Vulkan Video extensions not supported on this device");
		hardware_support_available = false;
		return false;
	}

	// Check AV1 decode capabilities
	Dictionary caps = video_ext->get_video_capabilities(VIDEO_CODEC_PROFILE_AV1_MAIN, VIDEO_OPERATION_DECODE);
	bool decode_supported = caps.get("decode_supported", false);
	
	if (!decode_supported) {
		WARN_PRINT("AV1 hardware decode not supported on this device");
		hardware_support_available = false;
		return false;
	}

	hardware_support_available = true;
	print_line("AV1 hardware decode support detected");
	return true;
#else
	hardware_support_available = false;
	return false;
#endif
}

bool AV1VulkanDecoder::create_video_session() {
	ERR_FAIL_NULL_V(rendering_device, false);

	// Create video extensions instance
	Ref<RenderingDeviceVideoExtensions> video_ext;
	video_ext.instantiate();
	video_ext->initialize(rendering_device);

	// Prepare video session create info
	Dictionary create_info;
	create_info["codec_profile"] = VIDEO_CODEC_PROFILE_AV1_MAIN;
	create_info["operation_type"] = VIDEO_OPERATION_DECODE;
	create_info["max_width"] = frame_width;
	create_info["max_height"] = frame_height;
	create_info["max_dpb_slots"] = 8; // AV1 supports up to 8 reference frames
	create_info["max_active_references"] = 7;

	// Create video session
	video_session = video_ext->video_session_create(create_info);
	if (!video_session.is_valid()) {
		ERR_PRINT("Failed to create AV1 video session");
		return false;
	}

	print_line("AV1 video session created successfully");
	return true;
}

bool AV1VulkanDecoder::create_session_parameters() {
	ERR_FAIL_COND_V(!video_session.is_valid(), false);

	// Create video extensions instance
	Ref<RenderingDeviceVideoExtensions> video_ext;
	video_ext.instantiate();
	video_ext->initialize(rendering_device);

	// Prepare session parameters create info
	Dictionary create_info;
	create_info["video_session"] = video_session;
	create_info["codec_profile"] = VIDEO_CODEC_PROFILE_AV1_MAIN;

	// Create session parameters
	video_session_parameters = video_ext->video_session_parameters_create(create_info);
	if (!video_session_parameters.is_valid()) {
		ERR_PRINT("Failed to create AV1 session parameters");
		return false;
	}

	print_line("AV1 session parameters created successfully");
	return true;
}

bool AV1VulkanDecoder::decode_frame(const WebMFrame &frame) {
	ERR_FAIL_COND_V(!initialized, false);
	ERR_FAIL_COND_V(!frame.isValid(), false);
	ERR_FAIL_COND_V(!hardware_support_available, false);

	// Create video extensions instance
	Ref<RenderingDeviceVideoExtensions> video_ext;
	video_ext.instantiate();
	video_ext->initialize(rendering_device);

	// Create bitstream buffer if needed
	if (!_create_bitstream_buffer(frame.bufferSize)) {
		ERR_PRINT("Failed to create bitstream buffer");
		return false;
	}

	// Upload bitstream data
	Vector<uint8_t> bitstream_data;
	bitstream_data.resize(frame.bufferSize);
	memcpy(bitstream_data.ptrw(), frame.buffer, frame.bufferSize);
	video_ext->video_buffer_update(_get_bitstream_buffer(), 0, bitstream_data);

	// Create output image if needed
	if (!_create_output_image()) {
		ERR_PRINT("Failed to create output image");
		return false;
	}

	// Record decode commands
	if (!_record_decode_commands(frame)) {
		ERR_PRINT("Failed to record decode commands");
		return false;
	}

	// Submit decode commands and wait for completion
	if (!_submit_decode_commands()) {
		ERR_PRINT("Failed to submit decode commands");
		return false;
	}

	// Create texture from decoded frame
	current_frame_texture = _create_texture_from_decoded_frame();

	return current_frame_texture.is_valid();
}

Ref<Texture2D> AV1VulkanDecoder::get_current_frame() const {
	if (current_frame_texture.is_valid()) {
		return current_frame_texture;
	}
	
	// Return placeholder if no frame available
	return create_placeholder_texture();
}

void AV1VulkanDecoder::cleanup() {
	if (!rendering_device) {
		return;
	}

	// Create video extensions instance for cleanup
	Ref<RenderingDeviceVideoExtensions> video_ext;
	video_ext.instantiate();
	video_ext->initialize(rendering_device);

	// Cleanup video resources
	if (_bitstream_buffer.is_valid()) {
		video_ext->video_buffer_destroy(_bitstream_buffer);
		_bitstream_buffer = RID();
	}

	if (_output_image.is_valid()) {
		video_ext->video_image_destroy(_output_image);
		_output_image = RID();
	}

	if (video_session_parameters.is_valid()) {
		video_ext->video_session_parameters_destroy(video_session_parameters);
		video_session_parameters = RID();
	}

	if (video_session.is_valid()) {
		video_ext->video_session_destroy(video_session);
		video_session = RID();
	}

	// Reset state
	current_frame_texture.unref();
	initialized = false;
	hardware_support_available = false;
	frame_width = 0;
	frame_height = 0;

	print_line("AV1VulkanDecoder cleaned up");
}

bool AV1VulkanDecoder::_create_bitstream_buffer(size_t size) {
	if (_bitstream_buffer.is_valid() && _bitstream_buffer_size >= size) {
		return true; // Reuse existing buffer
	}

	// Create video extensions instance
	Ref<RenderingDeviceVideoExtensions> video_ext;
	video_ext.instantiate();
	video_ext->initialize(rendering_device);

	// Destroy old buffer if exists
	if (_bitstream_buffer.is_valid()) {
		video_ext->video_buffer_destroy(_bitstream_buffer);
	}

	// Create new buffer with some extra space
	size_t buffer_size = MAX(size * 2, (size_t)(1024 * 1024)); // At least 1MB

	Dictionary create_info;
	create_info["video_session"] = video_session;
	create_info["size"] = (uint64_t)buffer_size;
	create_info["usage"] = "bitstream";

	_bitstream_buffer = video_ext->video_buffer_create(create_info);
	_bitstream_buffer_size = buffer_size;

	return _bitstream_buffer.is_valid();
}

bool AV1VulkanDecoder::_create_output_image() {
	if (_output_image.is_valid()) {
		return true; // Reuse existing image
	}

	// Create video extensions instance
	Ref<RenderingDeviceVideoExtensions> video_ext;
	video_ext.instantiate();
	video_ext->initialize(rendering_device);

	// Create DPB image array for reference frames
	Dictionary create_info;
	create_info["video_session"] = video_session;
	create_info["width"] = frame_width;
	create_info["height"] = frame_height;
	create_info["array_layers"] = 8; // AV1 DPB size
	create_info["format"] = RD::DATA_FORMAT_G8_B8R8_2PLANE_420_UNORM; // NV12 format
	create_info["usage"] = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;

	_output_image = video_ext->video_image_create(create_info);

	return _output_image.is_valid();
}

Ref<ImageTexture> AV1VulkanDecoder::_create_texture_from_decoded_frame() {
	ERR_FAIL_COND_V(!_output_image.is_valid(), Ref<ImageTexture>());

	// Create video extensions instance
	Ref<RenderingDeviceVideoExtensions> video_ext;
	video_ext.instantiate();
	video_ext->initialize(rendering_device);

	// Get texture from video image (layer 0 for current frame)
	RID texture_rid = video_ext->texture_from_video_image(_output_image, 0);
	ERR_FAIL_COND_V(!texture_rid.is_valid(), Ref<ImageTexture>());

	// Create ImageTexture wrapper
	Ref<ImageTexture> texture;
	texture.instantiate();

	// For now, create a placeholder image until we have proper texture conversion
	// TODO: Implement proper YUV to RGB conversion and texture creation
	Ref<Image> placeholder_image = Image::create_empty(frame_width, frame_height, false, Image::FORMAT_RGB8);
	placeholder_image->fill(Color(0.5, 0.5, 0.5)); // Gray placeholder

	texture->set_image(placeholder_image);
	return texture;
}

Ref<ImageTexture> AV1VulkanDecoder::create_placeholder_texture() const {
	Ref<ImageTexture> texture;
	texture.instantiate();

	// Create a simple placeholder image
	int width = frame_width > 0 ? frame_width : 1920;
	int height = frame_height > 0 ? frame_height : 1080;

	Ref<Image> placeholder_image = Image::create_empty(width, height, false, Image::FORMAT_RGB8);
	placeholder_image->fill(Color(0.2, 0.2, 0.2)); // Dark gray placeholder

	// Add some visual indication this is a placeholder
	for (int y = height / 2 - 10; y < height / 2 + 10; y++) {
		for (int x = width / 2 - 50; x < width / 2 + 50; x++) {
			if (x >= 0 && x < width && y >= 0 && y < height) {
				placeholder_image->set_pixel(x, y, Color(0.8, 0.4, 0.4)); // Reddish indicator
			}
		}
	}

	texture->set_image(placeholder_image);
	return texture;
}

RID AV1VulkanDecoder::_get_bitstream_buffer() const {
	return _bitstream_buffer;
}

RID AV1VulkanDecoder::_get_output_image() const {
	return _output_image;
}

bool AV1VulkanDecoder::_record_decode_commands(const WebMFrame &frame) {
	ERR_FAIL_COND_V(!video_session.is_valid(), false);
	ERR_FAIL_COND_V(!video_session_parameters.is_valid(), false);
	ERR_FAIL_COND_V(!_bitstream_buffer.is_valid(), false);
	ERR_FAIL_COND_V(!_output_image.is_valid(), false);

	// Create video extensions instance
	Ref<RenderingDeviceVideoExtensions> video_ext;
	video_ext.instantiate();
	video_ext->initialize(rendering_device);

	// Prepare decode info
	Dictionary decode_info;
	decode_info["video_session"] = video_session;
	decode_info["video_session_parameters"] = video_session_parameters;
	decode_info["bitstream_buffer"] = _bitstream_buffer;
	decode_info["output_image"] = _output_image;
	decode_info["frame_width"] = frame_width;
	decode_info["frame_height"] = frame_height;
	decode_info["is_keyframe"] = frame.key;
	decode_info["presentation_time"] = frame.time;
	decode_info["bitstream_size"] = (uint64_t)frame.bufferSize;

	// Record decode command
	video_ext->video_decode_frame(decode_info);

	return true;
}

bool AV1VulkanDecoder::_submit_decode_commands() {
	// Create video extensions instance
	Ref<RenderingDeviceVideoExtensions> video_ext;
	video_ext.instantiate();
	video_ext->initialize(rendering_device);

	// Submit decode commands to video queue
	video_ext->video_queue_submit();

	// Wait for decode completion
	video_ext->video_queue_wait_idle();

	return true;
}
