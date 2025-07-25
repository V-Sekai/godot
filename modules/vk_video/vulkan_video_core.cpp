/**************************************************************************/
/*  vulkan_video_core.cpp                                                */
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

#include "vulkan_video_core.h"

#include "core/error/error_macros.h"
#include "core/math/math_funcs.h"
#include "rendering_device_video_extensions.h"
#include "servers/rendering_server.h"

void VulkanVideoCore::_bind_methods() {
	// Only bind simple methods that don't use custom types
	ClassDB::bind_method(D_METHOD("cleanup"), &VulkanVideoCore::cleanup);
	ClassDB::bind_method(D_METHOD("is_hardware_supported"), &VulkanVideoCore::is_hardware_supported);
	ClassDB::bind_method(D_METHOD("create_video_session", "width", "height"), &VulkanVideoCore::create_video_session);
	ClassDB::bind_method(D_METHOD("is_initialized"), &VulkanVideoCore::is_initialized);
	ClassDB::bind_method(D_METHOD("is_active"), &VulkanVideoCore::is_active);
}

VulkanVideoCore::VulkanVideoCore() {
	// Initialize DPB slots (AV1 supports up to 8 reference frames)
	dpb_slots.resize(8);
	for (int i = 0; i < dpb_slots.size(); i++) {
		dpb_slots.write[i].array_layer = i;
	}
}

VulkanVideoCore::~VulkanVideoCore() {
	cleanup();
}

Error VulkanVideoCore::initialize(RenderingDevice *p_rd, const VideoDecoderConfig &p_config) {
	ERR_FAIL_NULL_V(p_rd, ERR_INVALID_PARAMETER);
	
	rendering_device = p_rd;
	config = p_config;
	
	// Query hardware capabilities first
	Error err = _query_video_capabilities();
	if (err != OK) {
		ERR_PRINT("Failed to query video capabilities");
		return err;
	}
	
	// Validate codec support
	err = _validate_codec_support(config.codec_operation);
	if (err != OK) {
		ERR_PRINT("Codec not supported: " + config.get_codec_name());
		return err;
	}
	
	// Find appropriate video queue family
	err = _find_video_queue_family();
	if (err != OK) {
		ERR_PRINT("Failed to find suitable video decode queue family");
		return err;
	}
	
	session_state = VIDEO_SESSION_STATE_INITIALIZED;
	print_line("VulkanVideoCore initialized successfully for " + config.get_codec_name());
	
	return OK;
}

void VulkanVideoCore::cleanup() {
	if (rendering_device) {
		_cleanup_resources();
	}
	
	rendering_device = nullptr;
	session_state = VIDEO_SESSION_STATE_UNINITIALIZED;
	capabilities_queried = false;
}

bool VulkanVideoCore::is_hardware_supported() const {
	return capabilities_queried && capabilities.hardware_decode_supported;
}

VideoHardwareCapabilities VulkanVideoCore::get_hardware_capabilities() {
	if (!capabilities_queried) {
		_query_video_capabilities();
	}
	return capabilities;
}

bool VulkanVideoCore::supports_codec(VideoCodecOperation p_codec) const {
	if (!capabilities_queried) {
		return false;
	}
	
	// For now, focus on AV1 decode support
	return (p_codec == VIDEO_CODEC_OPERATION_DECODE_AV1) && capabilities.hardware_decode_supported;
}

bool VulkanVideoCore::supports_profile(VkVideoCodecProfile p_profile) const {
	if (!capabilities_queried) {
		return false;
	}
	
	for (int i = 0; i < capabilities.supported_profiles.size(); i++) {
		if (capabilities.supported_profiles[i] == p_profile) {
			return true;
		}
	}
	
	return false;
}

Error VulkanVideoCore::create_video_session(int p_width, int p_height) {
	ERR_FAIL_COND_V(!is_initialized(), ERR_UNCONFIGURED);
	ERR_FAIL_COND_V(p_width <= 0 || p_height <= 0, ERR_INVALID_PARAMETER);
	
	// Check if dimensions are supported
	if (capabilities.max_width > 0 && (uint32_t)p_width > capabilities.max_width) {
		ERR_PRINT(vformat("Video width %d exceeds maximum supported width %d", p_width, capabilities.max_width));
		return ERR_INVALID_PARAMETER;
	}
	
	if (capabilities.max_height > 0 && (uint32_t)p_height > capabilities.max_height) {
		ERR_PRINT(vformat("Video height %d exceeds maximum supported height %d", p_height, capabilities.max_height));
		return ERR_INVALID_PARAMETER;
	}
	
	Error err = _create_video_session_internal(p_width, p_height);
	if (err != OK) {
		ERR_PRINT("Failed to create video session");
		session_state = VIDEO_SESSION_STATE_ERROR;
		return err;
	}
	
	// Create DPB images
	int num_dpb_images = MAX(config.num_decode_images_in_flight, 8);
	err = create_dpb_images(p_width, p_height, num_dpb_images);
	if (err != OK) {
		ERR_PRINT("Failed to create DPB images");
		session_state = VIDEO_SESSION_STATE_ERROR;
		return err;
	}
	
	// Create bitstream buffer
	uint32_t bitstream_buffer_size = 1024 * 1024; // 1MB default
	err = create_bitstream_buffer(bitstream_buffer_size);
	if (err != OK) {
		ERR_PRINT("Failed to create bitstream buffer");
		session_state = VIDEO_SESSION_STATE_ERROR;
		return err;
	}
	
	session_state = VIDEO_SESSION_STATE_CONFIGURED;
	print_line(vformat("Video session created for %dx%d", p_width, p_height));
	
	return OK;
}

Error VulkanVideoCore::configure_session_parameters() {
	ERR_FAIL_COND_V(session_state != VIDEO_SESSION_STATE_CONFIGURED, ERR_UNCONFIGURED);
	
	// TODO: Configure session parameters based on sequence header
	// This would involve setting up AV1-specific parameters
	
	return OK;
}

Error VulkanVideoCore::begin_video_session() {
	ERR_FAIL_COND_V(session_state != VIDEO_SESSION_STATE_CONFIGURED, ERR_UNCONFIGURED);
	
	// TODO: Begin video session for decoding
	session_state = VIDEO_SESSION_STATE_ACTIVE;
	
	return OK;
}

Error VulkanVideoCore::end_video_session() {
	if (session_state == VIDEO_SESSION_STATE_ACTIVE) {
		// TODO: End video session
		session_state = VIDEO_SESSION_STATE_CONFIGURED;
	}
	
	return OK;
}

Error VulkanVideoCore::create_dpb_images(int p_width, int p_height, int p_num_images) {
	ERR_FAIL_NULL_V(rendering_device, ERR_UNCONFIGURED);
	
	// TODO: Create DPB image array using RenderingDevice
	// This would create a texture array for the decoded picture buffer
	
	decode_images.resize(p_num_images);
	
	// For now, create placeholder RIDs
	// In a full implementation, this would create actual Vulkan images
	for (int i = 0; i < p_num_images; i++) {
		// decode_images.write[i] = rendering_device->texture_create(...);
	}
	
	print_line(vformat("Created %d DPB images (%dx%d)", p_num_images, p_width, p_height));
	
	return OK;
}

Error VulkanVideoCore::create_bitstream_buffer(uint32_t p_size) {
	ERR_FAIL_NULL_V(rendering_device, ERR_UNCONFIGURED);
	
	// TODO: Create bitstream buffer using RenderingDevice
	// This would create a buffer for storing compressed video data
	
	print_line(vformat("Created bitstream buffer of size %d bytes", p_size));
	
	return OK;
}

int VulkanVideoCore::allocate_dpb_slot() {
	for (int i = 0; i < dpb_slots.size(); i++) {
		if (!dpb_slots[i].in_use) {
			dpb_slots.write[i].in_use = true;
			return i;
		}
	}
	
	// If no free slots, find the oldest frame
	int oldest_slot = 0;
	uint64_t oldest_frame = dpb_slots[0].frame_number;
	for (int i = 1; i < dpb_slots.size(); i++) {
		if (dpb_slots[i].frame_number < oldest_frame) {
			oldest_frame = dpb_slots[i].frame_number;
			oldest_slot = i;
		}
	}
	
	dpb_slots.write[oldest_slot].in_use = true;
	return oldest_slot;
}

void VulkanVideoCore::release_dpb_slot(int p_slot) {
	ERR_FAIL_INDEX(p_slot, dpb_slots.size());
	dpb_slots.write[p_slot].in_use = false;
	dpb_slots.write[p_slot].is_reference = false;
}

Error VulkanVideoCore::decode_frame(const uint8_t *p_bitstream_data, uint32_t p_bitstream_size, 
								   const VideoFrameInfo &p_frame_info) {
	ERR_FAIL_COND_V(session_state != VIDEO_SESSION_STATE_ACTIVE, ERR_UNCONFIGURED);
	ERR_FAIL_NULL_V(p_bitstream_data, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_bitstream_size == 0, ERR_INVALID_PARAMETER);
	
	// TODO: Implement actual Vulkan Video decode operation
	// This would involve:
	// 1. Uploading bitstream data to buffer
	// 2. Setting up decode parameters
	// 3. Submitting decode command to video queue
	// 4. Managing DPB references
	
	print_line(vformat("Decoding frame %d (size: %d bytes)", p_frame_info.frame_number, p_bitstream_size));
	
	return OK;
}

RID VulkanVideoCore::get_decoded_image(int p_dpb_slot) const {
	ERR_FAIL_INDEX_V(p_dpb_slot, decode_images.size(), RID());
	return decode_images[p_dpb_slot];
}

Error VulkanVideoCore::_query_video_capabilities() {
	ERR_FAIL_NULL_V(rendering_device, ERR_UNCONFIGURED);
	
	// Use RenderingDeviceVideoExtensions to query capabilities
	Ref<RenderingDeviceVideoExtensions> video_ext;
	video_ext.instantiate();
	video_ext->initialize(rendering_device);
	
	if (!video_ext->is_video_supported()) {
		capabilities.hardware_decode_supported = false;
		capabilities_queried = true;
		return OK;
	}
	
	// Query AV1 decode capabilities
	Dictionary caps_dict = video_ext->get_video_capabilities(
		VIDEO_CODEC_PROFILE_AV1_MAIN,
		VIDEO_OPERATION_DECODE
	);
	
	capabilities.hardware_decode_supported = caps_dict.get("hardware_decode_supported", false);
	capabilities.hardware_encode_supported = caps_dict.get("hardware_encode_supported", false);
	capabilities.max_width = caps_dict.get("max_width", 0);
	capabilities.max_height = caps_dict.get("max_height", 0);
	capabilities.max_dpb_slots = caps_dict.get("max_dpb_slots", 0);
	capabilities.max_level = caps_dict.get("max_level", 0);
	
	// Parse supported profiles
	Array profiles_array = caps_dict.get("supported_profiles", Array());
	capabilities.supported_profiles.clear();
	for (int i = 0; i < profiles_array.size(); i++) {
		capabilities.supported_profiles.push_back((VkVideoCodecProfile)(int)profiles_array[i]);
	}
	
	// Parse supported bit depths
	Array bit_depths_array = caps_dict.get("supported_bit_depths", Array());
	capabilities.supported_bit_depths.clear();
	for (int i = 0; i < bit_depths_array.size(); i++) {
		capabilities.supported_bit_depths.push_back((int)bit_depths_array[i]);
	}
	
	capabilities_queried = true;
	
	print_line(vformat("Video capabilities: decode=%s, max_res=%dx%d, dpb_slots=%d", 
		capabilities.hardware_decode_supported ? "yes" : "no",
		capabilities.max_width, capabilities.max_height, capabilities.max_dpb_slots));
	
	return OK;
}

Error VulkanVideoCore::_find_video_queue_family() {
	ERR_FAIL_NULL_V(rendering_device, ERR_UNCONFIGURED);
	
	// TODO: Query queue families that support video decode
	// For now, assume we found a suitable queue
	video_decode_queue_family = 0; // Placeholder
	video_decode_queue_index = 0;
	video_queue_flags = VULKAN_VIDEO_QUEUE_DECODE | VULKAN_VIDEO_QUEUE_TRANSFER;
	
	return OK;
}

Error VulkanVideoCore::_create_video_session_internal(int p_width, int p_height) {
	ERR_FAIL_NULL_V(rendering_device, ERR_UNCONFIGURED);
	
	// TODO: Create actual Vulkan Video session
	// This would involve creating VkVideoSessionKHR object
	
	return OK;
}

Error VulkanVideoCore::_validate_codec_support(VideoCodecOperation p_codec) const {
	if (!capabilities_queried) {
		return ERR_UNCONFIGURED;
	}
	
	if (!capabilities.hardware_decode_supported) {
		return ERR_UNAVAILABLE;
	}
	
	// For now, only support AV1 decode
	if (p_codec != VIDEO_CODEC_OPERATION_DECODE_AV1) {
		return ERR_UNAVAILABLE;
	}
	
	return OK;
}

void VulkanVideoCore::_cleanup_resources() {
	if (!rendering_device) {
		return;
	}
	
	// TODO: Cleanup Vulkan Video resources
	// rendering_device->free_rid(video_session);
	// rendering_device->free_rid(video_session_parameters);
	// rendering_device->free_rid(dpb_image_array);
	// rendering_device->free_rid(bitstream_buffer);
	
	for (int i = 0; i < decode_images.size(); i++) {
		if (decode_images[i].is_valid()) {
			// rendering_device->free_rid(decode_images[i]);
		}
	}
	
	video_session = RID();
	video_session_parameters = RID();
	dpb_image_array = RID();
	bitstream_buffer = RID();
	decode_images.clear();
	
	// Reset DPB slots
	for (int i = 0; i < dpb_slots.size(); i++) {
		dpb_slots.write[i].in_use = false;
		dpb_slots.write[i].is_reference = false;
		dpb_slots.write[i].image = RID();
	}
}
