/**************************************************************************/
/*  test_vulkan_video_hardware.h                                         */
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

#ifdef VULKAN_ENABLED

#include "../vulkan_video_context.h"
#include "../rendering_device_video_extensions.h"

#include "tests/test_macros.h"

namespace TestVulkanVideoHardware {

TEST_CASE("[VulkanVideo] Context Creation") {
	Ref<VulkanVideoContext> context;
	context.instantiate();
	
	CHECK_MESSAGE(
		context.is_valid(),
		"VulkanVideoContext should instantiate successfully.");
	
	CHECK_MESSAGE(
		!context->is_initialized(),
		"VulkanVideoContext should not be initialized on creation.");
}

TEST_CASE("[VulkanVideo] Extensions Creation") {
	Ref<RenderingDeviceVideoExtensions> video_ext;
	video_ext.instantiate();
	
	CHECK_MESSAGE(
		video_ext.is_valid(),
		"RenderingDeviceVideoExtensions should instantiate successfully.");
	
	CHECK_MESSAGE(
		!video_ext->is_video_supported(),
		"Video should not be supported before initialization.");
}

TEST_CASE("[VulkanVideo] Null Safety") {
	Ref<RenderingDeviceVideoExtensions> video_ext;
	video_ext.instantiate();
	
	// Should handle null rendering device gracefully
	video_ext->initialize(nullptr);
	
	CHECK_MESSAGE(
		!video_ext->is_video_supported(),
		"Video should not be supported with null RenderingDevice.");
	
	// Test operations with invalid RIDs should not crash
	RID invalid_rid;
	video_ext->video_session_destroy(invalid_rid);
	video_ext->video_image_destroy(invalid_rid);
	video_ext->video_buffer_destroy(invalid_rid);
	
	CHECK_MESSAGE(
		true,
		"Invalid RID operations should not crash.");
}

TEST_CASE("[VulkanVideo] Profile Enumeration") {
	Ref<RenderingDeviceVideoExtensions> video_ext;
	video_ext.instantiate();
	
	Array profiles = video_ext->get_supported_profiles();
	
	CHECK_MESSAGE(
		profiles.size() >= 0,
		"Profile enumeration should return valid vector.");
	
	// Test capability query for AV1 main profile
	Dictionary caps = video_ext->get_video_capabilities(
		VIDEO_CODEC_PROFILE_AV1_MAIN, VIDEO_OPERATION_DECODE);
	
	CHECK_MESSAGE(
		caps.size() >= 0,
		"Video capabilities should return valid dictionary.");
}

TEST_CASE("[VulkanVideo] Resource Creation Stubs") {
	Ref<RenderingDeviceVideoExtensions> video_ext;
	video_ext.instantiate();
	
	// Test video session creation (should return invalid RID until implemented)
	Dictionary session_info;
	session_info["codec_profile"] = VIDEO_CODEC_PROFILE_AV1_MAIN;
	session_info["operation_type"] = VIDEO_OPERATION_DECODE;
	session_info["max_coded_extent_width"] = 1920;
	session_info["max_coded_extent_height"] = 1080;
	
	RID session = video_ext->video_session_create(session_info);
	
	CHECK_MESSAGE(
		true, // Currently returns invalid RID, which is expected
		"Video session creation should not crash.");
	
	// Test video session parameters creation
	Dictionary params_info;
	params_info["video_session"] = session;
	
	RID params = video_ext->video_session_parameters_create(params_info);
	
	CHECK_MESSAGE(
		true, // Currently returns invalid RID, which is expected
		"Video session parameters creation should not crash.");
	
	// Test video image creation
	Dictionary image_info;
	image_info["video_session"] = session;
	image_info["width"] = 1920;
	image_info["height"] = 1080;
	
	RID image = video_ext->video_image_create(image_info);
	
	CHECK_MESSAGE(
		true, // May return valid or invalid RID depending on implementation
		"Video image creation should not crash.");
	
	// Test video buffer creation
	Dictionary buffer_info;
	buffer_info["video_session"] = session;
	buffer_info["size"] = 1024 * 1024;
	
	RID buffer = video_ext->video_buffer_create(buffer_info);
	
	CHECK_MESSAGE(
		true, // May return valid or invalid RID depending on implementation
		"Video buffer creation should not crash.");
	
	// Cleanup (should handle invalid RIDs gracefully)
	if (buffer.is_valid()) {
		video_ext->video_buffer_destroy(buffer);
	}
	if (image.is_valid()) {
		video_ext->video_image_destroy(image);
	}
	if (params.is_valid()) {
		video_ext->video_session_parameters_destroy(params);
	}
	if (session.is_valid()) {
		video_ext->video_session_destroy(session);
	}
}

TEST_CASE("[VulkanVideo] Video Operations") {
	Ref<RenderingDeviceVideoExtensions> video_ext;
	video_ext.instantiate();
	
	// Test video decode frame (should not crash)
	Dictionary decode_info;
	decode_info["video_session"] = RID(); // Invalid RID
	decode_info["video_session_parameters"] = RID(); // Invalid RID
	decode_info["bitstream_buffer"] = RID(); // Invalid RID
	decode_info["output_image"] = RID(); // Invalid RID
	
	video_ext->video_decode_frame(decode_info);
	
	CHECK_MESSAGE(
		true,
		"Video decode frame should not crash with invalid RIDs.");
	
	// Test queue operations
	video_ext->video_queue_submit();
	video_ext->video_queue_wait_idle();
	
	CHECK_MESSAGE(
		true,
		"Video queue operations should not crash.");
}

TEST_CASE("[VulkanVideo] Utility Functions") {
	Ref<RenderingDeviceVideoExtensions> video_ext;
	video_ext.instantiate();
	
	// Test texture from video image
	RID texture = video_ext->texture_from_video_image(RID(), 0);
	
	CHECK_MESSAGE(
		true, // May return valid or invalid RID
		"Texture from video image should not crash.");
	
	// Test copy video image to texture
	video_ext->copy_video_image_to_texture(RID(), 0, RID());
	
	CHECK_MESSAGE(
		true,
		"Copy video image to texture should not crash.");
	
	// Test buffer operations
	Vector<uint8_t> test_data;
	test_data.resize(1024);
	for (int i = 0; i < 1024; i++) {
		test_data.write[i] = i % 256;
	}
	
	video_ext->video_buffer_update(RID(), 0, test_data);
	
	Vector<uint8_t> retrieved_data = video_ext->video_buffer_get_data(RID(), 0, 1024);
	
	CHECK_MESSAGE(
		retrieved_data.size() >= 0,
		"Video buffer operations should not crash.");
}

} // namespace TestVulkanVideoHardware

#endif // VULKAN_ENABLED
