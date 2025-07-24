/**************************************************************************/
/*  vulkan_video_decoder.h                                               */
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

#include "drivers/vulkan/godot_vulkan.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "servers/rendering/rendering_device_driver.h"

class RenderingDeviceDriverVulkan;

// Core Vulkan Video decoder implementation
// This class handles the low-level Vulkan Video API calls
class VulkanVideoDecoder {
public:
	enum VideoCodec {
		VIDEO_CODEC_AV1,
		VIDEO_CODEC_H264,
		VIDEO_CODEC_H265
	};

	struct VideoCapabilities {
		uint32_t max_width = 0;
		uint32_t max_height = 0;
		uint32_t max_dpb_slots = 0;
		uint32_t max_active_references = 0;
		bool film_grain_support = false;
	};

	struct VideoSessionInfo {
		VkVideoSessionKHR session = VK_NULL_HANDLE;
		VkVideoSessionParametersKHR parameters = VK_NULL_HANDLE;
		LocalVector<VkDeviceMemory> bound_memory;
		VideoCodec codec = VIDEO_CODEC_AV1;
		uint32_t width = 0;
		uint32_t height = 0;
		uint32_t dpb_slots = 0;
	};

	struct VideoImage {
		VkImage image = VK_NULL_HANDLE;
		VkImageView view = VK_NULL_HANDLE;
		VkDeviceMemory memory = VK_NULL_HANDLE;
		VkFormat format = VK_FORMAT_UNDEFINED;
		uint32_t width = 0;
		uint32_t height = 0;
	};

	struct VideoBuffer {
		VkBuffer buffer = VK_NULL_HANDLE;
		VkDeviceMemory memory = VK_NULL_HANDLE;
		uint64_t size = 0;
		void *mapped_ptr = nullptr;
	};

private:
	RenderingDeviceDriverVulkan *driver = nullptr;
	
	// Video sessions by ID
	HashMap<uint32_t, VideoSessionInfo> video_sessions;
	HashMap<uint32_t, VideoImage> dpb_images;
	HashMap<uint32_t, VideoImage> output_images;
	HashMap<uint32_t, VideoBuffer> video_buffers;
	
	uint32_t next_session_id = 1;
	uint32_t next_image_id = 1;
	uint32_t next_buffer_id = 1;

	// Helper methods
	bool _query_video_capabilities(VideoCodec p_codec, VideoCapabilities &r_caps);
	bool _create_video_session_internal(VideoSessionInfo &r_session, VideoCodec p_codec, uint32_t p_width, uint32_t p_height);
	bool _allocate_and_bind_session_memory(VideoSessionInfo &r_session);
	bool _create_dpb_image(VideoImage &r_image, VkFormat p_format, uint32_t p_width, uint32_t p_height);
	bool _create_output_image(VideoImage &r_image, VkFormat p_format, uint32_t p_width, uint32_t p_height);
	bool _create_video_buffer_internal(VideoBuffer &r_buffer, uint64_t p_size);

public:
	VulkanVideoDecoder();
	~VulkanVideoDecoder();

	// Initialization
	bool initialize(RenderingDeviceDriverVulkan *p_driver);
	void cleanup();

	// Capability queries
	bool is_codec_supported(VideoCodec p_codec);
	VideoCapabilities get_video_capabilities(VideoCodec p_codec);

	// Session management
	uint32_t create_video_session(VideoCodec p_codec, uint32_t p_width, uint32_t p_height, uint32_t p_dpb_slots = 8);
	void destroy_video_session(uint32_t p_session_id);
	bool is_session_valid(uint32_t p_session_id);

	// Image management
	uint32_t create_dpb_image(uint32_t p_session_id, VkFormat p_format);
	uint32_t create_output_image(uint32_t p_session_id, VkFormat p_format);
	void destroy_video_image(uint32_t p_image_id);
	VkImage get_image_handle(uint32_t p_image_id);
	VkImageView get_image_view(uint32_t p_image_id);

	// Buffer management
	uint32_t create_video_buffer(uint64_t p_size);
	void destroy_video_buffer(uint32_t p_buffer_id);
	bool update_video_buffer(uint32_t p_buffer_id, const Vector<uint8_t> &p_data, uint64_t p_offset = 0);
	VkBuffer get_buffer_handle(uint32_t p_buffer_id);

	// Decode operations
	bool begin_video_coding(VkCommandBuffer p_cmd_buffer, uint32_t p_session_id);
	bool end_video_coding(VkCommandBuffer p_cmd_buffer);
	bool decode_frame(VkCommandBuffer p_cmd_buffer, uint32_t p_session_id, uint32_t p_bitstream_buffer_id, 
					  uint32_t p_output_image_id, const Vector<uint32_t> &p_reference_images = Vector<uint32_t>());

	// YCbCr conversion support
	VkSamplerYcbcrConversion create_ycbcr_conversion(VkFormat p_format);
	void destroy_ycbcr_conversion(VkSamplerYcbcrConversion p_conversion);
};

#endif // VULKAN_ENABLED
