/**************************************************************************/
/*  vulkan_video_context.h                                               */
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
#include "core/templates/vector.h"
#include "servers/rendering/rendering_device.h"

class RenderingDeviceDriverVulkan;

// Hardware capability information
struct VulkanVideoHardwareInfo {
	bool video_queue_supported = false;
	bool decode_queue_supported = false;
	bool encode_queue_supported = false;

	// Queue family indices
	uint32_t video_queue_family = VK_QUEUE_FAMILY_IGNORED;
	uint32_t decode_queue_family = VK_QUEUE_FAMILY_IGNORED;
	uint32_t encode_queue_family = VK_QUEUE_FAMILY_IGNORED;

	// Supported codecs
	Vector<VkVideoCodecOperationFlagBitsKHR> supported_decode_codecs;
	Vector<VkVideoCodecOperationFlagBitsKHR> supported_encode_codecs;

	// Device limits
	uint32_t max_coded_extent_width = 0;
	uint32_t max_coded_extent_height = 0;
	uint32_t max_dpb_slots = 0;
	uint32_t max_active_reference_pictures = 0;

	// Memory requirements
	VkDeviceSize video_session_memory_size = 0;
	uint32_t video_session_memory_type_bits = 0;
};

// Video session wrapper
struct VulkanVideoSession {
	VkVideoSessionKHR session = VK_NULL_HANDLE;
	VkVideoSessionParametersKHR parameters = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
	VkVideoCodecOperationFlagBitsKHR codec_operation = VK_VIDEO_CODEC_OPERATION_NONE_KHR;
	uint32_t max_width = 0;
	uint32_t max_height = 0;
	uint32_t max_dpb_slots = 0;
	bool is_valid = false;
};

// Video image wrapper
struct VulkanVideoImage {
	VkImage image = VK_NULL_HANDLE;
	VkImageView image_view = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t array_layers = 0;
	VkFormat format = VK_FORMAT_UNDEFINED;
	bool is_valid = false;
};

// Video buffer wrapper
struct VulkanVideoBuffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
	VkDeviceSize size = 0;
	void *mapped_data = nullptr;
	bool is_valid = false;
};

class VulkanVideoContext {
private:
	RenderingDeviceDriverVulkan *driver = nullptr;
	VkDevice device = VK_NULL_HANDLE;
	VkPhysicalDevice physical_device = VK_NULL_HANDLE;
	VkQueue video_queue = VK_NULL_HANDLE;
	VkCommandPool video_command_pool = VK_NULL_HANDLE;

	VulkanVideoHardwareInfo hardware_info;

	// Resource tracking
	HashMap<uint32_t, VulkanVideoSession> video_sessions;
	HashMap<uint32_t, VulkanVideoImage> video_images;
	HashMap<uint32_t, VulkanVideoBuffer> video_buffers;

	uint32_t next_session_id = 1;
	uint32_t next_image_id = 1;
	uint32_t next_buffer_id = 1;

	bool initialized = false;

public:
	VulkanVideoContext();
	virtual ~VulkanVideoContext();

	// Initialization
	bool initialize(RenderingDeviceDriverVulkan *p_driver);
	void cleanup();
	bool is_initialized() const { return initialized; }

	// Hardware detection
	bool detect_video_hardware();
	VulkanVideoHardwareInfo get_hardware_info() const { return hardware_info; }

	// Capability queries
	bool is_codec_supported(VkVideoCodecOperationFlagBitsKHR p_codec_operation) const;
	bool get_video_capabilities(VkVideoCodecOperationFlagBitsKHR p_codec_operation, VkVideoCapabilitiesKHR &r_capabilities) const;
	Vector<VkVideoCodecOperationFlagBitsKHR> get_supported_codecs() const;

	// Video session management
	uint32_t create_video_session(VkVideoCodecOperationFlagBitsKHR p_codec_operation, uint32_t p_width, uint32_t p_height, uint32_t p_dpb_slots = 8);
	void destroy_video_session(uint32_t p_session_id);
	bool is_video_session_valid(uint32_t p_session_id) const;

	uint32_t create_video_session_parameters(uint32_t p_session_id);
	void destroy_video_session_parameters(uint32_t p_session_id);

	// Video resource creation
	uint32_t create_video_image(uint32_t p_session_id, VkFormat p_format, VkImageUsageFlags p_usage);
	void destroy_video_image(uint32_t p_image_id);

	uint32_t create_video_buffer(VkDeviceSize p_size, VkBufferUsageFlags p_usage);
	void destroy_video_buffer(uint32_t p_buffer_id);

	// Video operations
	bool begin_video_coding(VkCommandBuffer p_cmd_buffer, uint32_t p_session_id);
	bool end_video_coding(VkCommandBuffer p_cmd_buffer);
	bool decode_video_frame(VkCommandBuffer p_cmd_buffer, uint32_t p_session_id, uint32_t p_bitstream_buffer_id, uint32_t p_output_image_id);

	// Memory management
	bool update_video_buffer(uint32_t p_buffer_id, uint64_t p_offset, const Vector<uint8_t> &p_data);
	Vector<uint8_t> get_video_buffer_data(uint32_t p_buffer_id, uint64_t p_offset = 0, uint64_t p_size = 0);

	// Utility functions
	VkFormat get_vulkan_format_from_rd(RD::DataFormat p_format) const;
	RD::DataFormat get_rd_format_from_vulkan(VkFormat p_format) const;

private:
	// Internal helpers
	bool _find_video_queue_families();
	bool _create_video_command_pool();
	bool _query_video_capabilities();

	// Resource management helpers
	bool _allocate_video_session_memory(VulkanVideoSession &p_session);
	bool _allocate_video_image_memory(VulkanVideoImage &p_image);
	bool _allocate_video_buffer_memory(VulkanVideoBuffer &p_buffer);

	// Cleanup helpers
	void _cleanup_video_sessions();
	void _cleanup_video_images();
	void _cleanup_video_buffers();
};

#endif // VULKAN_ENABLED
