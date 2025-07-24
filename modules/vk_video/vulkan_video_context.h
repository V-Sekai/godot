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

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/vector.h"
#include "rendering_device_video_extensions.h"

#ifdef VULKAN_ENABLED
#include "drivers/vulkan/rendering_context_driver_vulkan.h"
#include "drivers/vulkan/godot_vulkan.h"

// Vulkan Video extension function pointers
struct VulkanVideoFunctions {
	// Core video functions
	PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR vkGetPhysicalDeviceVideoCapabilitiesKHR = nullptr;
	PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR vkGetPhysicalDeviceVideoFormatPropertiesKHR = nullptr;
	PFN_vkCreateVideoSessionKHR vkCreateVideoSessionKHR = nullptr;
	PFN_vkDestroyVideoSessionKHR vkDestroyVideoSessionKHR = nullptr;
	PFN_vkCreateVideoSessionParametersKHR vkCreateVideoSessionParametersKHR = nullptr;
	PFN_vkDestroyVideoSessionParametersKHR vkDestroyVideoSessionParametersKHR = nullptr;
	PFN_vkUpdateVideoSessionParametersKHR vkUpdateVideoSessionParametersKHR = nullptr;
	PFN_vkGetVideoSessionMemoryRequirementsKHR vkGetVideoSessionMemoryRequirementsKHR = nullptr;
	PFN_vkBindVideoSessionMemoryKHR vkBindVideoSessionMemoryKHR = nullptr;
	
	// Video decode functions
	PFN_vkCmdBeginVideoCodingKHR vkCmdBeginVideoCodingKHR = nullptr;
	PFN_vkCmdEndVideoCodingKHR vkCmdEndVideoCodingKHR = nullptr;
	PFN_vkCmdControlVideoCodingKHR vkCmdControlVideoCodingKHR = nullptr;
	PFN_vkCmdDecodeVideoKHR vkCmdDecodeVideoKHR = nullptr;
	
	// Video encode functions (for future use)
	PFN_vkCmdEncodeVideoKHR vkCmdEncodeVideoKHR = nullptr;
	
	bool is_loaded = false;
};

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
	VideoCodecProfile codec_profile = VIDEO_CODEC_PROFILE_AV1_MAIN;
	VideoOperationType operation_type = VIDEO_OPERATION_DECODE;
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

class VulkanVideoContext : public RefCounted {
	GDCLASS(VulkanVideoContext, RefCounted);

private:
	RenderingContextDriverVulkan *context_driver = nullptr;
	VkDevice device = VK_NULL_HANDLE;
	VkPhysicalDevice physical_device = VK_NULL_HANDLE;
	VkQueue video_queue = VK_NULL_HANDLE;
	VkCommandPool video_command_pool = VK_NULL_HANDLE;
	
	VulkanVideoFunctions video_functions;
	VulkanVideoHardwareInfo hardware_info;
	
	// Resource tracking
	HashMap<RID, VulkanVideoSession> video_sessions;
	HashMap<RID, VulkanVideoImage> video_images;
	HashMap<RID, VulkanVideoBuffer> video_buffers;
	
	bool initialized = false;

protected:
	static void _bind_methods();

public:
	VulkanVideoContext();
	virtual ~VulkanVideoContext();

	// Initialization
	bool initialize(RenderingContextDriverVulkan *p_context_driver);
	void cleanup();
	bool is_initialized() const { return initialized; }

	// Extension loading
	bool load_video_extensions();
	bool check_video_support();
	
	// Hardware detection
	bool detect_video_hardware();
	VulkanVideoHardwareInfo get_hardware_info() const { return hardware_info; }
	
	// Capability queries
	bool is_codec_supported(VideoCodecProfile p_profile, VideoOperationType p_operation) const;
	Dictionary get_video_capabilities(VideoCodecProfile p_profile, VideoOperationType p_operation) const;
	Array get_supported_profiles() const;
	
	// Video session management
	RID create_video_session(const Dictionary &p_create_info);
	void destroy_video_session(RID p_session_rid);
	bool is_video_session_valid(RID p_session_rid) const;
	
	RID create_video_session_parameters(const Dictionary &p_create_info);
	void destroy_video_session_parameters(RID p_parameters_rid);
	
	// Video resource creation
	RID create_video_image(const Dictionary &p_create_info);
	void destroy_video_image(RID p_image_rid);
	
	RID create_video_buffer(const Dictionary &p_create_info);
	void destroy_video_buffer(RID p_buffer_rid);
	
	// Video operations
	bool begin_video_coding(VkCommandBuffer p_cmd_buffer, RID p_session_rid);
	bool end_video_coding(VkCommandBuffer p_cmd_buffer);
	bool decode_video_frame(VkCommandBuffer p_cmd_buffer, const Dictionary &p_decode_info);
	
	// Memory management
	bool update_video_buffer(RID p_buffer_rid, uint64_t p_offset, const Vector<uint8_t> &p_data);
	Vector<uint8_t> get_video_buffer_data(RID p_buffer_rid, uint64_t p_offset = 0, uint64_t p_size = 0);
	
	// Utility functions
	VkFormat get_vulkan_format(RD::DataFormat p_format) const;
	RD::DataFormat get_rd_format(VkFormat p_format) const;
	VkVideoCodecOperationFlagBitsKHR get_vulkan_codec_operation(VideoCodecProfile p_profile, VideoOperationType p_operation) const;

private:
	// Internal helpers
	bool _load_video_function_pointers();
	bool _find_video_queue_families();
	bool _create_video_command_pool();
	bool _query_video_capabilities();
	
	// Resource management helpers
	RID _generate_video_rid();
	bool _allocate_video_session_memory(VulkanVideoSession &p_session);
	bool _allocate_video_image_memory(VulkanVideoImage &p_image);
	bool _allocate_video_buffer_memory(VulkanVideoBuffer &p_buffer);
	
	// Cleanup helpers
	void _cleanup_video_sessions();
	void _cleanup_video_images();
	void _cleanup_video_buffers();
};

#endif // VULKAN_ENABLED
