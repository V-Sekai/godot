/**************************************************************************/
/*  vulkan_video_session.h                                               */
/**************************************************************************/
/*                         This file is part of:                         */
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

#ifndef VULKAN_VIDEO_SESSION_H
#define VULKAN_VIDEO_SESSION_H

#include "core/error/error_macros.h"
#include "core/templates/local_vector.h"
#include "core/variant/variant.h"

#ifdef VULKAN_ENABLED

#include "drivers/vulkan/godot_vulkan.h"

class VulkanVideoSession {
public:
	enum VideoCodec {
		VIDEO_CODEC_AV1,
		VIDEO_CODEC_H264,
		VIDEO_CODEC_H265
	};

	struct VideoSessionCreateInfo {
		VideoCodec codec = VIDEO_CODEC_AV1;
		uint32_t max_width = 1920;
		uint32_t max_height = 1080;
		VkFormat output_format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM; // NV12
		VkFormat dpb_format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
		uint32_t max_dpb_slots = 8;
		uint32_t max_active_references = 7;
		bool enable_film_grain = false;
	};

	struct VideoCapabilities {
		VkVideoCapabilitiesKHR video_caps = {};
		VkVideoDecodeCapabilitiesKHR decode_caps = {};
		VkVideoDecodeAV1CapabilitiesKHR av1_caps = {};
		bool is_valid = false;
	};

	struct SessionInfo {
		VkVideoSessionKHR vk_video_session = VK_NULL_HANDLE;
		VkVideoSessionParametersKHR vk_session_parameters = VK_NULL_HANDLE;
		VkVideoProfileInfoKHR profile_info = {};
		VkVideoDecodeAV1ProfileInfoKHR av1_profile_info = {};
		VideoCapabilities capabilities;
		LocalVector<VkDeviceMemory> bound_memory;
		uint32_t width = 0;
		uint32_t height = 0;
		VideoCodec codec = VIDEO_CODEC_AV1;
		bool is_initialized = false;
	};

private:
	VkDevice vk_device = VK_NULL_HANDLE;
	VkPhysicalDevice physical_device = VK_NULL_HANDLE;
	uint32_t video_decode_queue_family = UINT32_MAX;

	// Function pointers for Vulkan Video API
	PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR GetPhysicalDeviceVideoCapabilitiesKHR = nullptr;
	PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR GetPhysicalDeviceVideoFormatPropertiesKHR = nullptr;
	PFN_vkCreateVideoSessionKHR CreateVideoSessionKHR = nullptr;
	PFN_vkDestroyVideoSessionKHR DestroyVideoSessionKHR = nullptr;
	PFN_vkCreateVideoSessionParametersKHR CreateVideoSessionParametersKHR = nullptr;
	PFN_vkDestroyVideoSessionParametersKHR DestroyVideoSessionParametersKHR = nullptr;
	PFN_vkUpdateVideoSessionParametersKHR UpdateVideoSessionParametersKHR = nullptr;
	PFN_vkGetVideoSessionMemoryRequirementsKHR GetVideoSessionMemoryRequirementsKHR = nullptr;
	PFN_vkBindVideoSessionMemoryKHR BindVideoSessionMemoryKHR = nullptr;

	SessionInfo session_info;

	bool _query_av1_decode_capabilities(VideoCapabilities *p_capabilities);
	Error _bind_video_session_memory();
	uint32_t _find_memory_type_index(uint32_t type_filter, VkMemoryPropertyFlags properties);

public:
	VulkanVideoSession();
	~VulkanVideoSession();

	Error initialize(VkDevice p_device, VkPhysicalDevice p_physical_device, uint32_t p_video_queue_family);
	void finalize();

	bool load_function_pointers(VkInstance p_instance);
	bool query_video_capabilities(VideoCodec p_codec, VideoCapabilities *p_capabilities);
	Error create_video_session(const VideoSessionCreateInfo &p_create_info);
	void destroy_video_session();

	bool is_initialized() const { return session_info.is_initialized; }
	const SessionInfo &get_session_info() const { return session_info; }
	VkVideoSessionKHR get_vk_session() const { return session_info.vk_video_session; }
	VkVideoSessionParametersKHR get_vk_parameters() const { return session_info.vk_session_parameters; }

	// Utility functions
	static String get_codec_name(VideoCodec p_codec);
	static VkVideoCodecOperationFlagBitsKHR get_vulkan_codec_operation(VideoCodec p_codec);
	Dictionary get_capabilities_info() const;
};

#endif // VULKAN_ENABLED

#endif // VULKAN_VIDEO_SESSION_H
