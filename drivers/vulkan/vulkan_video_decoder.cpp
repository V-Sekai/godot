/**************************************************************************/
/*  vulkan_video_decoder.cpp                                             */
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

#ifdef VULKAN_ENABLED

#include "vulkan_video_decoder.h"
#include "rendering_device_driver_vulkan.h"
#include "core/error/error_macros.h"

VulkanVideoDecoder::VulkanVideoDecoder() {
}

VulkanVideoDecoder::~VulkanVideoDecoder() {
	cleanup();
}

bool VulkanVideoDecoder::initialize(RenderingDeviceDriverVulkan *p_driver) {
	ERR_FAIL_NULL_V(p_driver, false);
	driver = p_driver;
	
	// Check if video decode queue is available
	if (driver->get_video_decode_queue_family() == UINT32_MAX) {
		WARN_PRINT("No video decode queue family found");
		return false;
	}
	
	print_line("Vulkan Video Decoder initialized successfully");
	return true;
}

void VulkanVideoDecoder::cleanup() {
	if (!driver) {
		return;
	}
	
	// Cleanup all video sessions
	for (KeyValue<uint32_t, VideoSessionInfo> &pair : video_sessions) {
		VideoSessionInfo &session = pair.value;
		
		if (session.parameters != VK_NULL_HANDLE) {
			driver->get_device_functions().DestroyVideoSessionParametersKHR(driver->get_vk_device(), session.parameters, nullptr);
		}
		if (session.session != VK_NULL_HANDLE) {
			driver->get_device_functions().DestroyVideoSessionKHR(driver->get_vk_device(), session.session, nullptr);
		}
		
		// Free bound memory
		for (VkDeviceMemory memory : session.bound_memory) {
			vkFreeMemory(driver->get_vk_device(), memory, nullptr);
		}
	}
	video_sessions.clear();
	
	// Cleanup images
	for (KeyValue<uint32_t, VideoImage> &pair : dpb_images) {
		VideoImage &image = pair.value;
		if (image.view != VK_NULL_HANDLE) {
			vkDestroyImageView(driver->get_vk_device(), image.view, nullptr);
		}
		if (image.image != VK_NULL_HANDLE) {
			vkDestroyImage(driver->get_vk_device(), image.image, nullptr);
		}
		if (image.memory != VK_NULL_HANDLE) {
			vkFreeMemory(driver->get_vk_device(), image.memory, nullptr);
		}
	}
	dpb_images.clear();
	
	for (KeyValue<uint32_t, VideoImage> &pair : output_images) {
		VideoImage &image = pair.value;
		if (image.view != VK_NULL_HANDLE) {
			vkDestroyImageView(driver->get_vk_device(), image.view, nullptr);
		}
		if (image.image != VK_NULL_HANDLE) {
			vkDestroyImage(driver->get_vk_device(), image.image, nullptr);
		}
		if (image.memory != VK_NULL_HANDLE) {
			vkFreeMemory(driver->get_vk_device(), image.memory, nullptr);
		}
	}
	output_images.clear();
	
	// Cleanup buffers
	for (KeyValue<uint32_t, VideoBuffer> &pair : video_buffers) {
		VideoBuffer &buffer = pair.value;
		if (buffer.mapped_ptr) {
			vkUnmapMemory(driver->get_vk_device(), buffer.memory);
		}
		if (buffer.buffer != VK_NULL_HANDLE) {
			vkDestroyBuffer(driver->get_vk_device(), buffer.buffer, nullptr);
		}
		if (buffer.memory != VK_NULL_HANDLE) {
			vkFreeMemory(driver->get_vk_device(), buffer.memory, nullptr);
		}
	}
	video_buffers.clear();
	
	driver = nullptr;
}

bool VulkanVideoDecoder::is_codec_supported(VideoCodec p_codec) {
	ERR_FAIL_NULL_V(driver, false);
	
	VideoCapabilities caps;
	return _query_video_capabilities(p_codec, caps);
}

VulkanVideoDecoder::VideoCapabilities VulkanVideoDecoder::get_video_capabilities(VideoCodec p_codec) {
	VideoCapabilities caps = {};
	if (driver) {
		_query_video_capabilities(p_codec, caps);
	}
	return caps;
}

bool VulkanVideoDecoder::_query_video_capabilities(VideoCodec p_codec, VideoCapabilities &r_caps) {
	ERR_FAIL_NULL_V(driver, false);
	
	// Set up video profile based on codec
	VkVideoProfileInfoKHR profile_info = {};
	profile_info.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR;
	
	VkVideoDecodeAV1ProfileInfoKHR av1_profile = {};
	VkVideoDecodeH264ProfileInfoKHR h264_profile = {};
	VkVideoDecodeH265ProfileInfoKHR h265_profile = {};
	
	switch (p_codec) {
		case VIDEO_CODEC_AV1:
			profile_info.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR;
			av1_profile.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PROFILE_INFO_KHR;
			av1_profile.stdProfile = STD_VIDEO_AV1_PROFILE_MAIN;
			profile_info.pNext = &av1_profile;
			break;
		case VIDEO_CODEC_H264:
			profile_info.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;
			h264_profile.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PROFILE_INFO_KHR;
			h264_profile.stdProfileIdc = STD_VIDEO_H264_PROFILE_IDC_MAIN;
			profile_info.pNext = &h264_profile;
			break;
		case VIDEO_CODEC_H265:
			profile_info.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR;
			h265_profile.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PROFILE_INFO_KHR;
			h265_profile.stdProfileIdc = STD_VIDEO_H265_PROFILE_IDC_MAIN;
			profile_info.pNext = &h265_profile;
			break;
		default:
			return false;
	}
	
	// Query capabilities
	VkVideoCapabilitiesKHR capabilities = {};
	capabilities.sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR;
	
	VkResult result = driver->get_device_functions().GetPhysicalDeviceVideoCapabilitiesKHR(
		driver->get_vk_physical_device(), &profile_info, &capabilities);
	
	if (result == VK_SUCCESS) {
		r_caps.max_width = capabilities.maxCodedExtent.width;
		r_caps.max_height = capabilities.maxCodedExtent.height;
		r_caps.max_dpb_slots = capabilities.maxDpbSlots;
		r_caps.max_active_references = capabilities.maxActiveReferencePictures;
		r_caps.film_grain_support = (p_codec == VIDEO_CODEC_AV1); // AV1 specific
		return true;
	}
	
	return false;
}

uint32_t VulkanVideoDecoder::create_video_session(VideoCodec p_codec, uint32_t p_width, uint32_t p_height, uint32_t p_dpb_slots) {
	ERR_FAIL_NULL_V(driver, 0);
	
	uint32_t session_id = next_session_id++;
	VideoSessionInfo session = {};
	session.codec = p_codec;
	session.width = p_width;
	session.height = p_height;
	session.dpb_slots = p_dpb_slots;
	
	if (!_create_video_session_internal(session, p_codec, p_width, p_height)) {
		ERR_PRINT("Failed to create video session");
		return 0;
	}
	
	if (!_allocate_and_bind_session_memory(session)) {
		ERR_PRINT("Failed to allocate video session memory");
		// Cleanup session
		if (session.parameters != VK_NULL_HANDLE) {
			driver->get_device_functions().DestroyVideoSessionParametersKHR(driver->get_vk_device(), session.parameters, nullptr);
		}
		if (session.session != VK_NULL_HANDLE) {
			driver->get_device_functions().DestroyVideoSessionKHR(driver->get_vk_device(), session.session, nullptr);
		}
		return 0;
	}
	
	video_sessions[session_id] = session;
	return session_id;
}

bool VulkanVideoDecoder::_create_video_session_internal(VideoSessionInfo &r_session, VideoCodec p_codec, uint32_t p_width, uint32_t p_height) {
	// Set up video profile
	VkVideoProfileInfoKHR profile_info = {};
	profile_info.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR;
	
	VkVideoDecodeAV1ProfileInfoKHR av1_profile = {};
	VkVideoDecodeH264ProfileInfoKHR h264_profile = {};
	VkVideoDecodeH265ProfileInfoKHR h265_profile = {};
	
	switch (p_codec) {
		case VIDEO_CODEC_AV1:
			profile_info.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR;
			av1_profile.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PROFILE_INFO_KHR;
			av1_profile.stdProfile = STD_VIDEO_AV1_PROFILE_MAIN;
			profile_info.pNext = &av1_profile;
			break;
		case VIDEO_CODEC_H264:
			profile_info.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;
			h264_profile.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PROFILE_INFO_KHR;
			h264_profile.stdProfileIdc = STD_VIDEO_H264_PROFILE_IDC_MAIN;
			profile_info.pNext = &h264_profile;
			break;
		case VIDEO_CODEC_H265:
			profile_info.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR;
			h265_profile.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PROFILE_INFO_KHR;
			h265_profile.stdProfileIdc = STD_VIDEO_H265_PROFILE_IDC_MAIN;
			profile_info.pNext = &h265_profile;
			break;
		default:
			return false;
	}
	
	VkVideoProfileListInfoKHR profile_list = {};
	profile_list.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR;
	profile_list.profileCount = 1;
	profile_list.pProfiles = &profile_info;
	
	// Create video session
	VkVideoSessionCreateInfoKHR session_create_info = {};
	session_create_info.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR;
	session_create_info.pNext = &profile_list;
	session_create_info.queueFamilyIndex = driver->get_video_decode_queue_family();
	session_create_info.flags = 0;
	session_create_info.pVideoProfile = &profile_info;
	session_create_info.pictureFormat = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM; // NV12
	session_create_info.maxCodedExtent = { p_width, p_height };
	session_create_info.referencePictureFormat = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
	session_create_info.maxDpbSlots = r_session.dpb_slots;
	session_create_info.maxActiveReferencePictures = r_session.dpb_slots - 1;
	
	VkResult result = driver->get_device_functions().CreateVideoSessionKHR(
		driver->get_vk_device(), &session_create_info, nullptr, &r_session.session);
	
	if (result != VK_SUCCESS) {
		ERR_PRINT("Failed to create video session: " + String::num_int64(result));
		return false;
	}
	
	// Create session parameters (empty for now)
	VkVideoSessionParametersCreateInfoKHR params_create_info = {};
	params_create_info.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR;
	params_create_info.videoSession = r_session.session;
	
	result = driver->get_device_functions().CreateVideoSessionParametersKHR(
		driver->get_vk_device(), &params_create_info, nullptr, &r_session.parameters);
	
	if (result != VK_SUCCESS) {
		ERR_PRINT("Failed to create video session parameters: " + String::num_int64(result));
		return false;
	}
	
	return true;
}

bool VulkanVideoDecoder::_allocate_and_bind_session_memory(VideoSessionInfo &r_session) {
	// Get memory requirements
	uint32_t memory_req_count = 0;
	VkResult result = driver->get_device_functions().GetVideoSessionMemoryRequirementsKHR(
		driver->get_vk_device(), r_session.session, &memory_req_count, nullptr);
	
	if (result != VK_SUCCESS || memory_req_count == 0) {
		return true; // No memory requirements
	}
	
	LocalVector<VkVideoSessionMemoryRequirementsKHR> memory_reqs;
	memory_reqs.resize(memory_req_count);
	
	for (uint32_t i = 0; i < memory_req_count; i++) {
		memory_reqs[i].sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR;
		memory_reqs[i].pNext = nullptr;
	}
	
	result = driver->get_device_functions().GetVideoSessionMemoryRequirementsKHR(
		driver->get_vk_device(), r_session.session, &memory_req_count, memory_reqs.ptr());
	
	if (result != VK_SUCCESS) {
		ERR_PRINT("Failed to get video session memory requirements");
		return false;
	}
	
	// Allocate and bind memory
	LocalVector<VkBindVideoSessionMemoryInfoKHR> bind_infos;
	bind_infos.resize(memory_req_count);
	r_session.bound_memory.resize(memory_req_count);
	
	for (uint32_t i = 0; i < memory_req_count; i++) {
		const VkVideoSessionMemoryRequirementsKHR &req = memory_reqs[i];
		
		// Find suitable memory type
		VkPhysicalDeviceMemoryProperties mem_props;
		vkGetPhysicalDeviceMemoryProperties(driver->get_vk_physical_device(), &mem_props);
		
		uint32_t memory_type_index = UINT32_MAX;
		for (uint32_t j = 0; j < mem_props.memoryTypeCount; j++) {
			if ((req.memoryRequirements.memoryTypeBits & (1 << j)) &&
				(mem_props.memoryTypes[j].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
				memory_type_index = j;
				break;
			}
		}
		
		if (memory_type_index == UINT32_MAX) {
			ERR_PRINT("Failed to find suitable memory type for video session");
			return false;
		}
		
		// Allocate memory
		VkMemoryAllocateInfo alloc_info = {};
		alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		alloc_info.allocationSize = req.memoryRequirements.size;
		alloc_info.memoryTypeIndex = memory_type_index;
		
		result = vkAllocateMemory(driver->get_vk_device(), &alloc_info, nullptr, &r_session.bound_memory[i]);
		if (result != VK_SUCCESS) {
			ERR_PRINT("Failed to allocate video session memory");
			return false;
		}
		
		// Set up bind info
		bind_infos[i].sType = VK_STRUCTURE_TYPE_BIND_VIDEO_SESSION_MEMORY_INFO_KHR;
		bind_infos[i].pNext = nullptr;
		bind_infos[i].memoryBindIndex = req.memoryBindIndex;
		bind_infos[i].memory = r_session.bound_memory[i];
		bind_infos[i].memoryOffset = 0;
		bind_infos[i].memorySize = req.memoryRequirements.size;
	}
	
	// Bind memory to session
	result = driver->get_device_functions().BindVideoSessionMemoryKHR(
		driver->get_vk_device(), r_session.session, memory_req_count, bind_infos.ptr());
	
	if (result != VK_SUCCESS) {
		ERR_PRINT("Failed to bind video session memory");
		return false;
	}
	
	return true;
}

void VulkanVideoDecoder::destroy_video_session(uint32_t p_session_id) {
	if (!video_sessions.has(p_session_id)) {
		return;
	}
	
	VideoSessionInfo &session = video_sessions[p_session_id];
	
	if (session.parameters != VK_NULL_HANDLE) {
		driver->get_device_functions().DestroyVideoSessionParametersKHR(driver->get_vk_device(), session.parameters, nullptr);
	}
	if (session.session != VK_NULL_HANDLE) {
		driver->get_device_functions().DestroyVideoSessionKHR(driver->get_vk_device(), session.session, nullptr);
	}
	
	// Free bound memory
	for (VkDeviceMemory memory : session.bound_memory) {
		vkFreeMemory(driver->get_vk_device(), memory, nullptr);
	}
	
	video_sessions.erase(p_session_id);
}

bool VulkanVideoDecoder::is_session_valid(uint32_t p_session_id) {
	return video_sessions.has(p_session_id);
}

uint32_t VulkanVideoDecoder::create_dpb_image(uint32_t p_session_id, VkFormat p_format) {
	if (!video_sessions.has(p_session_id)) {
		return 0;
	}
	
	const VideoSessionInfo &session = video_sessions[p_session_id];
	uint32_t image_id = next_image_id++;
	
	VideoImage image = {};
	image.format = p_format;
	image.width = session.width;
	image.height = session.height;
	
	if (!_create_dpb_image(image, p_format, session.width, session.height)) {
		return 0;
	}
	
	dpb_images[image_id] = image;
	return image_id;
}

uint32_t VulkanVideoDecoder::create_output_image(uint32_t p_session_id, VkFormat p_format) {
	if (!video_sessions.has(p_session_id)) {
		return 0;
	}
	
	const VideoSessionInfo &session = video_sessions[p_session_id];
	uint32_t image_id = next_image_id++;
	
	VideoImage image = {};
	image.format = p_format;
	image.width = session.width;
	image.height = session.height;
	
	if (!_create_output_image(image, p_format, session.width, session.height)) {
		return 0;
	}
	
	output_images[image_id] = image;
	return image_id;
}

bool VulkanVideoDecoder::_create_dpb_image(VideoImage &r_image, VkFormat p_format, uint32_t p_width, uint32_t p_height) {
	// Create image for DPB (Decoded Picture Buffer)
	VkImageCreateInfo image_info = {};
	image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	image_info.imageType = VK_IMAGE_TYPE_2D;
	image_info.format = p_format;
	image_info.extent = { p_width, p_height, 1 };
	image_info.mipLevels = 1;
	image_info.arrayLayers = 1;
	image_info.samples = VK_SAMPLE_COUNT_1_BIT;
	image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
	image_info.usage = VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR;
	image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	
	VkResult result = vkCreateImage(driver->get_vk_device(), &image_info, nullptr, &r_image.image);
	if (result != VK_SUCCESS) {
		return false;
	}
	
	// Allocate memory for image
	VkMemoryRequirements mem_reqs;
	vkGetImageMemoryRequirements(driver->get_vk_device(), r_image.image, &mem_reqs);
	
	VkPhysicalDeviceMemoryProperties mem_props;
	vkGetPhysicalDeviceMemoryProperties(driver->get_vk_physical_device(), &mem_props);
	
	uint32_t memory_type_index = UINT32_MAX;
	for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
		if ((mem_reqs.memoryTypeBits & (1 << i)) &&
			(mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
			memory_type_index = i;
			break;
		}
	}
	
	if (memory_type_index == UINT32_MAX) {
		vkDestroyImage(driver->get_vk_device(), r_image.image, nullptr);
		return false;
	}
	
	VkMemoryAllocateInfo alloc_info = {};
	alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	alloc_info.allocationSize = mem_reqs.size;
	alloc_info.memoryTypeIndex = memory_type_index;
	
	result = vkAllocateMemory(driver->get_vk_device(), &alloc_info, nullptr, &r_image.memory);
	if (result != VK_SUCCESS) {
		vkDestroyImage(driver->get_vk_device(), r_image.image, nullptr);
		return false;
	}
	
	result = vkBindImageMemory(driver->get_vk_device(), r_image.image, r_image.memory, 0);
	if (result != VK_SUCCESS) {
		vkFreeMemory(driver->get_vk_device(), r_image.memory, nullptr);
		vkDestroyImage(driver->get_vk_device(), r_image.image, nullptr);
		return false;
	}
	
	// Create image view
	VkImageViewCreateInfo view_info = {};
	view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	view_info.image = r_image.image;
	view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
	view_info.format = p_format;
	view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	view_info.subresourceRange.baseMipLevel = 0;
	view_info.subresourceRange.levelCount = 1;
	view_info.subresourceRange.baseArrayLayer = 0;
	view_info.subresourceRange.layerCount = 1;
	
	result = vkCreateImageView(driver->get_vk_device(), &view_info, nullptr, &r_image.view);
	if (result != VK_SUCCESS) {
		vkFreeMemory(driver->get_vk_device(), r_image.memory, nullptr);
		vkDestroyImage(driver->get_vk_device(), r_image.image, nullptr);
		return false;
	}
	
	return true;
}

bool VulkanVideoDecoder::_create_output_image(VideoImage &r_image, VkFormat p_format, uint32_t p_width, uint32_t p_height) {
	// Create image for decode output
	VkImageCreateInfo image_info = {};
	image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	image_info.imageType = VK_IMAGE_TYPE_2D;
	image_info.format = p_format;
	image_info.extent = { p_width, p_height, 1 };
	image_info.mipLevels = 1;
	image_info.arrayLayers = 1;
	image_info.samples = VK_SAMPLE_COUNT_1_BIT;
	image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
	image_info.usage = VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR | VK_IMAGE_USAGE_SAMPLED_BIT;
	image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	
	VkResult result = vkCreateImage(driver->get_vk_device(), &image_info, nullptr, &r_image.image);
	if (result != VK_SUCCESS) {
		return false;
	}
	
	// Allocate memory for image (same as DPB)
	VkMemoryRequirements mem_reqs;
	vkGetImageMemoryRequirements(driver->get_vk_device(), r_image.image, &mem_reqs);
	
	VkPhysicalDeviceMemoryProperties mem_props;
	vkGetPhysicalDeviceMemoryProperties(driver->get_vk_physical_device(), &mem_props);
	
	uint32_t memory_type_index = UINT32_MAX;
	for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
		if ((mem_reqs.memoryTypeBits & (1 << i)) &&
			(mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
			memory_type_index = i;
			break;
		}
	}
	
	if (memory_type_index == UINT32_MAX) {
		vkDestroyImage(driver->get_vk_device(), r_image.image, nullptr);
		return false;
	}
	
	VkMemoryAllocateInfo alloc_info = {};
	alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	alloc_info.allocationSize = mem_reqs.size;
	alloc_info.memoryTypeIndex = memory_type_index;
	
	result = vkAllocateMemory(driver->get_vk_device(), &alloc_info, nullptr, &r_image.memory);
	if (result != VK_SUCCESS) {
		vkDestroyImage(driver->get_vk_device(), r_image.image, nullptr);
		return false;
	}
	
	result = vkBindImageMemory(driver->get_vk_device(), r_image.image, r_image.memory, 0);
	if (result != VK_SUCCESS) {
		vkFreeMemory(driver->get_vk_device(), r_image.memory, nullptr);
		vkDestroyImage(driver->get_vk_device(), r_image.image, nullptr);
		return false;
	}
	
	// Create image view
	VkImageViewCreateInfo view_info = {};
	view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	view_info.image = r_image.image;
	view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
	view_info.format = p_format;
	view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	view_info.subresourceRange.baseMipLevel = 0;
	view_info.subresourceRange.levelCount = 1;
	view_info.subresourceRange.baseArrayLayer = 0;
	view_info.subresourceRange.layerCount = 1;
	
	result = vkCreateImageView(driver->get_vk_device(), &view_info, nullptr, &r_image.view);
	if (result != VK_SUCCESS) {
		vkFreeMemory(driver->get_vk_device(), r_image.memory, nullptr);
		vkDestroyImage(driver->get_vk_device(), r_image.image, nullptr);
		return false;
	}
	
	return true;
}

void VulkanVideoDecoder::destroy_video_image(uint32_t p_image_id) {
	if (dpb_images.has(p_image_id)) {
		VideoImage &image = dpb_images[p_image_id];
		if (image.view != VK_NULL_HANDLE) {
			vkDestroyImageView(driver->get_vk_device(), image.view, nullptr);
		}
		if (image.image != VK_NULL_HANDLE) {
			vkDestroyImage(driver->get_vk_device(), image.image, nullptr);
		}
		if (image.memory != VK_NULL_HANDLE) {
			vkFreeMemory(driver->get_vk_device(), image.memory, nullptr);
		}
		dpb_images.erase(p_image_id);
	} else if (output_images.has(p_image_id)) {
		VideoImage &image = output_images[p_image_id];
		if (image.view != VK_NULL_HANDLE) {
			vkDestroyImageView(driver->get_vk_device(), image.view, nullptr);
		}
		if (image.image != VK_NULL_HANDLE) {
			vkDestroyImage(driver->get_vk_device(), image.image, nullptr);
		}
		if (image.memory != VK_NULL_HANDLE) {
			vkFreeMemory(driver->get_vk_device(), image.memory, nullptr);
		}
		output_images.erase(p_image_id);
	}
}

VkImage VulkanVideoDecoder::get_image_handle(uint32_t p_image_id) {
	if (dpb_images.has(p_image_id)) {
		return dpb_images[p_image_id].image;
	}
	if (output_images.has(p_image_id)) {
		return output_images[p_image_id].image;
	}
	return VK_NULL_HANDLE;
}

VkImageView VulkanVideoDecoder::get_image_view(uint32_t p_image_id) {
	if (dpb_images.has(p_image_id)) {
		return dpb_images[p_image_id].view;
	}
	if (output_images.has(p_image_id)) {
		return output_images[p_image_id].view;
	}
	return VK_NULL_HANDLE;
}

uint32_t VulkanVideoDecoder::create_video_buffer(uint64_t p_size) {
	ERR_FAIL_NULL_V(driver, 0);
	
	uint32_t buffer_id = next_buffer_id++;
	VideoBuffer buffer = {};
	buffer.size = p_size;
	
	if (!_create_video_buffer_internal(buffer, p_size)) {
		return 0;
	}
	
	video_buffers[buffer_id] = buffer;
	return buffer_id;
}

bool VulkanVideoDecoder::_create_video_buffer_internal(VideoBuffer &r_buffer, uint64_t p_size) {
	// Create buffer for bitstream data
	VkBufferCreateInfo buffer_info = {};
	buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.size = p_size;
	buffer_info.usage = VK_BUFFER_USAGE_VIDEO_DECODE_SRC_BIT_KHR;
	buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	
	VkResult result = vkCreateBuffer(driver->get_vk_device(), &buffer_info, nullptr, &r_buffer.buffer);
	if (result != VK_SUCCESS) {
		return false;
	}
	
	// Allocate memory for buffer
	VkMemoryRequirements mem_reqs;
	vkGetBufferMemoryRequirements(driver->get_vk_device(), r_buffer.buffer, &mem_reqs);
	
	VkPhysicalDeviceMemoryProperties mem_props;
	vkGetPhysicalDeviceMemoryProperties(driver->get_vk_physical_device(), &mem_props);
	
	uint32_t memory_type_index = UINT32_MAX;
	for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
		if ((mem_reqs.memoryTypeBits & (1 << i)) &&
			(mem_props.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
			memory_type_index = i;
			break;
		}
	}
	
	if (memory_type_index == UINT32_MAX) {
		vkDestroyBuffer(driver->get_vk_device(), r_buffer.buffer, nullptr);
		return false;
	}
	
	VkMemoryAllocateInfo alloc_info = {};
	alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	alloc_info.allocationSize = mem_reqs.size;
	alloc_info.memoryTypeIndex = memory_type_index;
	
	result = vkAllocateMemory(driver->get_vk_device(), &alloc_info, nullptr, &r_buffer.memory);
	if (result != VK_SUCCESS) {
		vkDestroyBuffer(driver->get_vk_device(), r_buffer.buffer, nullptr);
		return false;
	}
	
	result = vkBindBufferMemory(driver->get_vk_device(), r_buffer.buffer, r_buffer.memory, 0);
	if (result != VK_SUCCESS) {
		vkFreeMemory(driver->get_vk_device(), r_buffer.memory, nullptr);
		vkDestroyBuffer(driver->get_vk_device(), r_buffer.buffer, nullptr);
		return false;
	}
	
	// Map memory for CPU access
	result = vkMapMemory(driver->get_vk_device(), r_buffer.memory, 0, p_size, 0, &r_buffer.mapped_ptr);
	if (result != VK_SUCCESS) {
		vkFreeMemory(driver->get_vk_device(), r_buffer.memory, nullptr);
		vkDestroyBuffer(driver->get_vk_device(), r_buffer.buffer, nullptr);
		return false;
	}
	
	return true;
}

void VulkanVideoDecoder::destroy_video_buffer(uint32_t p_buffer_id) {
	if (!video_buffers.has(p_buffer_id)) {
		return;
	}
	
	VideoBuffer &buffer = video_buffers[p_buffer_id];
	if (buffer.mapped_ptr) {
		vkUnmapMemory(driver->get_vk_device(), buffer.memory);
	}
	if (buffer.buffer != VK_NULL_HANDLE) {
		vkDestroyBuffer(driver->get_vk_device(), buffer.buffer, nullptr);
	}
	if (buffer.memory != VK_NULL_HANDLE) {
		vkFreeMemory(driver->get_vk_device(), buffer.memory, nullptr);
	}
	
	video_buffers.erase(p_buffer_id);
}

bool VulkanVideoDecoder::update_video_buffer(uint32_t p_buffer_id, const Vector<uint8_t> &p_data, uint64_t p_offset) {
	if (!video_buffers.has(p_buffer_id)) {
		return false;
	}
	
	VideoBuffer &buffer = video_buffers[p_buffer_id];
	if (!buffer.mapped_ptr || p_offset + p_data.size() > buffer.size) {
		return false;
	}
	
	memcpy(static_cast<uint8_t *>(buffer.mapped_ptr) + p_offset, p_data.ptr(), p_data.size());
	return true;
}

VkBuffer VulkanVideoDecoder::get_buffer_handle(uint32_t p_buffer_id) {
	if (video_buffers.has(p_buffer_id)) {
		return video_buffers[p_buffer_id].buffer;
	}
	return VK_NULL_HANDLE;
}

bool VulkanVideoDecoder::begin_video_coding(VkCommandBuffer p_cmd_buffer, uint32_t p_session_id) {
	if (!video_sessions.has(p_session_id)) {
		return false;
	}
	
	const VideoSessionInfo &session = video_sessions[p_session_id];
	
	VkVideoBeginCodingInfoKHR begin_info = {};
	begin_info.sType = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR;
	begin_info.videoSession = session.session;
	begin_info.videoSessionParameters = session.parameters;
	
	driver->get_device_functions().CmdBeginVideoCodingKHR(p_cmd_buffer, &begin_info);
	return true;
}

bool VulkanVideoDecoder::end_video_coding(VkCommandBuffer p_cmd_buffer) {
	VkVideoEndCodingInfoKHR end_info = {};
	end_info.sType = VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR;
	
	driver->get_device_functions().CmdEndVideoCodingKHR(p_cmd_buffer, &end_info);
	return true;
}

bool VulkanVideoDecoder::decode_frame(VkCommandBuffer p_cmd_buffer, uint32_t p_session_id, uint32_t p_bitstream_buffer_id, 
									  uint32_t p_output_image_id, const Vector<uint32_t> &p_reference_images) {
	if (!video_sessions.has(p_session_id) || !video_buffers.has(p_bitstream_buffer_id) || !output_images.has(p_output_image_id)) {
		return false;
	}
	
	const VideoSessionInfo &session = video_sessions[p_session_id];
	const VideoBuffer &bitstream_buffer = video_buffers[p_bitstream_buffer_id];
	const VideoImage &output_image = output_images[p_output_image_id];
	
	// Set up decode info
	VkVideoDecodeInfoKHR decode_info = {};
	decode_info.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR;
	decode_info.flags = 0;
	
	// Bitstream buffer info
	decode_info.srcBuffer = bitstream_buffer.buffer;
	decode_info.srcBufferOffset = 0;
	decode_info.srcBufferRange = bitstream_buffer.size;
	
	// Output image info
	VkVideoPictureResourceInfoKHR dst_picture_resource = {};
	dst_picture_resource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
	dst_picture_resource.codedOffset = { 0, 0 };
	dst_picture_resource.codedExtent = { session.width, session.height };
	dst_picture_resource.baseArrayLayer = 0;
	dst_picture_resource.imageViewBinding = output_image.view;
	
	decode_info.dstPictureResource = dst_picture_resource;
	
	// Reference pictures (if any)
	LocalVector<VkVideoPictureResourceInfoKHR> reference_resources;
	LocalVector<VkVideoReferenceSlotInfoKHR> reference_slots;
	
	if (!p_reference_images.is_empty()) {
		reference_resources.resize(p_reference_images.size());
		reference_slots.resize(p_reference_images.size());
		
		for (int i = 0; i < p_reference_images.size(); i++) {
			uint32_t ref_image_id = p_reference_images[i];
			if (dpb_images.has(ref_image_id)) {
				const VideoImage &ref_image = dpb_images[ref_image_id];
				
				reference_resources[i].sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
				reference_resources[i].codedOffset = { 0, 0 };
				reference_resources[i].codedExtent = { session.width, session.height };
				reference_resources[i].baseArrayLayer = 0;
				reference_resources[i].imageViewBinding = ref_image.view;
				
				reference_slots[i].sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
				reference_slots[i].slotIndex = i;
				reference_slots[i].pPictureResource = &reference_resources[i];
			}
		}
		
		decode_info.pSetupReferenceSlot = nullptr; // For now
		decode_info.referenceSlotCount = reference_slots.size();
		decode_info.pReferenceSlots = reference_slots.ptr();
	}
	
	// Execute decode command
	driver->get_device_functions().CmdDecodeVideoKHR(p_cmd_buffer, &decode_info);
	return true;
}

VkSamplerYcbcrConversion VulkanVideoDecoder::create_ycbcr_conversion(VkFormat p_format) {
	VkSamplerYcbcrConversionCreateInfo conversion_info = {};
	conversion_info.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO;
	conversion_info.format = p_format;
	conversion_info.ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709;
	conversion_info.ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_NARROW;
	conversion_info.components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
	conversion_info.xChromaOffset = VK_CHROMA_LOCATION_COSITED_EVEN;
	conversion_info.yChromaOffset = VK_CHROMA_LOCATION_COSITED_EVEN;
	conversion_info.chromaFilter = VK_FILTER_LINEAR;
	conversion_info.forceExplicitReconstruction = VK_FALSE;
	
	VkSamplerYcbcrConversion conversion = VK_NULL_HANDLE;
	VkResult result = driver->get_device_functions().CreateSamplerYcbcrConversionKHR(driver->get_vk_device(), &conversion_info, nullptr, &conversion);
	
	if (result != VK_SUCCESS) {
		return VK_NULL_HANDLE;
	}
	
	return conversion;
}

void VulkanVideoDecoder::destroy_ycbcr_conversion(VkSamplerYcbcrConversion p_conversion) {
	if (p_conversion != VK_NULL_HANDLE) {
		driver->get_device_functions().DestroySamplerYcbcrConversionKHR(driver->get_vk_device(), p_conversion, nullptr);
	}
}

#endif // VULKAN_ENABLED
