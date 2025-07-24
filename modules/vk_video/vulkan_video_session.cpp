/**************************************************************************/
/*  vulkan_video_session.cpp                                             */
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

#include "vulkan_video_session.h"

#ifdef VULKAN_ENABLED

#include "core/config/project_settings.h"
#include "core/string/print_string.h"

VulkanVideoSession::VulkanVideoSession() {
}

VulkanVideoSession::~VulkanVideoSession() {
	finalize();
}

Error VulkanVideoSession::initialize(VkDevice p_device, VkPhysicalDevice p_physical_device, uint32_t p_video_queue_family) {
	ERR_FAIL_COND_V(p_device == VK_NULL_HANDLE, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_physical_device == VK_NULL_HANDLE, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_video_queue_family == UINT32_MAX, ERR_INVALID_PARAMETER);

	vk_device = p_device;
	physical_device = p_physical_device;
	video_decode_queue_family = p_video_queue_family;

	return OK;
}

void VulkanVideoSession::finalize() {
	destroy_video_session();
	
	vk_device = VK_NULL_HANDLE;
	physical_device = VK_NULL_HANDLE;
	video_decode_queue_family = UINT32_MAX;
}

bool VulkanVideoSession::load_function_pointers(VkInstance p_instance) {
	ERR_FAIL_COND_V(p_instance == VK_NULL_HANDLE, false);

	// Load instance-level functions
	GetPhysicalDeviceVideoCapabilitiesKHR = (PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR)vkGetInstanceProcAddr(p_instance, "vkGetPhysicalDeviceVideoCapabilitiesKHR");
	GetPhysicalDeviceVideoFormatPropertiesKHR = (PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR)vkGetInstanceProcAddr(p_instance, "vkGetPhysicalDeviceVideoFormatPropertiesKHR");

	// Load device-level functions
	CreateVideoSessionKHR = (PFN_vkCreateVideoSessionKHR)vkGetDeviceProcAddr(vk_device, "vkCreateVideoSessionKHR");
	DestroyVideoSessionKHR = (PFN_vkDestroyVideoSessionKHR)vkGetDeviceProcAddr(vk_device, "vkDestroyVideoSessionKHR");
	CreateVideoSessionParametersKHR = (PFN_vkCreateVideoSessionParametersKHR)vkGetDeviceProcAddr(vk_device, "vkCreateVideoSessionParametersKHR");
	DestroyVideoSessionParametersKHR = (PFN_vkDestroyVideoSessionParametersKHR)vkGetDeviceProcAddr(vk_device, "vkDestroyVideoSessionParametersKHR");
	UpdateVideoSessionParametersKHR = (PFN_vkUpdateVideoSessionParametersKHR)vkGetDeviceProcAddr(vk_device, "vkUpdateVideoSessionParametersKHR");
	GetVideoSessionMemoryRequirementsKHR = (PFN_vkGetVideoSessionMemoryRequirementsKHR)vkGetDeviceProcAddr(vk_device, "vkGetVideoSessionMemoryRequirementsKHR");
	BindVideoSessionMemoryKHR = (PFN_vkBindVideoSessionMemoryKHR)vkGetDeviceProcAddr(vk_device, "vkBindVideoSessionMemoryKHR");

	// Validate critical functions are loaded
	bool functions_loaded = GetPhysicalDeviceVideoCapabilitiesKHR != nullptr &&
							CreateVideoSessionKHR != nullptr &&
							DestroyVideoSessionKHR != nullptr &&
							GetVideoSessionMemoryRequirementsKHR != nullptr &&
							BindVideoSessionMemoryKHR != nullptr;

	if (!functions_loaded) {
		ERR_PRINT("VulkanVideoSession: Failed to load required Vulkan Video function pointers");
		return false;
	}

	print_verbose("VulkanVideoSession: Successfully loaded Vulkan Video function pointers");
	return true;
}

bool VulkanVideoSession::query_video_capabilities(VideoCodec p_codec, VideoCapabilities *p_capabilities) {
	ERR_FAIL_COND_V(!GetPhysicalDeviceVideoCapabilitiesKHR, false);
	ERR_FAIL_NULL_V(p_capabilities, false);

	// Setup video profile based on codec
	VkVideoDecodeAV1ProfileInfoKHR av1_profile = {};
	av1_profile.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PROFILE_INFO_KHR;
	av1_profile.stdProfile = STD_VIDEO_AV1_PROFILE_MAIN;
	av1_profile.filmGrainSupport = VK_FALSE;

	VkVideoProfileInfoKHR profile_info = {};
	profile_info.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR;
	profile_info.videoCodecOperation = get_vulkan_codec_operation(p_codec);
	profile_info.chromaSubsampling = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR;
	profile_info.lumaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
	profile_info.chromaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;

	if (p_codec == VIDEO_CODEC_AV1) {
		profile_info.pNext = &av1_profile;
	}

	// Setup capability query structures
	p_capabilities->av1_caps.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_CAPABILITIES_KHR;
	p_capabilities->av1_caps.pNext = nullptr;

	p_capabilities->decode_caps.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR;
	p_capabilities->decode_caps.pNext = &p_capabilities->av1_caps;

	p_capabilities->video_caps.sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR;
	p_capabilities->video_caps.pNext = &p_capabilities->decode_caps;

	// Query capabilities
	VkResult result = GetPhysicalDeviceVideoCapabilitiesKHR(physical_device, &profile_info, &p_capabilities->video_caps);
	if (result != VK_SUCCESS) {
		print_verbose("VulkanVideoSession: Failed to query video capabilities for " + get_codec_name(p_codec) + " (VkResult: " + itos(result) + ")");
		p_capabilities->is_valid = false;
		return false;
	}

	p_capabilities->is_valid = true;

	// Log capabilities for debugging
	print_verbose("VulkanVideoSession: " + get_codec_name(p_codec) + " Capabilities:");
	print_verbose("  Max coded extent: " + itos(p_capabilities->video_caps.maxCodedExtent.width) + "x" + itos(p_capabilities->video_caps.maxCodedExtent.height));
	print_verbose("  Max DPB slots: " + itos(p_capabilities->video_caps.maxDpbSlots));
	print_verbose("  Max active references: " + itos(p_capabilities->video_caps.maxActiveReferencePictures));
	
	if (p_codec == VIDEO_CODEC_AV1) {
		print_verbose("  Max AV1 level: " + itos(p_capabilities->av1_caps.maxLevel));
	}

	return true;
}

Error VulkanVideoSession::create_video_session(const VideoSessionCreateInfo &p_create_info) {
	ERR_FAIL_COND_V(!CreateVideoSessionKHR, ERR_UNAVAILABLE);
	ERR_FAIL_COND_V(session_info.is_initialized, ERR_ALREADY_IN_USE);

	// Query capabilities first
	VideoCapabilities capabilities;
	if (!query_video_capabilities(p_create_info.codec, &capabilities)) {
		return ERR_CANT_CREATE;
	}

	// Validate requested parameters against capabilities
	ERR_FAIL_COND_V_MSG(p_create_info.max_width > capabilities.video_caps.maxCodedExtent.width, ERR_INVALID_PARAMETER,
			"Requested width " + itos(p_create_info.max_width) + " exceeds maximum " + itos(capabilities.video_caps.maxCodedExtent.width));
	ERR_FAIL_COND_V_MSG(p_create_info.max_height > capabilities.video_caps.maxCodedExtent.height, ERR_INVALID_PARAMETER,
			"Requested height " + itos(p_create_info.max_height) + " exceeds maximum " + itos(capabilities.video_caps.maxCodedExtent.height));
	ERR_FAIL_COND_V_MSG(p_create_info.max_dpb_slots > capabilities.video_caps.maxDpbSlots, ERR_INVALID_PARAMETER,
			"Requested DPB slots " + itos(p_create_info.max_dpb_slots) + " exceeds maximum " + itos(capabilities.video_caps.maxDpbSlots));

	// Setup video profile
	session_info.av1_profile_info.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PROFILE_INFO_KHR;
	session_info.av1_profile_info.stdProfile = STD_VIDEO_AV1_PROFILE_MAIN;
	session_info.av1_profile_info.filmGrainSupport = p_create_info.enable_film_grain ? VK_TRUE : VK_FALSE;

	session_info.profile_info.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR;
	session_info.profile_info.videoCodecOperation = get_vulkan_codec_operation(p_create_info.codec);
	session_info.profile_info.chromaSubsampling = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR;
	session_info.profile_info.lumaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
	session_info.profile_info.chromaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;

	if (p_create_info.codec == VIDEO_CODEC_AV1) {
		session_info.profile_info.pNext = &session_info.av1_profile_info;
	}

	// Create video session
	VkVideoSessionCreateInfoKHR session_create_info = {};
	session_create_info.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR;
	session_create_info.queueFamilyIndex = video_decode_queue_family;
	session_create_info.pVideoProfile = &session_info.profile_info;
	session_create_info.pictureFormat = p_create_info.output_format;
	session_create_info.maxCodedExtent.width = p_create_info.max_width;
	session_create_info.maxCodedExtent.height = p_create_info.max_height;
	session_create_info.referencePictureFormat = p_create_info.dpb_format;
	session_create_info.maxDpbSlots = p_create_info.max_dpb_slots;
	session_create_info.maxActiveReferencePictures = p_create_info.max_active_references;

	VkResult result = CreateVideoSessionKHR(vk_device, &session_create_info, nullptr, &session_info.vk_video_session);
	ERR_FAIL_COND_V_MSG(result != VK_SUCCESS, ERR_CANT_CREATE,
			"Failed to create video session (VkResult: " + itos(result) + ")");

	// Store session information
	session_info.capabilities = capabilities;
	session_info.width = p_create_info.max_width;
	session_info.height = p_create_info.max_height;
	session_info.codec = p_create_info.codec;

	// Bind memory to video session
	Error memory_result = _bind_video_session_memory();
	if (memory_result != OK) {
		destroy_video_session();
		return memory_result;
	}

	session_info.is_initialized = true;
	print_verbose("VulkanVideoSession: Successfully created " + get_codec_name(p_create_info.codec) + " video session (" + itos(p_create_info.max_width) + "x" + itos(p_create_info.max_height) + ")");

	return OK;
}

void VulkanVideoSession::destroy_video_session() {
	if (!session_info.is_initialized) {
		return;
	}

	// Free bound memory
	for (uint32_t i = 0; i < session_info.bound_memory.size(); i++) {
		if (session_info.bound_memory[i] != VK_NULL_HANDLE) {
			vkFreeMemory(vk_device, session_info.bound_memory[i], nullptr);
		}
	}
	session_info.bound_memory.clear();

	// Destroy session parameters
	if (session_info.vk_session_parameters != VK_NULL_HANDLE && DestroyVideoSessionParametersKHR) {
		DestroyVideoSessionParametersKHR(vk_device, session_info.vk_session_parameters, nullptr);
		session_info.vk_session_parameters = VK_NULL_HANDLE;
	}

	// Destroy video session
	if (session_info.vk_video_session != VK_NULL_HANDLE && DestroyVideoSessionKHR) {
		DestroyVideoSessionKHR(vk_device, session_info.vk_video_session, nullptr);
		session_info.vk_video_session = VK_NULL_HANDLE;
	}

	session_info.is_initialized = false;
	print_verbose("VulkanVideoSession: Video session destroyed");
}

bool VulkanVideoSession::_query_av1_decode_capabilities(VideoCapabilities *p_capabilities) {
	return query_video_capabilities(VIDEO_CODEC_AV1, p_capabilities);
}

Error VulkanVideoSession::_bind_video_session_memory() {
	ERR_FAIL_COND_V(!GetVideoSessionMemoryRequirementsKHR, ERR_UNAVAILABLE);
	ERR_FAIL_COND_V(!BindVideoSessionMemoryKHR, ERR_UNAVAILABLE);
	ERR_FAIL_COND_V(session_info.vk_video_session == VK_NULL_HANDLE, ERR_UNCONFIGURED);

	// Query memory requirements
	uint32_t memory_req_count = 0;
	VkResult result = GetVideoSessionMemoryRequirementsKHR(vk_device, session_info.vk_video_session, &memory_req_count, nullptr);
	ERR_FAIL_COND_V(result != VK_SUCCESS, ERR_CANT_CREATE);

	if (memory_req_count == 0) {
		print_verbose("VulkanVideoSession: No memory binding required for video session");
		return OK;
	}

	// Get memory requirements
	LocalVector<VkVideoSessionMemoryRequirementsKHR> memory_requirements;
	memory_requirements.resize(memory_req_count);
	for (uint32_t i = 0; i < memory_req_count; i++) {
		memory_requirements[i].sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR;
		memory_requirements[i].pNext = nullptr;
	}

	result = GetVideoSessionMemoryRequirementsKHR(vk_device, session_info.vk_video_session, &memory_req_count, memory_requirements.ptr());
	ERR_FAIL_COND_V(result != VK_SUCCESS, ERR_CANT_CREATE);

	// Allocate and bind memory
	session_info.bound_memory.resize(memory_req_count);
	LocalVector<VkBindVideoSessionMemoryInfoKHR> bind_infos;
	bind_infos.resize(memory_req_count);

	for (uint32_t i = 0; i < memory_req_count; i++) {
		const VkVideoSessionMemoryRequirementsKHR &req = memory_requirements[i];

		VkMemoryAllocateInfo alloc_info = {};
		alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		alloc_info.allocationSize = req.memoryRequirements.size;
		alloc_info.memoryTypeIndex = _find_memory_type_index(req.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		result = vkAllocateMemory(vk_device, &alloc_info, nullptr, &session_info.bound_memory[i]);
		ERR_FAIL_COND_V(result != VK_SUCCESS, ERR_CANT_CREATE);

		bind_infos[i].sType = VK_STRUCTURE_TYPE_BIND_VIDEO_SESSION_MEMORY_INFO_KHR;
		bind_infos[i].pNext = nullptr;
		bind_infos[i].memoryBindIndex = req.memoryBindIndex;
		bind_infos[i].memory = session_info.bound_memory[i];
		bind_infos[i].memoryOffset = 0;
		bind_infos[i].memorySize = req.memoryRequirements.size;
	}

	// Bind all memory at once
	result = BindVideoSessionMemoryKHR(vk_device, session_info.vk_video_session, bind_infos.size(), bind_infos.ptr());
	ERR_FAIL_COND_V(result != VK_SUCCESS, ERR_CANT_CREATE);

	print_verbose("VulkanVideoSession: Successfully bound " + itos(memory_req_count) + " memory objects to video session");
	return OK;
}

uint32_t VulkanVideoSession::_find_memory_type_index(uint32_t type_filter, VkMemoryPropertyFlags properties) {
	VkPhysicalDeviceMemoryProperties mem_properties;
	vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

	for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
		if ((type_filter & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}

	ERR_FAIL_V_MSG(UINT32_MAX, "VulkanVideoSession: Failed to find suitable memory type");
}

String VulkanVideoSession::get_codec_name(VideoCodec p_codec) {
	switch (p_codec) {
		case VIDEO_CODEC_AV1:
			return "AV1";
		case VIDEO_CODEC_H264:
			return "H.264";
		case VIDEO_CODEC_H265:
			return "H.265";
		default:
			return "Unknown";
	}
}

VkVideoCodecOperationFlagBitsKHR VulkanVideoSession::get_vulkan_codec_operation(VideoCodec p_codec) {
	switch (p_codec) {
		case VIDEO_CODEC_AV1:
			return VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR;
		case VIDEO_CODEC_H264:
			return VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;
		case VIDEO_CODEC_H265:
			return VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR;
		default:
			return VK_VIDEO_CODEC_OPERATION_NONE_KHR;
	}
}

Dictionary VulkanVideoSession::get_capabilities_info() const {
	Dictionary info;
	
	if (!session_info.capabilities.is_valid) {
		info["valid"] = false;
		return info;
	}

	const VideoCapabilities &caps = session_info.capabilities;
	
	info["valid"] = true;
	info["codec"] = get_codec_name(session_info.codec);
	info["max_coded_extent_width"] = caps.video_caps.maxCodedExtent.width;
	info["max_coded_extent_height"] = caps.video_caps.maxCodedExtent.height;
	info["max_dpb_slots"] = caps.video_caps.maxDpbSlots;
	info["max_active_reference_pictures"] = caps.video_caps.maxActiveReferencePictures;
	
	if (session_info.codec == VIDEO_CODEC_AV1) {
		info["max_av1_level"] = caps.av1_caps.maxLevel;
	}
	
	return info;
}

#endif // VULKAN_ENABLED
