/**************************************************************************/
/*  vulkan_video_context.cpp                                             */
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

#include "vulkan_video_context.h"

#ifdef VULKAN_ENABLED

#include "core/error/error_macros.h"
#include "core/object/class_db.h"
#include "servers/rendering_server.h"

VulkanVideoContext::VulkanVideoContext() {
}

VulkanVideoContext::~VulkanVideoContext() {
	cleanup();
}

void VulkanVideoContext::_bind_methods() {
	// Note: initialize method not bound due to RenderingContextDriverVulkan* parameter type issues
	// ClassDB::bind_method(D_METHOD("initialize", "context_driver"), &VulkanVideoContext::initialize);
	ClassDB::bind_method(D_METHOD("cleanup"), &VulkanVideoContext::cleanup);
	ClassDB::bind_method(D_METHOD("is_initialized"), &VulkanVideoContext::is_initialized);
	
	ClassDB::bind_method(D_METHOD("check_video_support"), &VulkanVideoContext::check_video_support);
	ClassDB::bind_method(D_METHOD("detect_video_hardware"), &VulkanVideoContext::detect_video_hardware);
	ClassDB::bind_method(D_METHOD("is_codec_supported", "profile", "operation"), &VulkanVideoContext::is_codec_supported);
	ClassDB::bind_method(D_METHOD("get_supported_profiles"), &VulkanVideoContext::get_supported_profiles);
}

bool VulkanVideoContext::initialize(RenderingContextDriverVulkan *p_context_driver) {
	ERR_FAIL_NULL_V(p_context_driver, false);
	
	context_driver = p_context_driver;
	
	// Get Vulkan device and physical device from context
	physical_device = context_driver->physical_device_get(0);
	
	// We need to get the VkDevice from the rendering device driver
	// For now, we'll mark this as not implemented until we have proper access
	device = VK_NULL_HANDLE; // TODO: Get actual device handle
	
	ERR_FAIL_COND_V(device == VK_NULL_HANDLE, false);
	ERR_FAIL_COND_V(physical_device == VK_NULL_HANDLE, false);
	
	// Load video extension function pointers
	if (!_load_video_function_pointers()) {
		WARN_PRINT("Failed to load Vulkan Video extension functions");
		return false;
	}
	
	// Check if video extensions are supported
	if (!check_video_support()) {
		WARN_PRINT("Vulkan Video extensions not supported on this device");
		return false;
	}
	
	// Detect video hardware capabilities
	if (!detect_video_hardware()) {
		WARN_PRINT("Failed to detect video hardware capabilities");
		return false;
	}
	
	// Find video queue families
	if (!_find_video_queue_families()) {
		WARN_PRINT("No video queue families found");
		return false;
	}
	
	// Create video command pool
	if (!_create_video_command_pool()) {
		WARN_PRINT("Failed to create video command pool");
		return false;
	}
	
	initialized = true;
	print_line("Vulkan Video context initialized successfully");
	return true;
}

void VulkanVideoContext::cleanup() {
	if (!initialized) {
		return;
	}
	
	// Cleanup all video resources
	_cleanup_video_sessions();
	_cleanup_video_images();
	_cleanup_video_buffers();
	
	// Destroy command pool
	if (video_command_pool != VK_NULL_HANDLE) {
		vkDestroyCommandPool(device, video_command_pool, nullptr);
		video_command_pool = VK_NULL_HANDLE;
	}
	
	// Reset state
	video_queue = VK_NULL_HANDLE;
	device = VK_NULL_HANDLE;
	physical_device = VK_NULL_HANDLE;
	context_driver = nullptr;
	initialized = false;
	
	print_line("Vulkan Video context cleaned up");
}

bool VulkanVideoContext::_load_video_function_pointers() {
	ERR_FAIL_COND_V(device == VK_NULL_HANDLE, false);
	
	// Load core video functions
	video_functions.vkGetPhysicalDeviceVideoCapabilitiesKHR = 
		(PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR)vkGetDeviceProcAddr(device, "vkGetPhysicalDeviceVideoCapabilitiesKHR");
	video_functions.vkGetPhysicalDeviceVideoFormatPropertiesKHR = 
		(PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR)vkGetDeviceProcAddr(device, "vkGetPhysicalDeviceVideoFormatPropertiesKHR");
	video_functions.vkCreateVideoSessionKHR = 
		(PFN_vkCreateVideoSessionKHR)vkGetDeviceProcAddr(device, "vkCreateVideoSessionKHR");
	video_functions.vkDestroyVideoSessionKHR = 
		(PFN_vkDestroyVideoSessionKHR)vkGetDeviceProcAddr(device, "vkDestroyVideoSessionKHR");
	video_functions.vkCreateVideoSessionParametersKHR = 
		(PFN_vkCreateVideoSessionParametersKHR)vkGetDeviceProcAddr(device, "vkCreateVideoSessionParametersKHR");
	video_functions.vkDestroyVideoSessionParametersKHR = 
		(PFN_vkDestroyVideoSessionParametersKHR)vkGetDeviceProcAddr(device, "vkDestroyVideoSessionParametersKHR");
	video_functions.vkUpdateVideoSessionParametersKHR = 
		(PFN_vkUpdateVideoSessionParametersKHR)vkGetDeviceProcAddr(device, "vkUpdateVideoSessionParametersKHR");
	video_functions.vkGetVideoSessionMemoryRequirementsKHR = 
		(PFN_vkGetVideoSessionMemoryRequirementsKHR)vkGetDeviceProcAddr(device, "vkGetVideoSessionMemoryRequirementsKHR");
	video_functions.vkBindVideoSessionMemoryKHR = 
		(PFN_vkBindVideoSessionMemoryKHR)vkGetDeviceProcAddr(device, "vkBindVideoSessionMemoryKHR");
	
	// Load video decode functions
	video_functions.vkCmdBeginVideoCodingKHR = 
		(PFN_vkCmdBeginVideoCodingKHR)vkGetDeviceProcAddr(device, "vkCmdBeginVideoCodingKHR");
	video_functions.vkCmdEndVideoCodingKHR = 
		(PFN_vkCmdEndVideoCodingKHR)vkGetDeviceProcAddr(device, "vkCmdEndVideoCodingKHR");
	video_functions.vkCmdControlVideoCodingKHR = 
		(PFN_vkCmdControlVideoCodingKHR)vkGetDeviceProcAddr(device, "vkCmdControlVideoCodingKHR");
	video_functions.vkCmdDecodeVideoKHR = 
		(PFN_vkCmdDecodeVideoKHR)vkGetDeviceProcAddr(device, "vkCmdDecodeVideoKHR");
	
	// Load video encode functions (optional for future use)
	video_functions.vkCmdEncodeVideoKHR = 
		(PFN_vkCmdEncodeVideoKHR)vkGetDeviceProcAddr(device, "vkCmdEncodeVideoKHR");
	
	// Check if core functions are loaded
	video_functions.is_loaded = (
		video_functions.vkGetPhysicalDeviceVideoCapabilitiesKHR != nullptr &&
		video_functions.vkCreateVideoSessionKHR != nullptr &&
		video_functions.vkDestroyVideoSessionKHR != nullptr &&
		video_functions.vkCmdBeginVideoCodingKHR != nullptr &&
		video_functions.vkCmdEndVideoCodingKHR != nullptr &&
		video_functions.vkCmdDecodeVideoKHR != nullptr
	);
	
	return video_functions.is_loaded;
}

bool VulkanVideoContext::check_video_support() {
	if (!video_functions.is_loaded) {
		return false;
	}
	
	// Check if video extensions are available
	uint32_t extension_count = 0;
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);
	
	Vector<VkExtensionProperties> extensions;
	extensions.resize(extension_count);
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, extensions.ptrw());
	
	bool has_video_queue = false;
	bool has_video_decode_queue = false;
	bool has_video_decode_av1 = false;
	
	for (const VkExtensionProperties &ext : extensions) {
		if (strcmp(ext.extensionName, VK_KHR_VIDEO_QUEUE_EXTENSION_NAME) == 0) {
			has_video_queue = true;
		} 
		if (strcmp(ext.extensionName, VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME) == 0) {
			has_video_decode_queue = true;
		} 
		if (strcmp(ext.extensionName, VK_KHR_VIDEO_DECODE_AV1_EXTENSION_NAME) == 0) {
			has_video_decode_av1 = true;
		}
	}
	
	return has_video_queue && has_video_decode_queue && has_video_decode_av1;
}

bool VulkanVideoContext::detect_video_hardware() {
	ERR_FAIL_COND_V(!video_functions.is_loaded, false);
	
	// Query queue family properties
	uint32_t queue_family_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
	
	Vector<VkQueueFamilyProperties> queue_families;
	queue_families.resize(queue_family_count);
	vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.ptrw());
	
	// Find video queue families
	for (uint32_t i = 0; i < queue_family_count; i++) {
		const VkQueueFamilyProperties &props = queue_families[i];
		
		if (props.queueFlags & VK_QUEUE_VIDEO_DECODE_BIT_KHR) {
			hardware_info.decode_queue_supported = true;
			hardware_info.decode_queue_family = i;
			hardware_info.video_queue_family = i; // Use decode queue as video queue
			hardware_info.video_queue_supported = true;
		}
		
		if (props.queueFlags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR) {
			hardware_info.encode_queue_supported = true;
			hardware_info.encode_queue_family = i;
		}
	}
	
	// Query video capabilities for AV1 decode
	if (hardware_info.decode_queue_supported) {
		_query_video_capabilities();
	}
	
	return hardware_info.video_queue_supported;
}

bool VulkanVideoContext::_query_video_capabilities() {
	// Query AV1 decode capabilities
	VkVideoProfileInfoKHR profile_info = {};
	profile_info.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR;
	profile_info.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR;
	
	VkVideoDecodeAV1ProfileInfoKHR av1_profile = {};
	av1_profile.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PROFILE_INFO_KHR;
	av1_profile.stdProfile = STD_VIDEO_AV1_PROFILE_MAIN;
	profile_info.pNext = &av1_profile;
	
	VkVideoCapabilitiesKHR capabilities = {};
	capabilities.sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR;
	
	if (video_functions.vkGetPhysicalDeviceVideoCapabilitiesKHR) {
		VkResult result = video_functions.vkGetPhysicalDeviceVideoCapabilitiesKHR(
			physical_device, &profile_info, &capabilities);
		
		if (result == VK_SUCCESS) {
			hardware_info.max_coded_extent_width = capabilities.maxCodedExtent.width;
			hardware_info.max_coded_extent_height = capabilities.maxCodedExtent.height;
			hardware_info.max_dpb_slots = capabilities.maxDpbSlots;
			hardware_info.max_active_reference_pictures = capabilities.maxActiveReferencePictures;
			
			// Add AV1 decode to supported codecs
			hardware_info.supported_decode_codecs.push_back(VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR);
			
			return true;
		}
	}
	
	return false;
}

bool VulkanVideoContext::_find_video_queue_families() {
	if (!hardware_info.video_queue_supported) {
		return false;
	}
	
	// Get video queue from the video queue family
	vkGetDeviceQueue(device, hardware_info.video_queue_family, 0, &video_queue);
	return video_queue != VK_NULL_HANDLE;
}

bool VulkanVideoContext::_create_video_command_pool() {
	ERR_FAIL_COND_V(video_queue == VK_NULL_HANDLE, false);
	
	VkCommandPoolCreateInfo pool_info = {};
	pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	pool_info.queueFamilyIndex = hardware_info.video_queue_family;
	
	VkResult result = vkCreateCommandPool(device, &pool_info, nullptr, &video_command_pool);
	return result == VK_SUCCESS;
}

bool VulkanVideoContext::is_codec_supported(VideoCodecProfile p_profile, VideoOperationType p_operation) const {
	if (!initialized || !hardware_info.video_queue_supported) {
		return false;
	}
	
	VkVideoCodecOperationFlagBitsKHR vk_operation = get_vulkan_codec_operation(p_profile, p_operation);
	
	if (p_operation == VIDEO_OPERATION_DECODE) {
		return hardware_info.supported_decode_codecs.has(vk_operation);
	} else if (p_operation == VIDEO_OPERATION_ENCODE) {
		return hardware_info.supported_encode_codecs.has(vk_operation);
	}
	
	return false;
}

Dictionary VulkanVideoContext::get_video_capabilities(VideoCodecProfile p_profile, VideoOperationType p_operation) const {
	Dictionary caps;
	
	if (!is_codec_supported(p_profile, p_operation)) {
		return caps;
	}
	
	// Fill capabilities from hardware info
	caps["decode_supported"] = (p_operation == VIDEO_OPERATION_DECODE);
	caps["encode_supported"] = (p_operation == VIDEO_OPERATION_ENCODE);
	caps["max_width"] = hardware_info.max_coded_extent_width;
	caps["max_height"] = hardware_info.max_coded_extent_height;
	caps["max_dpb_slots"] = hardware_info.max_dpb_slots;
	caps["max_active_references"] = hardware_info.max_active_reference_pictures;
	
	Array supported_profiles;
	supported_profiles.push_back(p_profile);
	caps["supported_profiles"] = supported_profiles;
	
	Array supported_formats;
	supported_formats.push_back(RD::DATA_FORMAT_G8_B8R8_2PLANE_420_UNORM);
	caps["supported_formats"] = supported_formats;
	
	return caps;
}

Array VulkanVideoContext::get_supported_profiles() const {
	Array profiles;
	
	if (!initialized) {
		return profiles;
	}
	
	// Check each profile
	if (is_codec_supported(VIDEO_CODEC_PROFILE_AV1_MAIN, VIDEO_OPERATION_DECODE)) {
		profiles.push_back(VIDEO_CODEC_PROFILE_AV1_MAIN);
	}
	if (is_codec_supported(VIDEO_CODEC_PROFILE_AV1_HIGH, VIDEO_OPERATION_DECODE)) {
		profiles.push_back(VIDEO_CODEC_PROFILE_AV1_HIGH);
	}
	if (is_codec_supported(VIDEO_CODEC_PROFILE_AV1_PROFESSIONAL, VIDEO_OPERATION_DECODE)) {
		profiles.push_back(VIDEO_CODEC_PROFILE_AV1_PROFESSIONAL);
	}
	
	return profiles;
}

VkVideoCodecOperationFlagBitsKHR VulkanVideoContext::get_vulkan_codec_operation(VideoCodecProfile p_profile, VideoOperationType p_operation) const {
	if (p_operation == VIDEO_OPERATION_DECODE) {
		switch (p_profile) {
			case VIDEO_CODEC_PROFILE_AV1_MAIN:
			case VIDEO_CODEC_PROFILE_AV1_HIGH:
			case VIDEO_CODEC_PROFILE_AV1_PROFESSIONAL:
				return VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR;
			case VIDEO_CODEC_PROFILE_H264_BASELINE:
			case VIDEO_CODEC_PROFILE_H264_MAIN:
			case VIDEO_CODEC_PROFILE_H264_HIGH:
				return VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;
			case VIDEO_CODEC_PROFILE_H265_MAIN:
			case VIDEO_CODEC_PROFILE_H265_MAIN_10:
				return VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR;
		}
	} else if (p_operation == VIDEO_OPERATION_ENCODE) {
		switch (p_profile) {
			case VIDEO_CODEC_PROFILE_AV1_MAIN:
			case VIDEO_CODEC_PROFILE_AV1_HIGH:
			case VIDEO_CODEC_PROFILE_AV1_PROFESSIONAL:
				// Note: AV1 encode not yet standardized in Vulkan Video
				return VK_VIDEO_CODEC_OPERATION_NONE_KHR;
			case VIDEO_CODEC_PROFILE_H264_BASELINE:
			case VIDEO_CODEC_PROFILE_H264_MAIN:
			case VIDEO_CODEC_PROFILE_H264_HIGH:
				return VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR;
			case VIDEO_CODEC_PROFILE_H265_MAIN:
			case VIDEO_CODEC_PROFILE_H265_MAIN_10:
				return VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR;
		}
	}
	
	return VK_VIDEO_CODEC_OPERATION_NONE_KHR;
}

VkFormat VulkanVideoContext::get_vulkan_format(RD::DataFormat p_format) const {
	switch (p_format) {
		case RD::DATA_FORMAT_G8_B8R8_2PLANE_420_UNORM:
			return VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
		case RD::DATA_FORMAT_G8_B8_R8_3PLANE_420_UNORM:
			return VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM;
		case RD::DATA_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16:
			return VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16;
		default:
			return VK_FORMAT_UNDEFINED;
	}
}

RD::DataFormat VulkanVideoContext::get_rd_format(VkFormat p_format) const {
	switch (p_format) {
		case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM:
			return RD::DATA_FORMAT_G8_B8R8_2PLANE_420_UNORM;
		case VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM:
			return RD::DATA_FORMAT_G8_B8_R8_3PLANE_420_UNORM;
		case VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16:
			return RD::DATA_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16;
		default:
			return RD::DATA_FORMAT_MAX;
	}
}

RID VulkanVideoContext::_generate_video_rid() {
	return RID::from_uint64(Math::rand());
}

// Placeholder implementations for resource management
// These would need full Vulkan Video implementation

RID VulkanVideoContext::create_video_session(const Dictionary &p_create_info) {
	ERR_FAIL_COND_V(!initialized, RID());
	WARN_PRINT("Video session creation not yet fully implemented");
	return RID();
}

void VulkanVideoContext::destroy_video_session(RID p_session_rid) {
	ERR_FAIL_COND(!initialized);
	WARN_PRINT("Video session destruction not yet fully implemented");
}

bool VulkanVideoContext::is_video_session_valid(RID p_session_rid) const {
	return video_sessions.has(p_session_rid);
}

RID VulkanVideoContext::create_video_session_parameters(const Dictionary &p_create_info) {
	ERR_FAIL_COND_V(!initialized, RID());
	WARN_PRINT("Video session parameters creation not yet fully implemented");
	return RID();
}

void VulkanVideoContext::destroy_video_session_parameters(RID p_parameters_rid) {
	ERR_FAIL_COND(!initialized);
	WARN_PRINT("Video session parameters destruction not yet fully implemented");
}

RID VulkanVideoContext::create_video_image(const Dictionary &p_create_info) {
	ERR_FAIL_COND_V(!initialized, RID());
	WARN_PRINT("Video image creation not yet fully implemented");
	return RID();
}

void VulkanVideoContext::destroy_video_image(RID p_image_rid) {
	ERR_FAIL_COND(!initialized);
	WARN_PRINT("Video image destruction not yet fully implemented");
}

RID VulkanVideoContext::create_video_buffer(const Dictionary &p_create_info) {
	ERR_FAIL_COND_V(!initialized, RID());
	WARN_PRINT("Video buffer creation not yet fully implemented");
	return RID();
}

void VulkanVideoContext::destroy_video_buffer(RID p_buffer_rid) {
	ERR_FAIL_COND(!initialized);
	WARN_PRINT("Video buffer destruction not yet fully implemented");
}

bool VulkanVideoContext::begin_video_coding(VkCommandBuffer p_cmd_buffer, RID p_session_rid) {
	ERR_FAIL_COND_V(!initialized, false);
	WARN_PRINT("Video coding begin not yet fully implemented");
	return false;
}

bool VulkanVideoContext::end_video_coding(VkCommandBuffer p_cmd_buffer) {
	ERR_FAIL_COND_V(!initialized, false);
	WARN_PRINT("Video coding end not yet fully implemented");
	return false;
}

bool VulkanVideoContext::decode_video_frame(VkCommandBuffer p_cmd_buffer, const Dictionary &p_decode_info) {
	ERR_FAIL_COND_V(!initialized, false);
	WARN_PRINT("Video frame decoding not yet fully implemented");
	return false;
}

bool VulkanVideoContext::update_video_buffer(RID p_buffer_rid, uint64_t p_offset, const Vector<uint8_t> &p_data) {
	ERR_FAIL_COND_V(!initialized, false);
	WARN_PRINT("Video buffer update not yet fully implemented");
	return false;
}

Vector<uint8_t> VulkanVideoContext::get_video_buffer_data(RID p_buffer_rid, uint64_t p_offset, uint64_t p_size) {
	ERR_FAIL_COND_V(!initialized, Vector<uint8_t>());
	WARN_PRINT("Video buffer data retrieval not yet fully implemented");
	return Vector<uint8_t>();
}

// Cleanup helpers
void VulkanVideoContext::_cleanup_video_sessions() {
	for (const KeyValue<RID, VulkanVideoSession> &pair : video_sessions) {
		const VulkanVideoSession &session = pair.value;
		if (session.session != VK_NULL_HANDLE && video_functions.vkDestroyVideoSessionKHR) {
			video_functions.vkDestroyVideoSessionKHR(device, session.session, nullptr);
		}
		if (session.parameters != VK_NULL_HANDLE && video_functions.vkDestroyVideoSessionParametersKHR) {
			video_functions.vkDestroyVideoSessionParametersKHR(device, session.parameters, nullptr);
		}
		if (session.memory != VK_NULL_HANDLE) {
			vkFreeMemory(device, session.memory, nullptr);
		}
	}
	video_sessions.clear();
}

void VulkanVideoContext::_cleanup_video_images() {
	for (const KeyValue<RID, VulkanVideoImage> &pair : video_images) {
		const VulkanVideoImage &image = pair.value;
		if (image.image_view != VK_NULL_HANDLE) {
			vkDestroyImageView(device, image.image_view, nullptr);
		}
		if (image.image != VK_NULL_HANDLE) {
			vkDestroyImage(device, image.image, nullptr);
		}
		if (image.memory != VK_NULL_HANDLE) {
			vkFreeMemory(device, image.memory, nullptr);
		}
	}
	video_images.clear();
}

void VulkanVideoContext::_cleanup_video_buffers() {
	for (const KeyValue<RID, VulkanVideoBuffer> &pair : video_buffers) {
		const VulkanVideoBuffer &buffer = pair.value;
		if (buffer.mapped_data != nullptr) {
			vkUnmapMemory(device, buffer.memory);
		}
		if (buffer.buffer != VK_NULL_HANDLE) {
			vkDestroyBuffer(device, buffer.buffer, nullptr);
		}
		if (buffer.memory != VK_NULL_HANDLE) {
			vkFreeMemory(device, buffer.memory, nullptr);
		}
	}
	video_buffers.clear();
}

bool VulkanVideoContext::_allocate_video_session_memory(VulkanVideoSession &p_session) {
	// TODO: Implement video session memory allocation
	return false;
}

bool VulkanVideoContext::_allocate_video_image_memory(VulkanVideoImage &p_image) {
	// TODO: Implement video image memory allocation
	return false;
}

bool VulkanVideoContext::_allocate_video_buffer_memory(VulkanVideoBuffer &p_buffer) {
	// TODO: Implement video buffer memory allocation
	return false;
}

#endif // VULKAN_ENABLED
