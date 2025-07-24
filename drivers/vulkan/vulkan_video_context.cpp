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

#include "drivers/vulkan/rendering_device_driver_vulkan.h"
#include "core/string/print_string.h"

VulkanVideoContext::VulkanVideoContext() {
}

VulkanVideoContext::~VulkanVideoContext() {
	cleanup();
}

bool VulkanVideoContext::initialize(RenderingDeviceDriverVulkan *p_driver) {
	ERR_FAIL_NULL_V(p_driver, false);
	
	driver = p_driver;
	device = p_driver->get_vk_device();
	physical_device = p_driver->get_vk_physical_device();
	
	ERR_FAIL_COND_V(device == VK_NULL_HANDLE, false);
	ERR_FAIL_COND_V(physical_device == VK_NULL_HANDLE, false);
	
	// Use the already-detected video queue family and queue from the driver
	if (p_driver->get_video_decode_queue_family() != UINT32_MAX) {
		hardware_info.video_queue_family = p_driver->get_video_decode_queue_family();
		hardware_info.decode_queue_family = p_driver->get_video_decode_queue_family();
		hardware_info.video_queue_supported = true;
		hardware_info.decode_queue_supported = true;
		video_queue = p_driver->get_video_decode_queue();
	}
	
	// Detect video hardware capabilities
	if (!detect_video_hardware()) {
		WARN_PRINT("Failed to detect video hardware capabilities");
		return false;
	}
	
	// Create video command pool
	if (!_create_video_command_pool()) {
		WARN_PRINT("Failed to create video command pool");
		return false;
	}
	
	initialized = true;
	print_verbose("VulkanVideoContext: Initialized successfully");
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
	driver = nullptr;
	initialized = false;
	
	print_verbose("VulkanVideoContext: Cleaned up");
}

bool VulkanVideoContext::detect_video_hardware() {
	ERR_FAIL_COND_V(!driver, false);
	
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
			hardware_info.video_queue_family = i;
			hardware_info.video_queue_supported = true;
		}
		
		if (props.queueFlags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR) {
			hardware_info.encode_queue_supported = true;
			hardware_info.encode_queue_family = i;
		}
	}
	
	// Query video capabilities for supported codecs
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
	
	if (driver->get_device_functions().GetPhysicalDeviceVideoCapabilitiesKHR) {
		VkResult result = driver->get_device_functions().GetPhysicalDeviceVideoCapabilitiesKHR(
			physical_device, &profile_info, &capabilities);
		
		if (result == VK_SUCCESS) {
			hardware_info.max_coded_extent_width = capabilities.maxCodedExtent.width;
			hardware_info.max_coded_extent_height = capabilities.maxCodedExtent.height;
			hardware_info.max_dpb_slots = capabilities.maxDpbSlots;
			hardware_info.max_active_reference_pictures = capabilities.maxActiveReferencePictures;
			
			// Add AV1 decode to supported codecs
			hardware_info.supported_decode_codecs.push_back(VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR);
			
			print_verbose("VulkanVideoContext: AV1 decode capabilities detected");
			return true;
		}
	}
	
	return false;
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

bool VulkanVideoContext::is_codec_supported(VkVideoCodecOperationFlagBitsKHR p_codec_operation) const {
	if (!initialized || !hardware_info.video_queue_supported) {
		return false;
	}
	
	return hardware_info.supported_decode_codecs.has(p_codec_operation);
}

bool VulkanVideoContext::get_video_capabilities(VkVideoCodecOperationFlagBitsKHR p_codec_operation, VkVideoCapabilitiesKHR &r_capabilities) const {
	if (!is_codec_supported(p_codec_operation)) {
		return false;
	}
	
	// Create profile info for the codec
	VkVideoProfileInfoKHR profile_info = {};
	profile_info.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR;
	profile_info.videoCodecOperation = p_codec_operation;
	
	VkVideoDecodeAV1ProfileInfoKHR av1_profile = {};
	if (p_codec_operation == VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR) {
		av1_profile.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PROFILE_INFO_KHR;
		av1_profile.stdProfile = STD_VIDEO_AV1_PROFILE_MAIN;
		profile_info.pNext = &av1_profile;
	}
	
	r_capabilities.sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR;
	
	if (driver->get_device_functions().GetPhysicalDeviceVideoCapabilitiesKHR) {
		VkResult result = driver->get_device_functions().GetPhysicalDeviceVideoCapabilitiesKHR(
			physical_device, &profile_info, &r_capabilities);
		return result == VK_SUCCESS;
	}
	
	return false;
}

Vector<VkVideoCodecOperationFlagBitsKHR> VulkanVideoContext::get_supported_codecs() const {
	return hardware_info.supported_decode_codecs;
}

uint32_t VulkanVideoContext::create_video_session(VkVideoCodecOperationFlagBitsKHR p_codec_operation, uint32_t p_width, uint32_t p_height, uint32_t p_dpb_slots) {
	ERR_FAIL_COND_V(!initialized, 0);
	ERR_FAIL_COND_V(!is_codec_supported(p_codec_operation), 0);
	
	// TODO: Implement full video session creation
	WARN_PRINT("VulkanVideoContext: Video session creation not yet fully implemented");
	
	uint32_t session_id = next_session_id++;
	VulkanVideoSession session;
	session.codec_operation = p_codec_operation;
	session.max_width = p_width;
	session.max_height = p_height;
	session.max_dpb_slots = p_dpb_slots;
	session.is_valid = false; // Mark as invalid until fully implemented
	
	video_sessions[session_id] = session;
	return session_id;
}

void VulkanVideoContext::destroy_video_session(uint32_t p_session_id) {
	ERR_FAIL_COND(!initialized);
	
	if (video_sessions.has(p_session_id)) {
		VulkanVideoSession &session = video_sessions[p_session_id];
		
		if (session.session != VK_NULL_HANDLE && driver->get_device_functions().DestroyVideoSessionKHR) {
			driver->get_device_functions().DestroyVideoSessionKHR(device, session.session, nullptr);
		}
		if (session.parameters != VK_NULL_HANDLE && driver->get_device_functions().DestroyVideoSessionParametersKHR) {
			driver->get_device_functions().DestroyVideoSessionParametersKHR(device, session.parameters, nullptr);
		}
		if (session.memory != VK_NULL_HANDLE) {
			vkFreeMemory(device, session.memory, nullptr);
		}
		
		video_sessions.erase(p_session_id);
	}
}

bool VulkanVideoContext::is_video_session_valid(uint32_t p_session_id) const {
	if (!video_sessions.has(p_session_id)) {
		return false;
	}
	return video_sessions[p_session_id].is_valid;
}

VkFormat VulkanVideoContext::get_vulkan_format_from_rd(RD::DataFormat p_format) const {
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

RD::DataFormat VulkanVideoContext::get_rd_format_from_vulkan(VkFormat p_format) const {
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

// Placeholder implementations for remaining methods
uint32_t VulkanVideoContext::create_video_session_parameters(uint32_t p_session_id) {
	WARN_PRINT("VulkanVideoContext: Video session parameters creation not yet implemented");
	return 0;
}

void VulkanVideoContext::destroy_video_session_parameters(uint32_t p_session_id) {
	WARN_PRINT("VulkanVideoContext: Video session parameters destruction not yet implemented");
}

uint32_t VulkanVideoContext::create_video_image(uint32_t p_session_id, VkFormat p_format, VkImageUsageFlags p_usage) {
	WARN_PRINT("VulkanVideoContext: Video image creation not yet implemented");
	return 0;
}

void VulkanVideoContext::destroy_video_image(uint32_t p_image_id) {
	WARN_PRINT("VulkanVideoContext: Video image destruction not yet implemented");
}

uint32_t VulkanVideoContext::create_video_buffer(VkDeviceSize p_size, VkBufferUsageFlags p_usage) {
	WARN_PRINT("VulkanVideoContext: Video buffer creation not yet implemented");
	return 0;
}

void VulkanVideoContext::destroy_video_buffer(uint32_t p_buffer_id) {
	WARN_PRINT("VulkanVideoContext: Video buffer destruction not yet implemented");
}

bool VulkanVideoContext::begin_video_coding(VkCommandBuffer p_cmd_buffer, uint32_t p_session_id) {
	WARN_PRINT("VulkanVideoContext: Video coding begin not yet implemented");
	return false;
}

bool VulkanVideoContext::end_video_coding(VkCommandBuffer p_cmd_buffer) {
	WARN_PRINT("VulkanVideoContext: Video coding end not yet implemented");
	return false;
}

bool VulkanVideoContext::decode_video_frame(VkCommandBuffer p_cmd_buffer, uint32_t p_session_id, uint32_t p_bitstream_buffer_id, uint32_t p_output_image_id) {
	WARN_PRINT("VulkanVideoContext: Video frame decoding not yet implemented");
	return false;
}

bool VulkanVideoContext::update_video_buffer(uint32_t p_buffer_id, uint64_t p_offset, const Vector<uint8_t> &p_data) {
	WARN_PRINT("VulkanVideoContext: Video buffer update not yet implemented");
	return false;
}

Vector<uint8_t> VulkanVideoContext::get_video_buffer_data(uint32_t p_buffer_id, uint64_t p_offset, uint64_t p_size) {
	WARN_PRINT("VulkanVideoContext: Video buffer data retrieval not yet implemented");
	return Vector<uint8_t>();
}

// Cleanup helpers
void VulkanVideoContext::_cleanup_video_sessions() {
	for (const KeyValue<uint32_t, VulkanVideoSession> &pair : video_sessions) {
		const VulkanVideoSession &session = pair.value;
		if (session.session != VK_NULL_HANDLE && driver && driver->get_device_functions().DestroyVideoSessionKHR) {
			driver->get_device_functions().DestroyVideoSessionKHR(device, session.session, nullptr);
		}
		if (session.parameters != VK_NULL_HANDLE && driver && driver->get_device_functions().DestroyVideoSessionParametersKHR) {
			driver->get_device_functions().DestroyVideoSessionParametersKHR(device, session.parameters, nullptr);
		}
		if (session.memory != VK_NULL_HANDLE) {
			vkFreeMemory(device, session.memory, nullptr);
		}
	}
	video_sessions.clear();
}

void VulkanVideoContext::_cleanup_video_images() {
	for (const KeyValue<uint32_t, VulkanVideoImage> &pair : video_images) {
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
	for (const KeyValue<uint32_t, VulkanVideoBuffer> &pair : video_buffers) {
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
