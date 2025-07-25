/**************************************************************************/
/*  rendering_device_video_extensions.cpp                                 */
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

#include "rendering_device_video_extensions.h"

#ifdef VULKAN_ENABLED
#define VK_NO_PROTOTYPES
#include "drivers/vulkan/rendering_device_driver_vulkan.h"
#include "servers/rendering_server.h"
#endif

RenderingDeviceVideoExtensions::RenderingDeviceVideoExtensions() {
#ifdef VULKAN_ENABLED
	vulkan_driver = nullptr;
	device_context = nullptr;
	initialized = false;
#endif
}

RenderingDeviceVideoExtensions::~RenderingDeviceVideoExtensions() {
#ifdef VULKAN_ENABLED
	_cleanup_device_context();
#endif
}

Error RenderingDeviceVideoExtensions::initialize(RenderingDevice *p_rendering_device) {
#ifdef VULKAN_ENABLED
	if (!p_rendering_device) {
		return ERR_INVALID_PARAMETER;
	}

	// Get the Vulkan driver from the rendering device
	RenderingDeviceDriver *driver = p_rendering_device->get_device_driver();
	vulkan_driver = static_cast<RenderingDeviceDriverVulkan *>(driver);
	if (!vulkan_driver) {
		ERR_PRINT("Failed to get Vulkan driver from RenderingDevice");
		return ERR_INVALID_PARAMETER;
	}

	// Initialize using Godot's built-in video decoder support
	if (!_initialize_device_context()) {
		ERR_PRINT("Failed to initialize video decoder support");
		return ERR_UNAVAILABLE;
	}

	initialized = true;
	print_line("RenderingDeviceVideoExtensions initialized successfully");
	return OK;
#else
	ERR_PRINT("Vulkan not enabled");
	return ERR_UNAVAILABLE;
#endif
}

bool RenderingDeviceVideoExtensions::is_video_supported() const {
#ifdef VULKAN_ENABLED
	if (!initialized || !vulkan_driver) {
		return false;
	}

	// Use Godot's built-in video decoder support check
	return vulkan_driver->video_decoder_is_supported();
#else
	return false;
#endif
}

Dictionary RenderingDeviceVideoExtensions::get_video_capabilities(int p_codec_profile, int p_operation) const {
	Dictionary caps;
	caps["hardware_decode_supported"] = false;
	caps["hardware_encode_supported"] = false;
	caps["max_width"] = 0;
	caps["max_height"] = 0;
	caps["max_dpb_slots"] = 0;
	caps["max_level"] = 0;
	caps["supported_profiles"] = Array();

#ifdef VULKAN_ENABLED
	if (!initialized || !vulkan_driver) {
		return caps;
	}

	// Check if video decoding is supported using Godot's API
	if (vulkan_driver->video_decoder_is_supported()) {
		caps["hardware_decode_supported"] = true;
		
		// For AV1, set reasonable defaults based on common hardware capabilities
		if (p_codec_profile == VIDEO_CODEC_PROFILE_AV1_MAIN) {
			caps["max_width"] = 7680;  // 8K width
			caps["max_height"] = 4320; // 8K height
			caps["max_dpb_slots"] = 8; // AV1 standard
			caps["max_level"] = 31;    // Level 3.1

			Array profiles;
			profiles.push_back(0); // Main profile
			caps["supported_profiles"] = profiles;
		}
	}

	// Encode support would need additional API
	caps["hardware_encode_supported"] = false;
#endif

	return caps;
}

VkSharedBaseObj<VulkanDeviceContext> *RenderingDeviceVideoExtensions::get_device_context() const {
#ifdef VULKAN_ENABLED
	return device_context;
#else
	return nullptr;
#endif
}

#ifdef VULKAN_ENABLED
VkDevice RenderingDeviceVideoExtensions::get_vk_device() const {
	if (!vulkan_driver) {
		return VK_NULL_HANDLE;
	}
	return vulkan_driver->get_vk_device();
}

VkPhysicalDevice RenderingDeviceVideoExtensions::get_vk_physical_device() const {
	if (!vulkan_driver) {
		return VK_NULL_HANDLE;
	}
	return vulkan_driver->get_vk_physical_device();
}

VkInstance RenderingDeviceVideoExtensions::get_vk_instance() const {
	// For now, return null handle as we don't have direct access
	// This would need to be exposed through the Vulkan driver API
	return VK_NULL_HANDLE;
}

uint32_t RenderingDeviceVideoExtensions::get_video_decode_queue_family() const {
	if (!vulkan_driver) {
		return UINT32_MAX;
	}
	return vulkan_driver->get_video_decode_queue_family();
}

VkQueue RenderingDeviceVideoExtensions::get_video_decode_queue() const {
	if (!vulkan_driver) {
		return VK_NULL_HANDLE;
	}
	return vulkan_driver->get_video_decode_queue();
}

bool RenderingDeviceVideoExtensions::_initialize_device_context() {
	if (!vulkan_driver) {
		return false;
	}

	// For now, just check if video decoding is supported
	// The actual VkCodecUtils integration would be done later
	// when we have proper access to Vulkan objects
	
	if (!vulkan_driver->video_decoder_is_supported()) {
		WARN_PRINT("Video decoding not supported on this device");
		return false;
	}

	print_line("Video decoder support detected");
	return true;
}

void RenderingDeviceVideoExtensions::_cleanup_device_context() {
	if (device_context) {
		// Cleanup would go here when VkCodecUtils is properly integrated
		device_context = nullptr;
	}
	vulkan_driver = nullptr;
	initialized = false;
}
#endif

void RenderingDeviceVideoExtensions::_bind_methods() {
	ClassDB::bind_method(D_METHOD("initialize", "rendering_device"), &RenderingDeviceVideoExtensions::initialize);
	ClassDB::bind_method(D_METHOD("is_video_supported"), &RenderingDeviceVideoExtensions::is_video_supported);
	ClassDB::bind_method(D_METHOD("get_video_capabilities", "codec_profile", "operation"), &RenderingDeviceVideoExtensions::get_video_capabilities);

	// Constants for codec profiles and operations
	BIND_CONSTANT(VIDEO_CODEC_PROFILE_AV1_MAIN);
	BIND_CONSTANT(VIDEO_OPERATION_DECODE);
	BIND_CONSTANT(VIDEO_OPERATION_ENCODE);
}
