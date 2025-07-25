/**************************************************************************/
/*  rendering_device_video_extensions.h                                   */
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
#include "core/io/image.h"
#include "core/variant/variant.h"

#ifdef VULKAN_ENABLED
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

// Forward declarations
class RenderingDeviceDriverVulkan;
class RenderingDevice;
class VulkanDeviceContext;
class VulkanVideoProcessor;

// Template forward declaration for smart pointer
template<class T>
class VkSharedBaseObj;
#endif

// Constants for video codec profiles and operations
enum {
	VIDEO_CODEC_PROFILE_AV1_MAIN = 0,
	VIDEO_OPERATION_DECODE = 0,
	VIDEO_OPERATION_ENCODE = 1
};

// Extension class for RenderingDevice to add video decoding functionality
class RenderingDeviceVideoExtensions : public RefCounted {
	GDCLASS(RenderingDeviceVideoExtensions, RefCounted);

private:
#ifdef VULKAN_ENABLED
	RenderingDeviceDriverVulkan *vulkan_driver = nullptr;
	VkSharedBaseObj<VulkanDeviceContext> *device_context = nullptr;
	bool initialized = false;
#endif

protected:
	static void _bind_methods();

public:
	RenderingDeviceVideoExtensions();
	virtual ~RenderingDeviceVideoExtensions();

	// Initialize the video extensions with a RenderingDevice
	Error initialize(RenderingDevice *p_rendering_device);

	// Check if video decoding is supported
	bool is_video_supported() const;

	// Get video capabilities for a specific codec and operation
	Dictionary get_video_capabilities(int p_codec_profile, int p_operation) const;

	// Create a VulkanDeviceContext that bridges with Godot's Vulkan driver
	VkSharedBaseObj<VulkanDeviceContext> *get_device_context() const;

#ifdef VULKAN_ENABLED
	// Internal methods for video decoder integration
	VkDevice get_vk_device() const;
	VkPhysicalDevice get_vk_physical_device() const;
	VkInstance get_vk_instance() const;
	uint32_t get_video_decode_queue_family() const;
	VkQueue get_video_decode_queue() const;
#endif

private:
	bool _initialize_device_context();
	void _cleanup_device_context();
};
