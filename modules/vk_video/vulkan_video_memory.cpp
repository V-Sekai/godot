/**************************************************************************/
/*  vulkan_video_memory.cpp                                              */
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

#include "vulkan_video_memory.h"

#ifdef VULKAN_ENABLED

#include "core/string/print_string.h"

VulkanVideoMemory::VulkanVideoMemory() {
}

VulkanVideoMemory::~VulkanVideoMemory() {
	finalize();
}

Error VulkanVideoMemory::initialize(VkDevice p_device, VkPhysicalDevice p_physical_device, uint32_t p_video_queue_family) {
	ERR_FAIL_COND_V(p_device == VK_NULL_HANDLE, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_physical_device == VK_NULL_HANDLE, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_video_queue_family == UINT32_MAX, ERR_INVALID_PARAMETER);

	vk_device = p_device;
	physical_device = p_physical_device;
	video_decode_queue_family = p_video_queue_family;

	print_verbose("VulkanVideoMemory: Initialized video memory manager");
	return OK;
}

void VulkanVideoMemory::finalize() {
	destroy_output_images();
	destroy_dpb_images();

	// Clean up any remaining video buffers
	for (uint32_t i = 0; i < video_buffers.size(); i++) {
		destroy_video_buffer(&video_buffers[i]);
	}
	video_buffers.clear();

	vk_device = VK_NULL_HANDLE;
	physical_device = VK_NULL_HANDLE;
	video_decode_queue_family = UINT32_MAX;
}

Error VulkanVideoMemory::create_dpb_images(uint32_t p_max_slots, uint32_t p_width, uint32_t p_height, VkFormat p_format, const VkVideoProfileInfoKHR *p_video_profile) {
	ERR_FAIL_COND_V(!is_initialized(), ERR_UNCONFIGURED);
	ERR_FAIL_COND_V(p_max_slots == 0, ERR_INVALID_PARAMETER);
	ERR_FAIL_NULL_V(p_video_profile, ERR_INVALID_PARAMETER);

	// Clean up existing DPB images
	destroy_dpb_images();

	max_dpb_slots = p_max_slots;
	dpb_slots.resize(max_dpb_slots);

	// Create DPB images for reference frame storage
	for (uint32_t i = 0; i < max_dpb_slots; i++) {
		DPBSlot &slot = dpb_slots[i];
		slot.slot_index = i;
		slot.is_active = false;
		slot.is_reference = false;
		slot.frame_number = 0;

		VideoImageInfo &image_info = slot.image_info;
		image_info.width = p_width;
		image_info.height = p_height;
		image_info.format = p_format;
		image_info.usage = VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR | VK_IMAGE_USAGE_SAMPLED_BIT;
		image_info.is_video_dpb = true;
		image_info.array_layers = 1;

		// Create image with video profile
		VkVideoProfileListInfoKHR profile_list = {};
		profile_list.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR;
		profile_list.profileCount = 1;
		profile_list.pProfiles = p_video_profile;

		VkImageCreateInfo image_create_info = {};
		image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		image_create_info.pNext = &profile_list;
		image_create_info.imageType = VK_IMAGE_TYPE_2D;
		image_create_info.format = p_format;
		image_create_info.extent.width = p_width;
		image_create_info.extent.height = p_height;
		image_create_info.extent.depth = 1;
		image_create_info.mipLevels = 1;
		image_create_info.arrayLayers = 1;
		image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
		image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
		image_create_info.usage = image_info.usage;
		image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VkResult result = vkCreateImage(vk_device, &image_create_info, nullptr, &image_info.vk_image);
		ERR_FAIL_COND_V_MSG(result != VK_SUCCESS, ERR_CANT_CREATE,
				"Failed to create DPB image " + itos(i) + " (VkResult: " + itos(result) + ")");

		// Allocate memory for the image
		Error memory_result = _allocate_image_memory(&image_info);
		if (memory_result != OK) {
			vkDestroyImage(vk_device, image_info.vk_image, nullptr);
			return memory_result;
		}

		// Create image view
		VkImageViewCreateInfo view_create_info = {};
		view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		view_create_info.image = image_info.vk_image;
		view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
		view_create_info.format = p_format;
		view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		view_create_info.subresourceRange.levelCount = 1;
		view_create_info.subresourceRange.layerCount = 1;

		result = vkCreateImageView(vk_device, &view_create_info, nullptr, &image_info.vk_image_view);
		ERR_FAIL_COND_V_MSG(result != VK_SUCCESS, ERR_CANT_CREATE,
				"Failed to create DPB image view " + itos(i) + " (VkResult: " + itos(result) + ")");

		image_info.is_allocated = true;
	}

	active_dpb_slots = 0;
	print_verbose("VulkanVideoMemory: Created " + itos(max_dpb_slots) + " DPB images (" + itos(p_width) + "x" + itos(p_height) + ")");
	return OK;
}

void VulkanVideoMemory::destroy_dpb_images() {
	for (uint32_t i = 0; i < dpb_slots.size(); i++) {
		VideoImageInfo &image_info = dpb_slots[i].image_info;
		
		if (image_info.vk_image_view != VK_NULL_HANDLE) {
			vkDestroyImageView(vk_device, image_info.vk_image_view, nullptr);
			image_info.vk_image_view = VK_NULL_HANDLE;
		}
		
		if (image_info.vk_image != VK_NULL_HANDLE) {
			vkDestroyImage(vk_device, image_info.vk_image, nullptr);
			image_info.vk_image = VK_NULL_HANDLE;
		}
		
		if (image_info.vk_memory != VK_NULL_HANDLE) {
			vkFreeMemory(vk_device, image_info.vk_memory, nullptr);
			image_info.vk_memory = VK_NULL_HANDLE;
		}
		
		image_info.is_allocated = false;
	}
	
	dpb_slots.clear();
	active_dpb_slots = 0;
	print_verbose("VulkanVideoMemory: Destroyed DPB images");
}

VulkanVideoMemory::DPBSlot *VulkanVideoMemory::acquire_dpb_slot() {
	// Find an inactive slot
	for (uint32_t i = 0; i < dpb_slots.size(); i++) {
		if (!dpb_slots[i].is_active) {
			dpb_slots[i].is_active = true;
			dpb_slots[i].is_reference = false;
			active_dpb_slots++;
			return &dpb_slots[i];
		}
	}
	
	ERR_PRINT("VulkanVideoMemory: No available DPB slots");
	return nullptr;
}

void VulkanVideoMemory::release_dpb_slot(uint32_t p_slot_index) {
	ERR_FAIL_COND(p_slot_index >= dpb_slots.size());
	
	if (dpb_slots[p_slot_index].is_active) {
		dpb_slots[p_slot_index].is_active = false;
		dpb_slots[p_slot_index].is_reference = false;
		active_dpb_slots--;
	}
}

VulkanVideoMemory::DPBSlot *VulkanVideoMemory::get_dpb_slot(uint32_t p_slot_index) {
	ERR_FAIL_COND_V(p_slot_index >= dpb_slots.size(), nullptr);
	return &dpb_slots[p_slot_index];
}

Error VulkanVideoMemory::create_output_images(uint32_t p_count, uint32_t p_width, uint32_t p_height, VkFormat p_format, const VkVideoProfileInfoKHR *p_video_profile) {
	ERR_FAIL_COND_V(!is_initialized(), ERR_UNCONFIGURED);
	ERR_FAIL_COND_V(p_count == 0, ERR_INVALID_PARAMETER);
	ERR_FAIL_NULL_V(p_video_profile, ERR_INVALID_PARAMETER);

	// Clean up existing output images
	destroy_output_images();

	output_images.resize(p_count);

	// Create output images for decoded frames
	for (uint32_t i = 0; i < p_count; i++) {
		VideoImageInfo &image_info = output_images[i];
		image_info.width = p_width;
		image_info.height = p_height;
		image_info.format = p_format;
		image_info.usage = VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		image_info.is_video_decode_output = true;
		image_info.array_layers = 1;

		// Create image with video profile
		VkVideoProfileListInfoKHR profile_list = {};
		profile_list.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR;
		profile_list.profileCount = 1;
		profile_list.pProfiles = p_video_profile;

		VkImageCreateInfo image_create_info = {};
		image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		image_create_info.pNext = &profile_list;
		image_create_info.imageType = VK_IMAGE_TYPE_2D;
		image_create_info.format = p_format;
		image_create_info.extent.width = p_width;
		image_create_info.extent.height = p_height;
		image_create_info.extent.depth = 1;
		image_create_info.mipLevels = 1;
		image_create_info.arrayLayers = 1;
		image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
		image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
		image_create_info.usage = image_info.usage;
		image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VkResult result = vkCreateImage(vk_device, &image_create_info, nullptr, &image_info.vk_image);
		ERR_FAIL_COND_V_MSG(result != VK_SUCCESS, ERR_CANT_CREATE,
				"Failed to create output image " + itos(i) + " (VkResult: " + itos(result) + ")");

		// Allocate memory for the image
		Error memory_result = _allocate_image_memory(&image_info);
		if (memory_result != OK) {
			vkDestroyImage(vk_device, image_info.vk_image, nullptr);
			return memory_result;
		}

		// Create image view
		VkImageViewCreateInfo view_create_info = {};
		view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		view_create_info.image = image_info.vk_image;
		view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
		view_create_info.format = p_format;
		view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		view_create_info.subresourceRange.levelCount = 1;
		view_create_info.subresourceRange.layerCount = 1;

		result = vkCreateImageView(vk_device, &view_create_info, nullptr, &image_info.vk_image_view);
		ERR_FAIL_COND_V_MSG(result != VK_SUCCESS, ERR_CANT_CREATE,
				"Failed to create output image view " + itos(i) + " (VkResult: " + itos(result) + ")");

		image_info.is_allocated = true;
	}

	current_output_index = 0;
	print_verbose("VulkanVideoMemory: Created " + itos(p_count) + " output images (" + itos(p_width) + "x" + itos(p_height) + ")");
	return OK;
}

void VulkanVideoMemory::destroy_output_images() {
	for (uint32_t i = 0; i < output_images.size(); i++) {
		VideoImageInfo &image_info = output_images[i];
		
		if (image_info.vk_image_view != VK_NULL_HANDLE) {
			vkDestroyImageView(vk_device, image_info.vk_image_view, nullptr);
			image_info.vk_image_view = VK_NULL_HANDLE;
		}
		
		if (image_info.vk_image != VK_NULL_HANDLE) {
			vkDestroyImage(vk_device, image_info.vk_image, nullptr);
			image_info.vk_image = VK_NULL_HANDLE;
		}
		
		if (image_info.vk_memory != VK_NULL_HANDLE) {
			vkFreeMemory(vk_device, image_info.vk_memory, nullptr);
			image_info.vk_memory = VK_NULL_HANDLE;
		}
		
		image_info.is_allocated = false;
	}
	
	output_images.clear();
	current_output_index = 0;
	print_verbose("VulkanVideoMemory: Destroyed output images");
}

VulkanVideoMemory::VideoImageInfo *VulkanVideoMemory::get_current_output_image() {
	ERR_FAIL_COND_V(output_images.is_empty(), nullptr);
	return &output_images[current_output_index];
}

VulkanVideoMemory::VideoImageInfo *VulkanVideoMemory::get_output_image(uint32_t p_index) {
	ERR_FAIL_COND_V(p_index >= output_images.size(), nullptr);
	return &output_images[p_index];
}

void VulkanVideoMemory::advance_output_image() {
	if (!output_images.is_empty()) {
		current_output_index = (current_output_index + 1) % output_images.size();
	}
}

Error VulkanVideoMemory::create_video_buffer(uint64_t p_size, VkBufferUsageFlags p_usage, VideoBufferInfo *p_buffer_info) {
	ERR_FAIL_COND_V(!is_initialized(), ERR_UNCONFIGURED);
	ERR_FAIL_NULL_V(p_buffer_info, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_size == 0, ERR_INVALID_PARAMETER);

	p_buffer_info->size = p_size;
	p_buffer_info->usage = p_usage;

	// Create buffer
	VkBufferCreateInfo buffer_create_info = {};
	buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_create_info.size = p_size;
	buffer_create_info.usage = p_usage;
	buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VkResult result = vkCreateBuffer(vk_device, &buffer_create_info, nullptr, &p_buffer_info->vk_buffer);
	ERR_FAIL_COND_V_MSG(result != VK_SUCCESS, ERR_CANT_CREATE,
			"Failed to create video buffer (VkResult: " + itos(result) + ")");

	// Allocate memory for the buffer
	Error memory_result = _allocate_buffer_memory(p_buffer_info);
	if (memory_result != OK) {
		vkDestroyBuffer(vk_device, p_buffer_info->vk_buffer, nullptr);
		return memory_result;
	}

	return OK;
}

void VulkanVideoMemory::destroy_video_buffer(VideoBufferInfo *p_buffer_info) {
	ERR_FAIL_NULL(p_buffer_info);

	if (p_buffer_info->is_mapped) {
		unmap_video_buffer(p_buffer_info);
	}

	if (p_buffer_info->vk_buffer != VK_NULL_HANDLE) {
		vkDestroyBuffer(vk_device, p_buffer_info->vk_buffer, nullptr);
		p_buffer_info->vk_buffer = VK_NULL_HANDLE;
	}

	if (p_buffer_info->vk_memory != VK_NULL_HANDLE) {
		vkFreeMemory(vk_device, p_buffer_info->vk_memory, nullptr);
		p_buffer_info->vk_memory = VK_NULL_HANDLE;
	}

	p_buffer_info->size = 0;
	p_buffer_info->mapped_ptr = nullptr;
	p_buffer_info->is_mapped = false;
}

Error VulkanVideoMemory::map_video_buffer(VideoBufferInfo *p_buffer_info) {
	ERR_FAIL_NULL_V(p_buffer_info, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_buffer_info->vk_memory == VK_NULL_HANDLE, ERR_UNCONFIGURED);
	ERR_FAIL_COND_V(p_buffer_info->is_mapped, ERR_ALREADY_IN_USE);

	VkResult result = vkMapMemory(vk_device, p_buffer_info->vk_memory, 0, p_buffer_info->size, 0, &p_buffer_info->mapped_ptr);
	ERR_FAIL_COND_V_MSG(result != VK_SUCCESS, ERR_CANT_CREATE,
			"Failed to map video buffer memory (VkResult: " + itos(result) + ")");

	p_buffer_info->is_mapped = true;
	return OK;
}

void VulkanVideoMemory::unmap_video_buffer(VideoBufferInfo *p_buffer_info) {
	ERR_FAIL_NULL(p_buffer_info);
	
	if (p_buffer_info->is_mapped && p_buffer_info->vk_memory != VK_NULL_HANDLE) {
		vkUnmapMemory(vk_device, p_buffer_info->vk_memory);
		p_buffer_info->mapped_ptr = nullptr;
		p_buffer_info->is_mapped = false;
	}
}

uint32_t VulkanVideoMemory::_find_memory_type_index(uint32_t type_filter, VkMemoryPropertyFlags properties) {
	VkPhysicalDeviceMemoryProperties mem_properties;
	vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

	for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
		if ((type_filter & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}

	ERR_FAIL_V_MSG(UINT32_MAX, "VulkanVideoMemory: Failed to find suitable memory type");
}

Error VulkanVideoMemory::_allocate_image_memory(VideoImageInfo *p_image_info) {
	ERR_FAIL_NULL_V(p_image_info, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_image_info->vk_image == VK_NULL_HANDLE, ERR_UNCONFIGURED);

	// Get memory requirements
	VkMemoryRequirements mem_requirements;
	vkGetImageMemoryRequirements(vk_device, p_image_info->vk_image, &mem_requirements);

	// Allocate memory
	VkMemoryAllocateInfo alloc_info = {};
	alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	alloc_info.allocationSize = mem_requirements.size;
	alloc_info.memoryTypeIndex = _find_memory_type_index(mem_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	VkResult result = vkAllocateMemory(vk_device, &alloc_info, nullptr, &p_image_info->vk_memory);
	ERR_FAIL_COND_V_MSG(result != VK_SUCCESS, ERR_CANT_CREATE,
			"Failed to allocate image memory (VkResult: " + itos(result) + ")");

	// Bind memory to image
	result = vkBindImageMemory(vk_device, p_image_info->vk_image, p_image_info->vk_memory, 0);
	ERR_FAIL_COND_V_MSG(result != VK_SUCCESS, ERR_CANT_CREATE,
			"Failed to bind image memory (VkResult: " + itos(result) + ")");

	return OK;
}

Error VulkanVideoMemory::_allocate_buffer_memory(VideoBufferInfo *p_buffer_info) {
	ERR_FAIL_NULL_V(p_buffer_info, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_buffer_info->vk_buffer == VK_NULL_HANDLE, ERR_UNCONFIGURED);

	// Get memory requirements
	VkMemoryRequirements mem_requirements;
	vkGetBufferMemoryRequirements(vk_device, p_buffer_info->vk_buffer, &mem_requirements);

	// Allocate memory (host visible for CPU access)
	VkMemoryAllocateInfo alloc_info = {};
	alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	alloc_info.allocationSize = mem_requirements.size;
	alloc_info.memoryTypeIndex = _find_memory_type_index(mem_requirements.memoryTypeBits, 
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	VkResult result = vkAllocateMemory(vk_device, &alloc_info, nullptr, &p_buffer_info->vk_memory);
	ERR_FAIL_COND_V_MSG(result != VK_SUCCESS, ERR_CANT_CREATE,
			"Failed to allocate buffer memory (VkResult: " + itos(result) + ")");

	// Bind memory to buffer
	result = vkBindBufferMemory(vk_device, p_buffer_info->vk_buffer, p_buffer_info->vk_memory, 0);
	ERR_FAIL_COND_V_MSG(result != VK_SUCCESS, ERR_CANT_CREATE,
			"Failed to bind buffer memory (VkResult: " + itos(result) + ")");

	return OK;
}

#endif // VULKAN_ENABLED
