/**************************************************************************/
/*  vulkan_video_memory.h                                                */
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

#ifndef VULKAN_VIDEO_MEMORY_H
#define VULKAN_VIDEO_MEMORY_H

#include "core/error/error_macros.h"
#include "core/templates/local_vector.h"
#include "core/templates/rid_owner.h"

#ifdef VULKAN_ENABLED

#include "drivers/vulkan/godot_vulkan.h"

class VulkanVideoMemory {
public:
	struct VideoImageInfo {
		VkImage vk_image = VK_NULL_HANDLE;
		VkImageView vk_image_view = VK_NULL_HANDLE;
		VkDeviceMemory vk_memory = VK_NULL_HANDLE;
		VkFormat format = VK_FORMAT_UNDEFINED;
		VkImageUsageFlags usage = 0;
		uint32_t width = 0;
		uint32_t height = 0;
		uint32_t array_layers = 1;
		bool is_video_decode_output = false;
		bool is_video_dpb = false;
		bool is_allocated = false;
	};

	struct DPBSlot {
		VideoImageInfo image_info;
		uint32_t slot_index = UINT32_MAX;
		bool is_active = false;
		bool is_reference = false;
		uint64_t frame_number = 0;
	};

	struct VideoBufferInfo {
		VkBuffer vk_buffer = VK_NULL_HANDLE;
		VkDeviceMemory vk_memory = VK_NULL_HANDLE;
		uint64_t size = 0;
		void *mapped_ptr = nullptr;
		VkBufferUsageFlags usage = 0;
		bool is_mapped = false;
	};

private:
	VkDevice vk_device = VK_NULL_HANDLE;
	VkPhysicalDevice physical_device = VK_NULL_HANDLE;
	uint32_t video_decode_queue_family = UINT32_MAX;

	// DPB (Decoded Picture Buffer) management
	LocalVector<DPBSlot> dpb_slots;
	uint32_t max_dpb_slots = 8;
	uint32_t active_dpb_slots = 0;

	// Video output images
	LocalVector<VideoImageInfo> output_images;
	uint32_t current_output_index = 0;

	// Video buffers for bitstream data
	LocalVector<VideoBufferInfo> video_buffers;

	uint32_t _find_memory_type_index(uint32_t type_filter, VkMemoryPropertyFlags properties);
	Error _allocate_image_memory(VideoImageInfo *p_image_info);
	Error _allocate_buffer_memory(VideoBufferInfo *p_buffer_info);

public:
	VulkanVideoMemory();
	~VulkanVideoMemory();

	Error initialize(VkDevice p_device, VkPhysicalDevice p_physical_device, uint32_t p_video_queue_family);
	void finalize();

	// DPB management
	Error create_dpb_images(uint32_t p_max_slots, uint32_t p_width, uint32_t p_height, VkFormat p_format, const VkVideoProfileInfoKHR *p_video_profile);
	void destroy_dpb_images();
	DPBSlot *acquire_dpb_slot();
	void release_dpb_slot(uint32_t p_slot_index);
	DPBSlot *get_dpb_slot(uint32_t p_slot_index);
	uint32_t get_active_dpb_count() const { return active_dpb_slots; }

	// Video output images
	Error create_output_images(uint32_t p_count, uint32_t p_width, uint32_t p_height, VkFormat p_format, const VkVideoProfileInfoKHR *p_video_profile);
	void destroy_output_images();
	VideoImageInfo *get_current_output_image();
	VideoImageInfo *get_output_image(uint32_t p_index);
	void advance_output_image();
	uint32_t get_output_image_count() const { return output_images.size(); }

	// Video buffers
	Error create_video_buffer(uint64_t p_size, VkBufferUsageFlags p_usage, VideoBufferInfo *p_buffer_info);
	void destroy_video_buffer(VideoBufferInfo *p_buffer_info);
	Error map_video_buffer(VideoBufferInfo *p_buffer_info);
	void unmap_video_buffer(VideoBufferInfo *p_buffer_info);

	// Utility functions
	bool is_initialized() const { return vk_device != VK_NULL_HANDLE; }
	uint32_t get_max_dpb_slots() const { return max_dpb_slots; }
	void set_max_dpb_slots(uint32_t p_max_slots) { max_dpb_slots = p_max_slots; }
};

#endif // VULKAN_ENABLED

#endif // VULKAN_VIDEO_MEMORY_H
