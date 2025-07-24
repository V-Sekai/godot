/**************************************************************************/
/*  vulkan_ycbcr_sampler.h                                               */
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

#ifndef VULKAN_YCBCR_SAMPLER_H
#define VULKAN_YCBCR_SAMPLER_H

#include "core/error/error_macros.h"
#include "core/templates/local_vector.h"
#include "core/variant/variant.h"

#ifdef VULKAN_ENABLED

#include "drivers/vulkan/godot_vulkan.h"

class VulkanYCbCrSampler {
public:
	enum ColorSpace {
		COLOR_SPACE_REC_709,    // ITU-R BT.709 (HDTV)
		COLOR_SPACE_REC_601,    // ITU-R BT.601 (SDTV)
		COLOR_SPACE_REC_2020,   // ITU-R BT.2020 (UHDTV)
		COLOR_SPACE_SMPTE_240M  // SMPTE-240M
	};

	enum ColorRange {
		COLOR_RANGE_NARROW,     // Limited range (16-235 for Y, 16-240 for Cb/Cr)
		COLOR_RANGE_FULL        // Full range (0-255)
	};

	enum ChromaLocation {
		CHROMA_LOCATION_COSITED_EVEN,  // Chroma samples are co-sited with even luma samples
		CHROMA_LOCATION_MIDPOINT       // Chroma samples are located at the midpoint between luma samples
	};

	struct YCbCrSamplerCreateInfo {
		VkFormat format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM; // NV12 format
		ColorSpace color_space = COLOR_SPACE_REC_709;
		ColorRange color_range = COLOR_RANGE_NARROW;
		ChromaLocation x_chroma_offset = CHROMA_LOCATION_COSITED_EVEN;
		ChromaLocation y_chroma_offset = CHROMA_LOCATION_COSITED_EVEN;
		VkFilter chroma_filter = VK_FILTER_LINEAR;
		bool force_explicit_reconstruction = false;
	};

	struct YCbCrSamplerInfo {
		VkSamplerYcbcrConversion vk_conversion = VK_NULL_HANDLE;
		VkSampler vk_sampler = VK_NULL_HANDLE;
		VkFormat format = VK_FORMAT_UNDEFINED;
		ColorSpace color_space = COLOR_SPACE_REC_709;
		ColorRange color_range = COLOR_RANGE_NARROW;
		ChromaLocation x_chroma_offset = CHROMA_LOCATION_COSITED_EVEN;
		ChromaLocation y_chroma_offset = CHROMA_LOCATION_COSITED_EVEN;
		bool is_created = false;
	};

private:
	VkDevice vk_device = VK_NULL_HANDLE;

	// Function pointers for YCbCr conversion
	PFN_vkCreateSamplerYcbcrConversionKHR CreateSamplerYcbcrConversionKHR = nullptr;
	PFN_vkDestroySamplerYcbcrConversionKHR DestroySamplerYcbcrConversionKHR = nullptr;

	LocalVector<YCbCrSamplerInfo> samplers;

	VkSamplerYcbcrModelConversion _get_vulkan_color_model(ColorSpace p_color_space);
	VkSamplerYcbcrRange _get_vulkan_color_range(ColorRange p_color_range);
	VkChromaLocation _get_vulkan_chroma_location(ChromaLocation p_chroma_location);

public:
	VulkanYCbCrSampler();
	~VulkanYCbCrSampler();

	Error initialize(VkDevice p_device);
	void finalize();

	bool load_function_pointers(VkInstance p_instance);
	Error create_ycbcr_sampler(const YCbCrSamplerCreateInfo &p_create_info, YCbCrSamplerInfo *p_sampler_info);
	void destroy_ycbcr_sampler(YCbCrSamplerInfo *p_sampler_info);

	// Convenience methods for common video formats
	Error create_nv12_sampler(ColorSpace p_color_space, ColorRange p_color_range, YCbCrSamplerInfo *p_sampler_info);
	Error create_yuv420p_sampler(ColorSpace p_color_space, ColorRange p_color_range, YCbCrSamplerInfo *p_sampler_info);

	// Utility functions
	bool is_initialized() const { return vk_device != VK_NULL_HANDLE; }
	static String get_color_space_name(ColorSpace p_color_space);
	static String get_color_range_name(ColorRange p_color_range);
	static String get_chroma_location_name(ChromaLocation p_chroma_location);
	static bool is_format_supported(VkFormat p_format);
	Dictionary get_sampler_info(const YCbCrSamplerInfo &p_sampler_info) const;
};

#endif // VULKAN_ENABLED

#endif // VULKAN_YCBCR_SAMPLER_H
