/**************************************************************************/
/*  vulkan_ycbcr_sampler.cpp                                              */
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

#include "vulkan_ycbcr_sampler.h"

#ifdef VULKAN_ENABLED

#include "core/string/print_string.h"

VulkanYCbCrSampler::VulkanYCbCrSampler() {
}

VulkanYCbCrSampler::~VulkanYCbCrSampler() {
	finalize();
}

Error VulkanYCbCrSampler::initialize(VkDevice p_device) {
	ERR_FAIL_COND_V(p_device == VK_NULL_HANDLE, ERR_INVALID_PARAMETER);

	vk_device = p_device;

	print_verbose("VulkanYCbCrSampler: Initialized YCbCr sampler manager");
	return OK;
}

void VulkanYCbCrSampler::finalize() {
	// Destroy all samplers
	for (uint32_t i = 0; i < samplers.size(); i++) {
		destroy_ycbcr_sampler(&samplers[i]);
	}
	samplers.clear();

	// Reset function pointers
	CreateSamplerYcbcrConversionKHR = nullptr;
	DestroySamplerYcbcrConversionKHR = nullptr;

	vk_device = VK_NULL_HANDLE;
	print_verbose("VulkanYCbCrSampler: Finalized YCbCr sampler manager");
}

bool VulkanYCbCrSampler::load_function_pointers(VkInstance p_instance) {
	ERR_FAIL_COND_V(p_instance == VK_NULL_HANDLE, false);

	// Load YCbCr conversion function pointers
	CreateSamplerYcbcrConversionKHR = (PFN_vkCreateSamplerYcbcrConversionKHR)vkGetInstanceProcAddr(p_instance, "vkCreateSamplerYcbcrConversionKHR");
	DestroySamplerYcbcrConversionKHR = (PFN_vkDestroySamplerYcbcrConversionKHR)vkGetInstanceProcAddr(p_instance, "vkDestroySamplerYcbcrConversionKHR");

	bool success = (CreateSamplerYcbcrConversionKHR != nullptr && DestroySamplerYcbcrConversionKHR != nullptr);

	if (success) {
		print_verbose("VulkanYCbCrSampler: Successfully loaded YCbCr conversion function pointers");
	} else {
		WARN_PRINT("VulkanYCbCrSampler: Failed to load YCbCr conversion function pointers");
	}

	return success;
}

Error VulkanYCbCrSampler::create_ycbcr_sampler(const YCbCrSamplerCreateInfo &p_create_info, YCbCrSamplerInfo *p_sampler_info) {
	ERR_FAIL_COND_V(!is_initialized(), ERR_UNCONFIGURED);
	ERR_FAIL_NULL_V(p_sampler_info, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(CreateSamplerYcbcrConversionKHR == nullptr, ERR_UNCONFIGURED);

	// Validate format
	if (!is_format_supported(p_create_info.format)) {
		ERR_PRINT("VulkanYCbCrSampler: Unsupported YCbCr format");
		return ERR_INVALID_PARAMETER;
	}

	// Create YCbCr conversion
	VkSamplerYcbcrConversionCreateInfo conversion_info = {};
	conversion_info.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO;
	conversion_info.format = p_create_info.format;
	conversion_info.ycbcrModel = _get_vulkan_color_model(p_create_info.color_space);
	conversion_info.ycbcrRange = _get_vulkan_color_range(p_create_info.color_range);
	conversion_info.xChromaOffset = _get_vulkan_chroma_location(p_create_info.x_chroma_offset);
	conversion_info.yChromaOffset = _get_vulkan_chroma_location(p_create_info.y_chroma_offset);
	conversion_info.chromaFilter = p_create_info.chroma_filter;
	conversion_info.forceExplicitReconstruction = p_create_info.force_explicit_reconstruction ? VK_TRUE : VK_FALSE;

	VkResult result = CreateSamplerYcbcrConversionKHR(vk_device, &conversion_info, nullptr, &p_sampler_info->vk_conversion);
	ERR_FAIL_COND_V_MSG(result != VK_SUCCESS, ERR_CANT_CREATE,
			"Failed to create YCbCr conversion (VkResult: " + itos(result) + ")");

	// Create sampler with YCbCr conversion
	VkSamplerYcbcrConversionInfo conversion_sampler_info = {};
	conversion_sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO;
	conversion_sampler_info.conversion = p_sampler_info->vk_conversion;

	VkSamplerCreateInfo sampler_info = {};
	sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	sampler_info.pNext = &conversion_sampler_info;
	sampler_info.magFilter = VK_FILTER_LINEAR;
	sampler_info.minFilter = VK_FILTER_LINEAR;
	sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.mipLodBias = 0.0f;
	sampler_info.anisotropyEnable = VK_FALSE;
	sampler_info.compareEnable = VK_FALSE;
	sampler_info.minLod = 0.0f;
	sampler_info.maxLod = 0.0f;
	sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
	sampler_info.unnormalizedCoordinates = VK_FALSE;

	result = vkCreateSampler(vk_device, &sampler_info, nullptr, &p_sampler_info->vk_sampler);
	if (result != VK_SUCCESS) {
		DestroySamplerYcbcrConversionKHR(vk_device, p_sampler_info->vk_conversion, nullptr);
		ERR_FAIL_V_MSG(ERR_CANT_CREATE, "Failed to create YCbCr sampler (VkResult: " + itos(result) + ")");
	}

	// Fill sampler info
	p_sampler_info->format = p_create_info.format;
	p_sampler_info->color_space = p_create_info.color_space;
	p_sampler_info->color_range = p_create_info.color_range;
	p_sampler_info->x_chroma_offset = p_create_info.x_chroma_offset;
	p_sampler_info->y_chroma_offset = p_create_info.y_chroma_offset;
	p_sampler_info->is_created = true;

	print_verbose("VulkanYCbCrSampler: Created YCbCr sampler for format " + itos(p_create_info.format) +
			" with color space " + get_color_space_name(p_create_info.color_space));

	return OK;
}

void VulkanYCbCrSampler::destroy_ycbcr_sampler(YCbCrSamplerInfo *p_sampler_info) {
	ERR_FAIL_NULL(p_sampler_info);

	if (!p_sampler_info->is_created) {
		return;
	}

	if (p_sampler_info->vk_sampler != VK_NULL_HANDLE) {
		vkDestroySampler(vk_device, p_sampler_info->vk_sampler, nullptr);
		p_sampler_info->vk_sampler = VK_NULL_HANDLE;
	}

	if (p_sampler_info->vk_conversion != VK_NULL_HANDLE && DestroySamplerYcbcrConversionKHR != nullptr) {
		DestroySamplerYcbcrConversionKHR(vk_device, p_sampler_info->vk_conversion, nullptr);
		p_sampler_info->vk_conversion = VK_NULL_HANDLE;
	}

	p_sampler_info->is_created = false;
	print_verbose("VulkanYCbCrSampler: Destroyed YCbCr sampler");
}

Error VulkanYCbCrSampler::create_nv12_sampler(ColorSpace p_color_space, ColorRange p_color_range, YCbCrSamplerInfo *p_sampler_info) {
	YCbCrSamplerCreateInfo create_info;
	create_info.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM; // NV12
	create_info.color_space = p_color_space;
	create_info.color_range = p_color_range;
	create_info.x_chroma_offset = CHROMA_LOCATION_COSITED_EVEN;
	create_info.y_chroma_offset = CHROMA_LOCATION_COSITED_EVEN;
	create_info.chroma_filter = VK_FILTER_LINEAR;
	create_info.force_explicit_reconstruction = false;

	return create_ycbcr_sampler(create_info, p_sampler_info);
}

Error VulkanYCbCrSampler::create_yuv420p_sampler(ColorSpace p_color_space, ColorRange p_color_range, YCbCrSamplerInfo *p_sampler_info) {
	YCbCrSamplerCreateInfo create_info;
	create_info.format = VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM; // YUV420P
	create_info.color_space = p_color_space;
	create_info.color_range = p_color_range;
	create_info.x_chroma_offset = CHROMA_LOCATION_COSITED_EVEN;
	create_info.y_chroma_offset = CHROMA_LOCATION_COSITED_EVEN;
	create_info.chroma_filter = VK_FILTER_LINEAR;
	create_info.force_explicit_reconstruction = false;

	return create_ycbcr_sampler(create_info, p_sampler_info);
}

VkSamplerYcbcrModelConversion VulkanYCbCrSampler::_get_vulkan_color_model(ColorSpace p_color_space) {
	switch (p_color_space) {
		case COLOR_SPACE_REC_709:
			return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709;
		case COLOR_SPACE_REC_601:
			return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601;
		case COLOR_SPACE_REC_2020:
			return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_2020;
		case COLOR_SPACE_SMPTE_240M:
			// SMPTE-240M uses similar conversion to Rec.709
			return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709;
		default:
			return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709;
	}
}

VkSamplerYcbcrRange VulkanYCbCrSampler::_get_vulkan_color_range(ColorRange p_color_range) {
	switch (p_color_range) {
		case COLOR_RANGE_NARROW:
			return VK_SAMPLER_YCBCR_RANGE_ITU_NARROW;
		case COLOR_RANGE_FULL:
			return VK_SAMPLER_YCBCR_RANGE_ITU_FULL;
		default:
			return VK_SAMPLER_YCBCR_RANGE_ITU_NARROW;
	}
}

VkChromaLocation VulkanYCbCrSampler::_get_vulkan_chroma_location(ChromaLocation p_chroma_location) {
	switch (p_chroma_location) {
		case CHROMA_LOCATION_COSITED_EVEN:
			return VK_CHROMA_LOCATION_COSITED_EVEN;
		case CHROMA_LOCATION_MIDPOINT:
			return VK_CHROMA_LOCATION_MIDPOINT;
		default:
			return VK_CHROMA_LOCATION_COSITED_EVEN;
	}
}

String VulkanYCbCrSampler::get_color_space_name(ColorSpace p_color_space) {
	switch (p_color_space) {
		case COLOR_SPACE_REC_709:
			return "Rec.709 (HDTV)";
		case COLOR_SPACE_REC_601:
			return "Rec.601 (SDTV)";
		case COLOR_SPACE_REC_2020:
			return "Rec.2020 (UHDTV)";
		case COLOR_SPACE_SMPTE_240M:
			return "SMPTE-240M";
		default:
			return "Unknown";
	}
}

String VulkanYCbCrSampler::get_color_range_name(ColorRange p_color_range) {
	switch (p_color_range) {
		case COLOR_RANGE_NARROW:
			return "Narrow (16-235)";
		case COLOR_RANGE_FULL:
			return "Full (0-255)";
		default:
			return "Unknown";
	}
}

String VulkanYCbCrSampler::get_chroma_location_name(ChromaLocation p_chroma_location) {
	switch (p_chroma_location) {
		case CHROMA_LOCATION_COSITED_EVEN:
			return "Co-sited Even";
		case CHROMA_LOCATION_MIDPOINT:
			return "Midpoint";
		default:
			return "Unknown";
	}
}

bool VulkanYCbCrSampler::is_format_supported(VkFormat p_format) {
	switch (p_format) {
		case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM: // NV12
		case VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM: // YUV420P
		case VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16: // P010
		case VK_FORMAT_G16_B16R16_2PLANE_420_UNORM: // P016
		case VK_FORMAT_G8_B8R8_2PLANE_422_UNORM: // NV16
		case VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM: // YUV422P
		case VK_FORMAT_G8_B8R8_2PLANE_444_UNORM: // NV24
		case VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM: // YUV444P
			return true;
		default:
			return false;
	}
}

Dictionary VulkanYCbCrSampler::get_sampler_info(const YCbCrSamplerInfo &p_sampler_info) const {
	Dictionary info;

	info["is_created"] = p_sampler_info.is_created;
	info["format"] = (int)p_sampler_info.format;
	info["color_space"] = get_color_space_name(p_sampler_info.color_space);
	info["color_range"] = get_color_range_name(p_sampler_info.color_range);
	info["x_chroma_offset"] = get_chroma_location_name(p_sampler_info.x_chroma_offset);
	info["y_chroma_offset"] = get_chroma_location_name(p_sampler_info.y_chroma_offset);

	return info;
}

#endif // VULKAN_ENABLED
