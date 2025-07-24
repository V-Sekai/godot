/**************************************************************************/
/*  rendering_device_video_extensions.h                                  */
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
#include "servers/rendering/rendering_device.h"
#include "servers/rendering/rendering_device_driver.h"

#ifdef VULKAN_ENABLED
class VulkanVideoContext;
#endif

// Video codec profiles
enum VideoCodecProfile {
	VIDEO_CODEC_PROFILE_H264_BASELINE = 0,
	VIDEO_CODEC_PROFILE_H264_MAIN = 1,
	VIDEO_CODEC_PROFILE_H264_HIGH = 2,
	VIDEO_CODEC_PROFILE_H265_MAIN = 10,
	VIDEO_CODEC_PROFILE_H265_MAIN_10 = 11,
	VIDEO_CODEC_PROFILE_AV1_MAIN = 20,
	VIDEO_CODEC_PROFILE_AV1_HIGH = 21,
	VIDEO_CODEC_PROFILE_AV1_PROFESSIONAL = 22,
};

// Video operation types
enum VideoOperationType {
	VIDEO_OPERATION_DECODE = 0,
	VIDEO_OPERATION_ENCODE = 1,
};

VARIANT_ENUM_CAST(VideoCodecProfile);
VARIANT_ENUM_CAST(VideoOperationType);

// Note: Using Dictionary for configuration to avoid Godot binding issues
// Dictionary keys are documented in the implementation

// Video capability query results
struct VideoCapabilities {
	bool decode_supported = false;
	bool encode_supported = false;
	uint32_t max_width = 0;
	uint32_t max_height = 0;
	uint32_t max_dpb_slots = 0;
	uint32_t max_active_references = 0;
	Vector<VideoCodecProfile> supported_profiles;
	Vector<RD::DataFormat> supported_formats;
};

// Extension class for RenderingDevice to add video functionality
class RenderingDeviceVideoExtensions : public RefCounted {
	GDCLASS(RenderingDeviceVideoExtensions, RefCounted);

private:
	RenderingDevice *rd = nullptr;
	
#ifdef VULKAN_ENABLED
	VulkanVideoContext *video_context = nullptr;
#endif

protected:
	static void _bind_methods();

public:
	RenderingDeviceVideoExtensions();
	virtual ~RenderingDeviceVideoExtensions();

	void initialize(RenderingDevice *p_rendering_device);

	// Video capability queries
	bool is_video_supported() const;
	Dictionary get_video_capabilities(VideoCodecProfile p_profile, VideoOperationType p_operation) const;
	Array get_supported_profiles() const;

	// Video session management
	RID video_session_create(const Dictionary &p_create_info);
	void video_session_destroy(RID p_video_session);
	
	RID video_session_parameters_create(const Dictionary &p_create_info);
	void video_session_parameters_destroy(RID p_video_session_parameters);

	// Video resource creation
	RID video_image_create(const Dictionary &p_create_info);
	void video_image_destroy(RID p_video_image);
	
	RID video_buffer_create(const Dictionary &p_create_info);
	void video_buffer_destroy(RID p_video_buffer);

	// Video operations
	void video_decode_frame(const Dictionary &p_decode_info);
	void video_queue_submit();
	void video_queue_wait_idle();

	// Utility functions
	RID texture_from_video_image(RID p_video_image, uint32_t p_layer = 0);
	void copy_video_image_to_texture(RID p_video_image, uint32_t p_src_layer, RID p_dst_texture);

	// Memory management
	void video_buffer_update(RID p_video_buffer, uint64_t p_offset, const Vector<uint8_t> &p_data);
	Vector<uint8_t> video_buffer_get_data(RID p_video_buffer, uint64_t p_offset = 0, uint64_t p_size = 0);

private:
	// Internal implementation details
	bool _check_video_support() const;
	bool _check_codec_support(VideoCodecProfile p_profile, VideoOperationType p_operation) const;
	void _initialize_video_queues();
	void _cleanup_video_resources();
};
