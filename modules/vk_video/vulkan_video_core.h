/**************************************************************************/
/*  vulkan_video_core.h                                                  */
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

#ifndef VULKAN_VIDEO_CORE_H
#define VULKAN_VIDEO_CORE_H

#include "core/object/ref_counted.h"
#include "core/templates/vector.h"
#include "servers/rendering/rendering_device.h"
#include "video_decoder_config.h"

// Forward declarations
class RenderingDevice;

// Vulkan Video queue types (extracted from VK spec)
enum VulkanVideoQueueType {
	VULKAN_VIDEO_QUEUE_GRAPHICS = 0x1,
	VULKAN_VIDEO_QUEUE_COMPUTE = 0x2,
	VULKAN_VIDEO_QUEUE_TRANSFER = 0x4,
	VULKAN_VIDEO_QUEUE_DECODE = 0x20,
	VULKAN_VIDEO_QUEUE_ENCODE = 0x40,
};

// Video session state
enum VideoSessionState {
	VIDEO_SESSION_STATE_UNINITIALIZED = 0,
	VIDEO_SESSION_STATE_INITIALIZED = 1,
	VIDEO_SESSION_STATE_CONFIGURED = 2,
	VIDEO_SESSION_STATE_ACTIVE = 3,
	VIDEO_SESSION_STATE_ERROR = 4,
};

// DPB (Decoded Picture Buffer) slot information
struct DPBSlot {
	RID image;
	int array_layer = -1;
	uint64_t frame_number = 0;
	double presentation_time = 0.0;
	bool in_use = false;
	bool is_reference = false;
};

// Video frame information
struct VideoFrameInfo {
	uint64_t frame_number = 0;
	double presentation_time = 0.0;
	bool keyframe = false;
	bool show_frame = true;
	int width = 0;
	int height = 0;
	int dpb_slot = -1;
};

// Hardware capabilities structure
struct VideoHardwareCapabilities {
	bool hardware_decode_supported = false;
	bool hardware_encode_supported = false;
	uint32_t max_width = 0;
	uint32_t max_height = 0;
	uint32_t max_dpb_slots = 0;
	uint32_t max_level = 0;
	Vector<VkVideoCodecProfile> supported_profiles;
	Vector<int> supported_bit_depths;
};

class VulkanVideoCore : public RefCounted {
	GDCLASS(VulkanVideoCore, RefCounted);

private:
	RenderingDevice *rendering_device = nullptr;
	VideoDecoderConfig config;
	VideoSessionState session_state = VIDEO_SESSION_STATE_UNINITIALIZED;
	
	// Vulkan Video resources
	RID video_session;
	RID video_session_parameters;
	RID dpb_image_array;
	RID bitstream_buffer;
	Vector<RID> decode_images;
	Vector<DPBSlot> dpb_slots;
	
	// Queue information
	int video_decode_queue_family = -1;
	int video_decode_queue_index = 0;
	uint32_t video_queue_flags = 0;
	
	// Capabilities
	VideoHardwareCapabilities capabilities;
	bool capabilities_queried = false;

protected:
	static void _bind_methods();

public:
	VulkanVideoCore();
	~VulkanVideoCore();
	
	// Initialization
	Error initialize(RenderingDevice *p_rd, const VideoDecoderConfig &p_config);
	void cleanup();
	
	// Capability queries
	bool is_hardware_supported() const;
	VideoHardwareCapabilities get_hardware_capabilities();
	bool supports_codec(VideoCodecOperation p_codec) const;
	bool supports_profile(VkVideoCodecProfile p_profile) const;
	
	// Session management
	Error create_video_session(int p_width, int p_height);
	Error configure_session_parameters();
	Error begin_video_session();
	Error end_video_session();
	
	// Resource management
	Error create_dpb_images(int p_width, int p_height, int p_num_images);
	Error create_bitstream_buffer(uint32_t p_size);
	int allocate_dpb_slot();
	void release_dpb_slot(int p_slot);
	
	// Decoding operations
	Error decode_frame(const uint8_t *p_bitstream_data, uint32_t p_bitstream_size, 
					  const VideoFrameInfo &p_frame_info);
	RID get_decoded_image(int p_dpb_slot) const;
	
	// State queries
	VideoSessionState get_session_state() const { return session_state; }
	bool is_initialized() const { return session_state >= VIDEO_SESSION_STATE_INITIALIZED; }
	bool is_active() const { return session_state == VIDEO_SESSION_STATE_ACTIVE; }
	
	// Configuration
	const VideoDecoderConfig &get_config() const { return config; }
	void set_config(const VideoDecoderConfig &p_config) { config = p_config; }

private:
	// Internal helper methods
	Error _query_video_capabilities();
	Error _find_video_queue_family();
	Error _create_video_session_internal(int p_width, int p_height);
	Error _validate_codec_support(VideoCodecOperation p_codec) const;
	void _cleanup_resources();
};

#endif // VULKAN_VIDEO_CORE_H
