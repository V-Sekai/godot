/**************************************************************************/
/*  av1_vulkan_decoder.h                                                  */
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
#include "scene/resources/image_texture.h"
#include "servers/rendering/rendering_device.h"

class WebMFrame;

class AV1VulkanDecoder : public RefCounted {
	GDCLASS(AV1VulkanDecoder, RefCounted);

private:
	RenderingDevice *rendering_device = nullptr;
	bool hardware_support_available = false;
	bool initialized = false;

	// Video session and parameters
	RID video_session;
	RID video_session_parameters;

	// Frame dimensions
	int frame_width = 0;
	int frame_height = 0;

	// Decoded frame texture
	Ref<ImageTexture> current_frame_texture;

public:
	AV1VulkanDecoder();
	virtual ~AV1VulkanDecoder();

	// Initialization and capability detection
	bool initialize(int width, int height);
	bool is_hardware_supported() const { return hardware_support_available; }
	bool is_initialized() const { return initialized; }

	// Frame decoding
	bool decode_frame(const WebMFrame &frame);
	Ref<Texture2D> get_current_frame() const;

	// Cleanup
	void cleanup();

private:
	bool check_hardware_support();
	bool create_video_session();
	bool create_session_parameters();
	Ref<ImageTexture> create_placeholder_texture() const;

	// Internal resource management
	bool _create_bitstream_buffer(size_t size);
	bool _create_output_image();
	Ref<ImageTexture> _create_texture_from_decoded_frame();
	RID _get_bitstream_buffer() const;
	RID _get_output_image() const;

	// Private member variables for resource tracking
	RID _bitstream_buffer;
	RID _output_image;
	size_t _bitstream_buffer_size = 0;
};
