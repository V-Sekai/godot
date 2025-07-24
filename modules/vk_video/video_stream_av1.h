/**************************************************************************/
/*  video_stream_av1.h                                                   */
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

#include "core/io/resource_loader.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/texture.h"
#include "scene/resources/video_stream.h"
#include "servers/rendering/rendering_device.h"

// Forward declarations for AV1 and container support
class WebMDemuxer;
class OpusVorbisDecoder;
class WebMFrame;
class AV1VulkanDecoder;

// AV1 sequence header structure
struct AV1SequenceHeader {
	uint32_t profile = 0;
	uint32_t level = 0;
	uint32_t tier = 0;
	uint32_t bit_depth = 8;
	uint32_t chroma_subsampling_x = 1;
	uint32_t chroma_subsampling_y = 1;
	uint32_t max_frame_width = 0;
	uint32_t max_frame_height = 0;
	bool monochrome = false;
	bool color_range = false;
	uint32_t color_primaries = 2;
	uint32_t transfer_characteristics = 2;
	uint32_t matrix_coefficients = 2;
};

// AV1 frame header structure
struct AV1FrameHeader {
	uint32_t frame_type = 0;
	uint32_t frame_width = 0;
	uint32_t frame_height = 0;
	uint64_t frame_number = 0;
	double presentation_time = 0.0;
	bool show_frame = true;
	bool keyframe = false;
	uint32_t refresh_frame_flags = 0;
	int32_t ref_frame_idx[7] = { -1, -1, -1, -1, -1, -1, -1 };
};

// Hardware capability information
struct AV1VideoCapabilities {
	bool hardware_decode_supported = false;
	bool hardware_encode_supported = false;
	uint32_t max_width = 0;
	uint32_t max_height = 0;
	uint32_t max_dpb_slots = 0;
	uint32_t max_level = 0;
	Vector<uint32_t> supported_profiles;
};

class VideoStreamPlaybackAV1 : public VideoStreamPlayback {
	GDCLASS(VideoStreamPlaybackAV1, VideoStreamPlayback);

private:
	// File and container handling
	String file_name;
	int audio_track = 0;
	WebMDemuxer *webm = nullptr;
	OpusVorbisDecoder *audio = nullptr;
	WebMFrame *audio_frame = nullptr;

	// Playback state
	bool playing = false;
	bool paused = false;
	double time = 0.0;
	double video_pos = 0.0;
	double delay_compensation = 0.0;

	// Audio handling
	int num_decoded_samples = 0;
	int samples_offset = -1;
	float *pcm = nullptr;
	AudioMixCallback mix_callback = nullptr;
	void *mix_udata = nullptr;

	// AV1 stream information
	AV1SequenceHeader sequence_header;
	AV1VideoCapabilities capabilities;
	bool hardware_decode_available = false;

	// Hardware decoder
	Ref<AV1VulkanDecoder> av1_decoder;

	// Vulkan Video resources (RIDs for abstraction)
	RID video_session;
	RID video_session_parameters;
	RID dpb_image_array;
	RID bitstream_buffer;
	RID output_texture;
	
	// DPB management
	struct DPBSlot {
		bool in_use = false;
		uint64_t frame_number = 0;
		double presentation_time = 0.0;
		int array_layer = -1;
	};
	Vector<DPBSlot> dpb_slots;
	int current_dpb_slot = 0;

	// Frame queue for presentation
	struct QueuedFrame {
		Ref<Texture2D> texture;
		double presentation_time = 0.0;
		uint64_t frame_number = 0;
	};
	Vector<QueuedFrame> frame_queue;
	int max_queued_frames = 3;

	// Statistics
	uint64_t frames_decoded = 0;
	uint64_t frames_dropped = 0;
	double average_decode_time = 0.0;

	// Internal methods
	bool _initialize_hardware_decoder();
	void _cleanup_hardware_resources();
	bool _parse_sequence_header(const uint8_t *data, size_t size);
	bool _parse_frame_header(const uint8_t *data, size_t size, AV1FrameHeader &header);
	bool _decode_frame_hardware(const uint8_t *bitstream_data, size_t bitstream_size, const AV1FrameHeader &header);
	bool _decode_frame_software(const uint8_t *bitstream_data, size_t bitstream_size, const AV1FrameHeader &header);
	int _find_free_dpb_slot();
	void _update_dpb_references(const AV1FrameHeader &header);
	Ref<Texture2D> _get_frame_for_time(double target_time);
	void _cleanup_old_frames(double current_time);

public:
	VideoStreamPlaybackAV1();
	virtual ~VideoStreamPlaybackAV1() override;

	// File operations
	bool open_file(const String &p_file);

	// Playback control
	virtual void stop() override;
	virtual void play() override;
	virtual bool is_playing() const override;
	virtual void set_paused(bool p_paused) override;
	virtual bool is_paused() const override;

	// Timing and seeking
	virtual double get_length() const override;
	virtual double get_playback_position() const override;
	virtual void seek(double p_time) override;

	// Audio
	virtual void set_audio_track(int p_idx) override;
	virtual void set_mix_callback(AudioMixCallback p_callback, void *p_userdata) override;
	virtual int get_channels() const override;
	virtual int get_mix_rate() const override;

	// Video
	virtual Ref<Texture2D> get_texture() const override;
	virtual void update(double p_delta) override;

	// Hardware capability queries
	static bool is_hardware_supported();
	static Dictionary get_hardware_capabilities();
	
	// AV1-specific getters
	Dictionary get_sequence_header() const;
	bool is_hardware_decode_active() const { return hardware_decode_available; }
	uint64_t get_frames_decoded() const { return frames_decoded; }
	uint64_t get_frames_dropped() const { return frames_dropped; }
	double get_average_decode_time() const { return average_decode_time; }

private:
	void delete_pointers();
};

class VideoStreamAV1 : public VideoStream {
	GDCLASS(VideoStreamAV1, VideoStream);

private:
	int audio_track = 0;
	AV1SequenceHeader cached_sequence_header;
	bool sequence_header_valid = false;

protected:
	static void _bind_methods();

public:
	VideoStreamAV1();
	virtual ~VideoStreamAV1();

	virtual Ref<VideoStreamPlayback> instantiate_playback() override;
	virtual void set_audio_track(int p_track) override;

	// AV1-specific methods
	bool load_file(const String &p_path);
	Dictionary get_sequence_header() const;
	bool is_sequence_header_valid() const { return sequence_header_valid; }

	// Hardware capability queries (static)
	static bool is_hardware_supported();
	static Dictionary get_hardware_capabilities();
};

class ResourceFormatLoaderAV1 : public ResourceFormatLoader {
public:
	virtual Ref<Resource> load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_use_sub_threads = false, float *r_progress = nullptr, CacheMode p_cache_mode = CACHE_MODE_REUSE) override;
	virtual void get_recognized_extensions(List<String> *p_extensions) const override;
	virtual bool handles_type(const String &p_type) const override;
	virtual String get_resource_type(const String &p_path) const override;
};
