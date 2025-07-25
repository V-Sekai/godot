/**************************************************************************/
/*  mkv_demuxer_interface.h                                              */
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

#ifndef MKV_DEMUXER_INTERFACE_H
#define MKV_DEMUXER_INTERFACE_H

#include "core/object/ref_counted.h"
#include "core/templates/vector.h"
#include "core/io/file_access.h"
#include "video_decoder_config.h"

// Forward declarations
struct WebMDemuxer;

// Stream types
enum StreamType {
	STREAM_TYPE_UNKNOWN = 0,
	STREAM_TYPE_VIDEO = 1,
	STREAM_TYPE_AUDIO = 2,
	STREAM_TYPE_SUBTITLE = 3,
};

// Codec types
enum CodecType {
	CODEC_TYPE_UNKNOWN = 0,
	CODEC_TYPE_AV1 = 1,
	CODEC_TYPE_VP8 = 2,
	CODEC_TYPE_VP9 = 3,
	CODEC_TYPE_H264 = 4,
	CODEC_TYPE_H265 = 5,
	CODEC_TYPE_OPUS = 10,
	CODEC_TYPE_VORBIS = 11,
};

// Stream information
struct StreamInfo {
	int stream_index = -1;
	StreamType stream_type = STREAM_TYPE_UNKNOWN;
	CodecType codec_type = CODEC_TYPE_UNKNOWN;
	String codec_name;
	
	// Video-specific properties
	int width = 0;
	int height = 0;
	double frame_rate = 0.0;
	int bit_depth = 8;
	
	// Audio-specific properties
	int channels = 0;
	int sample_rate = 0;
	int bits_per_sample = 0;
	
	// Codec private data (sequence headers, etc.)
	PackedByteArray codec_private;
	
	// Duration and timing
	double duration = 0.0;
	uint64_t time_base_num = 1;
	uint64_t time_base_den = 1000;
};

// Packet data
struct PacketData {
	int stream_index = -1;
	PackedByteArray data;
	double timestamp = 0.0;
	double duration = 0.0;
	bool keyframe = false;
	bool end_of_stream = false;
};

class MKVDemuxerInterface : public RefCounted {
	GDCLASS(MKVDemuxerInterface, RefCounted);

private:
	Ref<FileAccess> file;
	WebMDemuxer *demuxer = nullptr;
	Vector<StreamInfo> streams;
	bool initialized = false;
	bool end_of_file = false;
	
	// Current position tracking
	double current_time = 0.0;
	uint64_t current_frame = 0;

protected:
	static void _bind_methods();

public:
	MKVDemuxerInterface();
	virtual ~MKVDemuxerInterface();
	
	// File operations
	Error open_file(const String &p_path);
	void close_file();
	
	// Stream information
	int get_stream_count() const;
	StreamInfo get_stream_info(int p_stream_index) const;
	Vector<StreamInfo> get_all_streams() const;
	
	// Find streams by type
	int find_video_stream() const;
	int find_audio_stream() const;
	Vector<int> find_streams_by_type(StreamType p_type) const;
	Vector<int> find_streams_by_codec(CodecType p_codec) const;
	
	// Demuxing operations
	Error read_packet(PacketData &r_packet);
	Error seek_to_time(double p_time);
	Error seek_to_frame(uint64_t p_frame);
	
	// State queries
	bool is_initialized() const { return initialized; }
	bool is_end_of_file() const { return end_of_file; }
	double get_current_time() const { return current_time; }
	uint64_t get_current_frame() const { return current_frame; }
	double get_duration() const;
	
	// Utility methods
	static CodecType detect_codec_from_fourcc(const String &p_fourcc);
	static String get_codec_name(CodecType p_codec);
	static bool is_video_codec(CodecType p_codec);
	static bool is_audio_codec(CodecType p_codec);

private:
	// Internal helper methods
	Error _initialize_demuxer();
	Error _parse_streams();
	StreamInfo _parse_video_stream(int p_track_index);
	StreamInfo _parse_audio_stream(int p_track_index);
	CodecType _detect_codec_type(const String &p_codec_id);
	void _cleanup_demuxer();
};

#endif // MKV_DEMUXER_INTERFACE_H
