/**************************************************************************/
/*  video_stream_av1.cpp                                                 */
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

#include "video_stream_av1.h"

#include "core/config/project_settings.h"
#include "core/error/error_macros.h"
#include "core/io/file_access.h"
#include "core/os/os.h"
#include "servers/audio_server.h"
#include "servers/rendering_server.h"
#include "scene/resources/audio_video_synchronizer.h"

// libsimplewebm
#include <OpusVorbisDecoder.hpp>
#include <WebMDemuxer.hpp>

// libwebm
#include <mkvparser/mkvparser.h>

// AV1 bitstream parsing constants
#define AV1_OBU_SEQUENCE_HEADER 1
#define AV1_OBU_TEMPORAL_DELIMITER 2
#define AV1_OBU_FRAME_HEADER 3
#define AV1_OBU_TILE_GROUP 4
#define AV1_OBU_METADATA 5
#define AV1_OBU_FRAME 6
#define AV1_OBU_REDUNDANT_FRAME_HEADER 7
#define AV1_OBU_TILE_LIST 8
#define AV1_OBU_PADDING 15

VideoStreamPlaybackAV1::VideoStreamPlaybackAV1() {
	// Initialize audio-video synchronization
	av_synchronizer.instantiate();
	av_synchronizer->set_sync_mode(AudioVideoSynchronizer::SYNC_MODE_AUDIO_MASTER);
	av_synchronizer->set_use_timing_filter(true);
	av_synchronizer->set_sync_threshold(0.040); // 40ms threshold
	
	// Initialize DPB slots
	dpb_slots.resize(8); // AV1 supports up to 8 reference frames
	for (int i = 0; i < dpb_slots.size(); i++) {
		dpb_slots.write[i].array_layer = i;
	}
}

VideoStreamPlaybackAV1::~VideoStreamPlaybackAV1() {
	delete_pointers();
}

void VideoStreamPlaybackAV1::delete_pointers() {
	_cleanup_hardware_resources();
	
	if (pcm) {
		memfree(pcm);
		pcm = nullptr;
	}

	if (audio_frame) {
		memdelete(audio_frame);
		audio_frame = nullptr;
	}

	if (audio) {
		memdelete(audio);
		audio = nullptr;
	}

	if (webm) {
		memdelete(webm);
		webm = nullptr;
	}
}

bool VideoStreamPlaybackAV1::open_file(const String &p_file) {
	file_name = p_file;
	
	// Open MKV/WebM container
	Ref<FileAccess> f = FileAccess::open(p_file, FileAccess::READ);
	if (f.is_null()) {
		ERR_PRINT("Cannot open file: " + p_file);
		return false;
	}
	
	// Use the same MkvReader pattern as VideoStreamMKV
	class MkvReader : public mkvparser::IMkvReader {
	public:
		MkvReader(const String &p_file) {
			file = FileAccess::open(p_file, FileAccess::READ);
			ERR_FAIL_COND_MSG(file.is_null(), "Failed loading resource: '" + p_file + "'.");
		}
		~MkvReader() {}

		virtual int Read(long long pos, long len, unsigned char *buf) {
			if (file.is_valid()) {
				if (file->get_position() != (uint64_t)pos) {
					file->seek(pos);
				}
				if (file->get_buffer(buf, len) == (uint64_t)len) {
					return 0;
				}
			}
			return -1;
		}

		virtual int Length(long long *total, long long *available) {
			if (file.is_valid()) {
				const uint64_t len = file->get_length();
				if (total) {
					*total = len;
				}
				if (available) {
					*available = len;
				}
				return 0;
			}
			return -1;
		}

	private:
		Ref<FileAccess> file;
	};

	webm = memnew(WebMDemuxer(new MkvReader(p_file), 0, audio_track));
	if (!webm->isOpen()) {
		ERR_PRINT("Failed to open WebM demuxer for: " + p_file);
		delete_pointers();
		return false;
	}
	
	// Initialize audio decoder
	audio = memnew(OpusVorbisDecoder(*webm));
	if (audio->isOpen()) {
		audio_frame = memnew(WebMFrame);
		pcm = (float *)memalloc(sizeof(float) * audio->getBufferSamples() * webm->getChannels());
	} else {
		memdelete(audio);
		audio = nullptr;
	}
	
	// Parse AV1 sequence header from the first video packet
	WebMFrame video_frame;
	if (webm->readFrame(&video_frame, nullptr)) {
		if (video_frame.isValid()) {
			if (!_parse_sequence_header(video_frame.buffer, video_frame.bufferSize)) {
				ERR_PRINT("Failed to parse AV1 sequence header");
				delete_pointers();
				return false;
			}
		}
		webm->seek(0.0); // Reset to beginning
	}
	
	// Check hardware capabilities
	Dictionary caps_dict = get_hardware_capabilities();
	capabilities.hardware_decode_supported = caps_dict.get("hardware_decode_supported", false);
	capabilities.max_width = caps_dict.get("max_width", 0);
	capabilities.max_height = caps_dict.get("max_height", 0);
	capabilities.max_dpb_slots = caps_dict.get("max_dpb_slots", 0);
	capabilities.max_level = caps_dict.get("max_level", 0);
	
	hardware_decode_available = capabilities.hardware_decode_supported &&
								sequence_header.max_frame_width <= capabilities.max_width &&
								sequence_header.max_frame_height <= capabilities.max_height;
	
	if (hardware_decode_available) {
		if (!_initialize_hardware_decoder()) {
			WARN_PRINT("Hardware AV1 decoder initialization failed, falling back to software");
			hardware_decode_available = false;
		}
	}
	
	if (!hardware_decode_available) {
		WARN_PRINT("AV1 hardware decoding not available, software fallback not implemented yet");
		// TODO: Initialize software decoder fallback
	}
	
	return true;
}

bool VideoStreamPlaybackAV1::_parse_sequence_header(const uint8_t *data, size_t size) {
	// Simplified AV1 sequence header parsing
	// In a full implementation, this would use a proper AV1 bitstream parser
	if (size < 8) {
		return false;
	}
	
	// Look for sequence header OBU
	for (size_t i = 0; i < size - 4; i++) {
		uint8_t obu_header = data[i];
		uint8_t obu_type = (obu_header >> 3) & 0xF;
		
		if (obu_type == AV1_OBU_SEQUENCE_HEADER) {
			// Parse basic sequence header fields
			// This is a simplified parser - real implementation would be more complex
			sequence_header.profile = (data[i + 2] >> 5) & 0x7;
			sequence_header.level = data[i + 3] & 0x1F;
			sequence_header.bit_depth = 8; // Default, would need proper parsing
			sequence_header.max_frame_width = 1920; // Default, would need proper parsing
			sequence_header.max_frame_height = 1080; // Default, would need proper parsing
			return true;
		}
	}
	
	return false;
}

bool VideoStreamPlaybackAV1::_parse_frame_header(const uint8_t *data, size_t size, AV1FrameHeader &header) {
	// Simplified frame header parsing
	// Real implementation would use proper AV1 bitstream parser
	if (size < 4) {
		return false;
	}
	
	// Look for frame header OBU
	for (size_t i = 0; i < size - 4; i++) {
		uint8_t obu_header = data[i];
		uint8_t obu_type = (obu_header >> 3) & 0xF;
		
		if (obu_type == AV1_OBU_FRAME_HEADER || obu_type == AV1_OBU_FRAME) {
			// Parse basic frame header fields
			header.frame_type = data[i + 2] & 0x3;
			header.keyframe = (header.frame_type == 0);
			header.show_frame = true; // Default
			header.frame_width = sequence_header.max_frame_width;
			header.frame_height = sequence_header.max_frame_height;
			return true;
		}
	}
	
	return false;
}

bool VideoStreamPlaybackAV1::_initialize_hardware_decoder() {
	RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
	if (!rd) {
		return false;
	}
	
	// TODO: Implement RenderingDevice video extensions
	// This would call new methods like:
	// video_session = rd->video_session_create(video_profile);
	// video_session_parameters = rd->video_session_parameters_create(video_session);
	// dpb_image_array = rd->video_image_create(dpb_create_info);
	// bitstream_buffer = rd->video_buffer_create(bitstream_create_info);
	
	WARN_PRINT("Hardware decoder initialization not yet implemented - RenderingDevice extensions needed");
	return false;
}

void VideoStreamPlaybackAV1::_cleanup_hardware_resources() {
	RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
	if (!rd) {
		return;
	}
	
	// TODO: Cleanup video resources
	// rd->video_session_destroy(video_session);
	// rd->video_session_parameters_destroy(video_session_parameters);
	// etc.
	
	video_session = RID();
	video_session_parameters = RID();
	dpb_image_array = RID();
	bitstream_buffer = RID();
	output_texture = RID();
}

bool VideoStreamPlaybackAV1::_decode_frame_hardware(const uint8_t *bitstream_data, size_t bitstream_size, const AV1FrameHeader &header) {
	// TODO: Implement hardware decoding using RenderingDevice video extensions
	// This would involve:
	// 1. Upload bitstream data to GPU buffer
	// 2. Record video decode commands
	// 3. Submit to video queue
	// 4. Wait for completion and get decoded frame
	
	WARN_PRINT("Hardware frame decoding not yet implemented");
	return false;
}

bool VideoStreamPlaybackAV1::_decode_frame_software(const uint8_t *bitstream_data, size_t bitstream_size, const AV1FrameHeader &header) {
	// TODO: Implement software AV1 decoding fallback
	// This could use libaom or dav1d library
	
	WARN_PRINT("Software frame decoding not yet implemented");
	return false;
}

int VideoStreamPlaybackAV1::_find_free_dpb_slot() {
	for (int i = 0; i < dpb_slots.size(); i++) {
		if (!dpb_slots[i].in_use) {
			return i;
		}
	}
	
	// If no free slots, find oldest frame
	int oldest_slot = 0;
	uint64_t oldest_frame = dpb_slots[0].frame_number;
	for (int i = 1; i < dpb_slots.size(); i++) {
		if (dpb_slots[i].frame_number < oldest_frame) {
			oldest_frame = dpb_slots[i].frame_number;
			oldest_slot = i;
		}
	}
	
	return oldest_slot;
}

void VideoStreamPlaybackAV1::_update_dpb_references(const AV1FrameHeader &header) {
	// Update DPB based on refresh frame flags
	if (header.keyframe) {
		// Keyframes clear all references
		for (int i = 0; i < dpb_slots.size(); i++) {
			dpb_slots.write[i].in_use = false;
		}
	}
	
	// Mark current slot as in use
	int slot = _find_free_dpb_slot();
	dpb_slots.write[slot].in_use = true;
	dpb_slots.write[slot].frame_number = header.frame_number;
	dpb_slots.write[slot].presentation_time = header.presentation_time;
	current_dpb_slot = slot;
}

Ref<Texture2D> VideoStreamPlaybackAV1::_get_frame_for_time(double target_time) {
	// Use audio-video synchronizer to get the appropriate frame
	if (use_synchronization && av_synchronizer.is_valid()) {
		return av_synchronizer->get_current_frame();
	}
	
	// Fallback: find closest frame in queue
	if (frame_queue.is_empty()) {
		return Ref<Texture2D>();
	}
	
	int best_frame = 0;
	double best_diff = Math::abs(frame_queue[0].presentation_time - target_time);
	
	for (int i = 1; i < frame_queue.size(); i++) {
		double diff = Math::abs(frame_queue[i].presentation_time - target_time);
		if (diff < best_diff) {
			best_diff = diff;
			best_frame = i;
		}
	}
	
	return frame_queue[best_frame].texture;
}

void VideoStreamPlaybackAV1::_cleanup_old_frames(double current_time) {
	// Remove frames that are too old
	const double max_age = 1.0; // 1 second
	
	for (int i = frame_queue.size() - 1; i >= 0; i--) {
		if (current_time - frame_queue[i].presentation_time > max_age) {
			frame_queue.remove_at(i);
		}
	}
}

void VideoStreamPlaybackAV1::play() {
	if (!webm || !audio) {
		return;
	}
	
	playing = true;
	paused = false;
	
	if (use_synchronization && av_synchronizer.is_valid()) {
		av_synchronizer->reset();
	}
}

void VideoStreamPlaybackAV1::stop() {
	playing = false;
	paused = false;
	time = 0.0;
	video_pos = 0.0;
	
	if (webm) {
		webm->seek(0.0);
	}
	
	frame_queue.clear();
	
	if (use_synchronization && av_synchronizer.is_valid()) {
		av_synchronizer->clear_frame_queue();
	}
}

bool VideoStreamPlaybackAV1::is_playing() const {
	return playing && !paused;
}

void VideoStreamPlaybackAV1::set_paused(bool p_paused) {
	paused = p_paused;
}

bool VideoStreamPlaybackAV1::is_paused() const {
	return paused;
}

double VideoStreamPlaybackAV1::get_length() const {
	if (!webm) {
		return 0.0;
	}
	return webm->getLength();
}

double VideoStreamPlaybackAV1::get_playback_position() const {
	return time;
}

void VideoStreamPlaybackAV1::seek(double p_time) {
	if (!webm) {
		return;
	}
	
	time = webm->seek(p_time);
	video_pos = time;
	
	frame_queue.clear();
	
	if (use_synchronization && av_synchronizer.is_valid()) {
		av_synchronizer->clear_frame_queue();
	}
}

void VideoStreamPlaybackAV1::set_audio_track(int p_idx) {
	audio_track = p_idx;
}

Ref<Texture2D> VideoStreamPlaybackAV1::get_texture() const {
	if (use_synchronization && av_synchronizer.is_valid()) {
		return av_synchronizer->get_current_frame();
	}
	
	// Fallback to direct frame queue access
	return const_cast<VideoStreamPlaybackAV1*>(this)->_get_frame_for_time(time);
}

void VideoStreamPlaybackAV1::update(double p_delta) {
	if (!playing || paused || !webm) {
		return;
	}
	
	time += p_delta;
	
	// Update audio-video synchronization clocks
	if (use_synchronization && av_synchronizer.is_valid()) {
		av_synchronizer->update_master_clock(time);
		av_synchronizer->update_video_clock(video_pos);
	}
	
	bool audio_buffer_full = false;

	if (samples_offset > -1) {
		//Mix remaining samples
		const int to_read = num_decoded_samples - samples_offset;
		const int mixed = mix_callback(mix_udata, pcm + samples_offset * webm->getChannels(), to_read);
		if (mixed != to_read) {
			samples_offset += mixed;
			audio_buffer_full = true;
		} else {
			samples_offset = -1;
		}
	}

	const bool hasAudio = (audio && mix_callback);

	// Process both video and audio frames
	if (hasAudio && !audio_buffer_full) {
		WebMFrame video_frame; // Video frame for AV1 processing

		if (!webm->readFrame(&video_frame, audio_frame)) {
			// Can't demux, EOS?
			if (webm->isEOS()) {
				stop();
			}
			return;
		}

		// Process video frame for AV1 decoding
		if (video_frame.isValid()) {
			AV1FrameHeader frame_header;
			if (_parse_frame_header(video_frame.buffer, video_frame.bufferSize, frame_header)) {
				frame_header.presentation_time = video_frame.time;
				frame_header.frame_number = frames_decoded;
				
				bool decode_success = false;
				if (hardware_decode_available) {
					decode_success = _decode_frame_hardware(video_frame.buffer, video_frame.bufferSize, frame_header);
				} else {
					decode_success = _decode_frame_software(video_frame.buffer, video_frame.bufferSize, frame_header);
				}
				
				if (decode_success) {
					frames_decoded++;
					_update_dpb_references(frame_header);
					
					// TODO: Create texture from decoded frame and queue it
					// For now, create a placeholder
					Ref<ImageTexture> placeholder_texture;
					placeholder_texture.instantiate();
					
					if (use_synchronization && av_synchronizer.is_valid()) {
						av_synchronizer->queue_video_frame(placeholder_texture, frame_header.presentation_time, frame_header.frame_number);
					} else {
						QueuedFrame queued_frame;
						queued_frame.texture = placeholder_texture;
						queued_frame.presentation_time = frame_header.presentation_time;
						queued_frame.frame_number = frame_header.frame_number;
						frame_queue.push_back(queued_frame);
					}
				} else {
					frames_dropped++;
				}
			}
			
			video_pos = video_frame.time;
		}

		// Process audio frame
		if (audio_frame->isValid() && audio->getPCMF(*audio_frame, pcm, num_decoded_samples) && num_decoded_samples > 0) {
			if (use_synchronization && av_synchronizer.is_valid()) {
				av_synchronizer->update_audio_clock(audio_frame->time);
			}

			const int mixed = mix_callback(mix_udata, pcm, num_decoded_samples);

			if (mixed != num_decoded_samples) {
				samples_offset = mixed;
				audio_buffer_full = true;
			}
		}
	}

	if (webm && webm->isEOS()) {
		stop();
	}
	
	// Cleanup old frames
	_cleanup_old_frames(time);
}

void VideoStreamPlaybackAV1::set_mix_callback(AudioMixCallback p_callback, void *p_userdata) {
	mix_callback = p_callback;
	mix_udata = p_userdata;
	
	if (audio && p_callback) {
		if (!pcm) {
			pcm = (float *)memalloc(sizeof(float) * 1024 * get_channels());
		}
	}
}

int VideoStreamPlaybackAV1::get_channels() const {
	if (audio) {
		return webm->getChannels();
	}
	return 0;
}

int VideoStreamPlaybackAV1::get_mix_rate() const {
	if (audio) {
		return webm->getSampleRate();
	}
	return 0;
}

bool VideoStreamPlaybackAV1::is_hardware_supported() {
	return VideoStreamAV1::is_hardware_supported();
}

Dictionary VideoStreamPlaybackAV1::get_sequence_header() const {
	Dictionary header;
	header["profile"] = sequence_header.profile;
	header["level"] = sequence_header.level;
	header["tier"] = sequence_header.tier;
	header["bit_depth"] = sequence_header.bit_depth;
	header["chroma_subsampling_x"] = sequence_header.chroma_subsampling_x;
	header["chroma_subsampling_y"] = sequence_header.chroma_subsampling_y;
	header["max_frame_width"] = sequence_header.max_frame_width;
	header["max_frame_height"] = sequence_header.max_frame_height;
	header["monochrome"] = sequence_header.monochrome;
	header["color_range"] = sequence_header.color_range;
	header["color_primaries"] = sequence_header.color_primaries;
	header["transfer_characteristics"] = sequence_header.transfer_characteristics;
	header["matrix_coefficients"] = sequence_header.matrix_coefficients;
	return header;
}

Dictionary VideoStreamPlaybackAV1::get_hardware_capabilities() {
	return VideoStreamAV1::get_hardware_capabilities();
}

// VideoStreamAV1 implementation

VideoStreamAV1::VideoStreamAV1() {
}

VideoStreamAV1::~VideoStreamAV1() {
}

void VideoStreamAV1::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_audio_track", "track"), &VideoStreamAV1::set_audio_track);
	ClassDB::bind_method(D_METHOD("get_sequence_header"), &VideoStreamAV1::get_sequence_header);
	ClassDB::bind_method(D_METHOD("is_sequence_header_valid"), &VideoStreamAV1::is_sequence_header_valid);
	ClassDB::bind_static_method("VideoStreamAV1", D_METHOD("is_hardware_supported"), &VideoStreamAV1::is_hardware_supported);
	ClassDB::bind_static_method("VideoStreamAV1", D_METHOD("get_hardware_capabilities"), &VideoStreamAV1::get_hardware_capabilities);
}

Ref<VideoStreamPlayback> VideoStreamAV1::instantiate_playback() {
	Ref<VideoStreamPlaybackAV1> pb;
	pb.instantiate();
	pb->set_audio_track(audio_track);
	if (pb->open_file(file)) {
		return pb;
	}
	return Ref<VideoStreamPlayback>();
}

void VideoStreamAV1::set_audio_track(int p_track) {
	audio_track = p_track;
}

Dictionary VideoStreamAV1::get_sequence_header() const {
	Dictionary header;
	if (sequence_header_valid) {
		header["profile"] = cached_sequence_header.profile;
		header["level"] = cached_sequence_header.level;
		header["tier"] = cached_sequence_header.tier;
		header["bit_depth"] = cached_sequence_header.bit_depth;
		header["chroma_subsampling_x"] = cached_sequence_header.chroma_subsampling_x;
		header["chroma_subsampling_y"] = cached_sequence_header.chroma_subsampling_y;
		header["max_frame_width"] = cached_sequence_header.max_frame_width;
		header["max_frame_height"] = cached_sequence_header.max_frame_height;
		header["monochrome"] = cached_sequence_header.monochrome;
		header["color_range"] = cached_sequence_header.color_range;
		header["color_primaries"] = cached_sequence_header.color_primaries;
		header["transfer_characteristics"] = cached_sequence_header.transfer_characteristics;
		header["matrix_coefficients"] = cached_sequence_header.matrix_coefficients;
	}
	return header;
}

bool VideoStreamAV1::load_file(const String &p_path) {
	file = p_path;
	
	// Parse sequence header for validation
	Ref<VideoStreamPlaybackAV1> temp_playback;
	temp_playback.instantiate();
	if (temp_playback->open_file(p_path)) {
		Dictionary seq_header_dict = temp_playback->get_sequence_header();
		// Convert Dictionary back to struct for internal storage
		cached_sequence_header.profile = seq_header_dict.get("profile", 0);
		cached_sequence_header.level = seq_header_dict.get("level", 0);
		cached_sequence_header.tier = seq_header_dict.get("tier", 0);
		cached_sequence_header.bit_depth = seq_header_dict.get("bit_depth", 8);
		cached_sequence_header.max_frame_width = seq_header_dict.get("max_frame_width", 0);
		cached_sequence_header.max_frame_height = seq_header_dict.get("max_frame_height", 0);
		sequence_header_valid = true;
		return true;
	}
	
	return false;
}

bool VideoStreamAV1::is_hardware_supported() {
	RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
	if (!rd) {
		return false;
	}
	
	// TODO: Query RenderingDevice for Vulkan Video support
	// This would check for VK_KHR_video_decode_av1 extension
	// For now, return false until RenderingDevice extensions are implemented
	return false;
}

Dictionary VideoStreamAV1::get_hardware_capabilities() {
	Dictionary caps;
	
	if (!is_hardware_supported()) {
		caps["hardware_decode_supported"] = false;
		caps["hardware_encode_supported"] = false;
		caps["max_width"] = 0;
		caps["max_height"] = 0;
		caps["max_dpb_slots"] = 0;
		caps["max_level"] = 0;
		caps["supported_profiles"] = Array();
		return caps;
	}
	
	// TODO: Query actual hardware capabilities from RenderingDevice
	// This would call Vulkan Video capability queries
	caps["hardware_decode_supported"] = false; // Until implemented
	caps["hardware_encode_supported"] = false;
	caps["max_width"] = 3840;
	caps["max_height"] = 2160;
	caps["max_dpb_slots"] = 8;
	caps["max_level"] = 31; // Level 6.3
	
	Array profiles;
	profiles.push_back(0); // Main profile
	caps["supported_profiles"] = profiles;
	
	return caps;
}

// ResourceFormatLoaderAV1 implementation

Ref<Resource> ResourceFormatLoaderAV1::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	Ref<VideoStreamAV1> stream;
	stream.instantiate();
	
	if (stream->load_file(p_path)) {
		if (r_error) {
			*r_error = OK;
		}
		return stream;
	}
	
	if (r_error) {
		*r_error = ERR_FILE_CANT_OPEN;
	}
	return Ref<Resource>();
}

void ResourceFormatLoaderAV1::get_recognized_extensions(List<String> *p_extensions) const {
	// AV1 in MKV/WebM containers only
	p_extensions->push_back("mkv");
	p_extensions->push_back("webm");
}

bool ResourceFormatLoaderAV1::handles_type(const String &p_type) const {
	return p_type == "VideoStreamAV1";
}

String ResourceFormatLoaderAV1::get_resource_type(const String &p_path) const {
	String extension = p_path.get_extension().to_lower();
	if (extension == "mkv" || extension == "webm") {
		// TODO: Check if file actually contains AV1 video
		return "VideoStreamAV1";
	}
	return "";
}
