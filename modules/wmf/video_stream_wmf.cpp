/**************************************************************************/
/*  video_stream_wmf.cpp                                                  */
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

#include "video_stream_wmf.h"

#include "core/config/project_settings.h"
#include "core/error/error_list.h"
#include "core/error/error_macros.h"
#include "core/io/file_access.h"
#include "core/string/print_string.h"
#include "video_stream_playback_wmf.h"
#include "audio_stream_playback_wmf.h"
#include "wmf_video_decoder.h"
#include "wmf_audio_decoder.h"
#include "wmf_container_decoder.h"

VideoStreamWMF::VideoStreamWMF() {
	container_decoder = memnew(WMFContainerDecoder);
}

VideoStreamWMF::~VideoStreamWMF() {
	cleanup();
}

void VideoStreamWMF::cleanup() {
	MutexLock lock(container_mutex);
	
	if (video_decoder) {
		video_decoder->shutdown();
		memdelete(video_decoder);
		video_decoder = nullptr;
	}
	
	if (audio_decoder) {
		audio_decoder->shutdown();
		memdelete(audio_decoder);
		audio_decoder = nullptr;
	}
	
	if (container_decoder) {
		container_decoder->shutdown();
		memdelete(container_decoder);
		container_decoder = nullptr;
	}
	
	video_streams.clear();
	audio_streams.clear();
	selected_video_stream = -1;
	selected_audio_stream = -1;
	is_initialized = false;
}

Error VideoStreamWMF::initialize_container() {
	if (!container_decoder) {
		return ERR_UNAVAILABLE;
	}
	
	Error err = container_decoder->initialize();
	if (err != OK) {
		print_line("VideoStreamWMF: Failed to initialize container decoder");
		return err;
	}
	
	// Get stream information
	int stream_count = container_decoder->get_stream_count();
	for (int i = 0; i < stream_count; i++) {
		WMFContainerDecoder::StreamInfo info = container_decoder->get_stream_info(i);
		
		if (info.type == WMFContainerDecoder::STREAM_TYPE_VIDEO) {
			video_streams.push_back(i);
			if (selected_video_stream == -1) {
				selected_video_stream = i; // Select first video stream by default
			}
		} else if (info.type == WMFContainerDecoder::STREAM_TYPE_AUDIO) {
			audio_streams.push_back(i);
			if (selected_audio_stream == -1) {
				selected_audio_stream = i; // Select first audio stream by default
			}
		}
	}
	
	// Create video decoder if we have a video stream
	if (selected_video_stream >= 0) {
		video_decoder = memnew(WMFVideoDecoder);
		err = video_decoder->initialize(container_decoder, selected_video_stream);
		if (err != OK) {
			print_line("VideoStreamWMF: Failed to initialize video decoder");
			memdelete(video_decoder);
			video_decoder = nullptr;
		}
	}
	
	// Create audio decoder if we have an audio stream
	if (selected_audio_stream >= 0) {
		audio_decoder = memnew(WMFAudioDecoder);
		err = audio_decoder->initialize(container_decoder, selected_audio_stream);
		if (err != OK) {
			print_line("VideoStreamWMF: Failed to initialize audio decoder");
			memdelete(audio_decoder);
			audio_decoder = nullptr;
		} else {
			// Create audio stream for this decoder
			audio_stream = memnew(AudioStreamWMF);
			audio_stream->set_audio_decoder(audio_decoder);
			audio_stream->set_length(container_decoder->get_duration());
		}
	}
	
	is_initialized = true;
	return OK;
}

Ref<VideoStreamPlayback> VideoStreamWMF::instantiate_playback() {
	MutexLock lock(container_mutex);
	
	if (!is_initialized) {
		print_line("VideoStreamWMF: Container not initialized");
		return Ref<VideoStreamPlayback>();
	}
	
	Ref<VideoStreamPlaybackWMF> playback = memnew(VideoStreamPlaybackWMF);
	
	// Set video decoder if available
	if (video_decoder) {
		playback->set_video_decoder(video_decoder);
	}
	
	// Set audio decoder if available
	if (audio_decoder) {
		playback->set_audio_decoder(audio_decoder);
	}
	
	// Get the shared synchronizer from video playback and set it up for audio/video sync
	Ref<AudioVideoSynchronizer> synchronizer = playback->get_synchronizer();
	if (synchronizer.is_valid()) {
		// Configure synchronizer for audio-master sync (audio provides timing)
		synchronizer->set_sync_mode(AudioVideoSynchronizer::SYNC_MODE_AUDIO_MASTER);
		synchronizer->set_max_queue_size(6);
		synchronizer->set_use_timing_filter(true);
		
		// Share the synchronizer with the audio stream
		if (audio_stream.is_valid()) {
			audio_stream->set_shared_synchronizer(synchronizer);
		}
		
		print_line("VideoStreamWMF: Configured synchronizer for audio/video sync");
	}
	
	return playback;
}

int VideoStreamWMF::get_video_stream_count() const {
	return video_streams.size();
}

int VideoStreamWMF::get_audio_stream_count() const {
	return audio_streams.size();
}

void VideoStreamWMF::set_video_stream(int p_stream_index) {
	MutexLock lock(container_mutex);
	
	if (p_stream_index < 0 || p_stream_index >= video_streams.size()) {
		return;
	}
	
	int actual_stream_index = video_streams[p_stream_index];
	if (actual_stream_index == selected_video_stream) {
		return; // Already selected
	}
	
	selected_video_stream = actual_stream_index;
	
	// Recreate video decoder with new stream
	if (video_decoder) {
		video_decoder->shutdown();
		memdelete(video_decoder);
	}
	
	video_decoder = memnew(WMFVideoDecoder);
	Error err = video_decoder->initialize(container_decoder, selected_video_stream);
	if (err != OK) {
		print_line("VideoStreamWMF: Failed to initialize video decoder for stream " + itos(p_stream_index));
		memdelete(video_decoder);
		video_decoder = nullptr;
	}
}

void VideoStreamWMF::set_audio_stream(int p_stream_index) {
	MutexLock lock(container_mutex);
	
	if (p_stream_index < 0 || p_stream_index >= audio_streams.size()) {
		return;
	}
	
	int actual_stream_index = audio_streams[p_stream_index];
	if (actual_stream_index == selected_audio_stream) {
		return; // Already selected
	}
	
	selected_audio_stream = actual_stream_index;
	
	// Recreate audio decoder with new stream
	if (audio_decoder) {
		audio_decoder->shutdown();
		memdelete(audio_decoder);
	}
	
	audio_decoder = memnew(WMFAudioDecoder);
	Error err = audio_decoder->initialize(container_decoder, selected_audio_stream);
	if (err != OK) {
		print_line("VideoStreamWMF: Failed to initialize audio decoder for stream " + itos(p_stream_index));
		memdelete(audio_decoder);
		audio_decoder = nullptr;
	}
}

Error VideoStreamWMF::seek_container(double time_seconds) {
	MutexLock lock(container_mutex);
	
	if (!container_decoder || !is_initialized) {
		return ERR_UNAVAILABLE;
	}
	
	Error err = container_decoder->seek_to_time(time_seconds);
	if (err != OK) {
		return err;
	}
	
	// Reset decoders after seek
	if (video_decoder) {
		video_decoder->seek_to_time(time_seconds);
	}
	
	if (audio_decoder) {
		audio_decoder->seek_to_time(time_seconds);
	}
	
	return OK;
}

double VideoStreamWMF::get_duration() const {
	if (!container_decoder || !is_initialized) {
		return 0.0;
	}
	
	return container_decoder->get_duration();
}

void VideoStreamWMF::set_file(const String &p_file) {
	cleanup();
	
	if (p_file.is_empty()) {
		file_path = "";
		return;
	}
	
	// Store the original file path
	file_path = p_file;
	
	// Handle Godot resource paths
	String actual_file_path = p_file;
	if (p_file.begins_with("res://")) {
		actual_file_path = ProjectSettings::get_singleton()->globalize_path(p_file);
	}
	
	// Verify the file exists
	Error e;
	Ref<FileAccess> fa = FileAccess::open(actual_file_path, FileAccess::READ, &e);
	if (e != OK) {
		print_line("VideoStreamWMF: Failed to open file: " + actual_file_path + " Error: " + itos(e));
		file_path = "";
		return;
	}
	fa.unref();
	
	// Initialize container with the file
	container_decoder = memnew(WMFContainerDecoder);
	Error err = container_decoder->load_file(actual_file_path);
	if (err != OK) {
		print_line("VideoStreamWMF: Failed to load file into container: " + actual_file_path);
		cleanup();
		file_path = "";
		return;
	}
	
	// Initialize the container and create decoders
	err = initialize_container();
	if (err != OK) {
		print_line("VideoStreamWMF: Failed to initialize container for file: " + actual_file_path);
		cleanup();
		file_path = "";
		return;
	}
	
	print_line("VideoStreamWMF: Successfully loaded file: " + actual_file_path);
	print_line("VideoStreamWMF: Video streams: " + itos(video_streams.size()) + ", Audio streams: " + itos(audio_streams.size()));
}

void VideoStreamWMF::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_file", "file"), &VideoStreamWMF::set_file);
	ClassDB::bind_method(D_METHOD("get_file"), &VideoStreamWMF::get_file);
	
	ClassDB::bind_method(D_METHOD("get_video_stream_count"), &VideoStreamWMF::get_video_stream_count);
	ClassDB::bind_method(D_METHOD("get_audio_stream_count"), &VideoStreamWMF::get_audio_stream_count);
	
	ClassDB::bind_method(D_METHOD("set_video_stream", "stream_index"), &VideoStreamWMF::set_video_stream);
	ClassDB::bind_method(D_METHOD("set_audio_stream", "stream_index"), &VideoStreamWMF::set_audio_stream);
	ClassDB::bind_method(D_METHOD("get_selected_video_stream"), &VideoStreamWMF::get_selected_video_stream);
	ClassDB::bind_method(D_METHOD("get_selected_audio_stream"), &VideoStreamWMF::get_selected_audio_stream);
	
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "file", PROPERTY_HINT_FILE, "*.mp4,*.avi,*.wmv,*.mov,*.mkv,*.webm,*.flv"), "set_file", "get_file");
}

// ResourceFormatLoaderWMF Implementation

Ref<Resource> ResourceFormatLoaderWMF::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
	if (f.is_null()) {
		if (r_error) {
			*r_error = ERR_CANT_OPEN;
		}
		return Ref<Resource>();
	}
	f.unref();

	VideoStreamWMF *stream = memnew(VideoStreamWMF);
	stream->set_file(p_path);

	// Check if the file was loaded successfully
	if (!stream->is_container_valid()) {
		memdelete(stream);
		if (r_error) {
			*r_error = ERR_FILE_CORRUPT;
		}
		return Ref<Resource>();
	}

	Ref<VideoStreamWMF> wmf_stream = Ref<VideoStreamWMF>(stream);

	if (r_error) {
		*r_error = OK;
	}

	return wmf_stream;
}

void ResourceFormatLoaderWMF::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("mp4");
	p_extensions->push_back("avi");
	p_extensions->push_back("wmv");
	p_extensions->push_back("mov");
	p_extensions->push_back("mkv");
	p_extensions->push_back("webm");
	p_extensions->push_back("flv");
}

bool ResourceFormatLoaderWMF::handles_type(const String &p_type) const {
	return ClassDB::is_parent_class(p_type, "VideoStream");
}

String ResourceFormatLoaderWMF::get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el == "mp4" || el == "avi" || el == "wmv" || el == "mov" || el == "mkv" || el == "webm" || el == "flv") {
		return "VideoStreamWMF";
	}
	return "";
}
