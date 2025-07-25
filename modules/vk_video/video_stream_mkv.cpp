/**************************************************************************/
/*  video_stream_mkv.cpp                                                  */
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

#include "video_stream_mkv.h"

#include "core/io/file_access.h"
#include "video_stream_av1.h"

void VideoStreamMKV::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_file", "file"), &VideoStreamMKV::set_file);
	ClassDB::bind_method(D_METHOD("get_file"), &VideoStreamMKV::get_file);
	ClassDB::bind_method(D_METHOD("set_data", "data"), &VideoStreamMKV::set_data);
	ClassDB::bind_method(D_METHOD("get_data"), &VideoStreamMKV::get_data);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "file", PROPERTY_HINT_FILE, "*.mkv,*.webm"), "set_file", "get_file");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_BYTE_ARRAY, "data"), "set_data", "get_data");
}

VideoStreamMKV::VideoStreamMKV() {
}

VideoStreamMKV::~VideoStreamMKV() {
}

Ref<VideoStreamPlayback> VideoStreamMKV::instantiate_playback() {
	Ref<VideoStreamPlaybackAV1> playback;
	playback.instantiate();
	
	if (!file_path.is_empty()) {
		playback->open_file(file_path);
	}
	// Note: data-based playback not yet implemented in VideoStreamPlaybackAV1
	
	return playback;
}

void VideoStreamMKV::set_file(const String &p_file) {
	file_path = p_file;
	data.clear();
}

String VideoStreamMKV::get_file() const {
	return file_path;
}

void VideoStreamMKV::set_data(const PackedByteArray &p_data) {
	data = p_data;
	file_path = "";
}

PackedByteArray VideoStreamMKV::get_data() const {
	return data;
}

// ResourceFormatLoaderMKV implementation

Ref<Resource> ResourceFormatLoaderMKV::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	Ref<VideoStreamMKV> stream;
	stream.instantiate();
	stream->set_file(p_path);
	
	if (r_error) {
		*r_error = OK;
	}
	
	return stream;
}

void ResourceFormatLoaderMKV::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("mkv");
	p_extensions->push_back("webm");
}

bool ResourceFormatLoaderMKV::handles_type(const String &p_type) const {
	return p_type == "VideoStreamMKV" || p_type == "VideoStream";
}

String ResourceFormatLoaderMKV::get_resource_type(const String &p_path) const {
	String extension = p_path.get_extension().to_lower();
	if (extension == "mkv" || extension == "webm") {
		return "VideoStreamMKV";
	}
	return "";
}
