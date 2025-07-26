/**************************************************************************/
/*  wmf_container_decoder.h                                               */
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

#include "core/error/error_list.h"
#include "core/os/mutex.h"
#include "core/string/ustring.h"
#include "core/templates/vector.h"

#include <mfapi.h>
#include <mfidl.h>

// Undefine Windows macros that conflict with Godot
#ifdef CONNECT_DEFERRED
#undef CONNECT_DEFERRED
#endif
#ifdef CONNECT_ONESHOT
#undef CONNECT_ONESHOT
#endif

struct IMFMediaSource;
struct IMFPresentationDescriptor;
struct IMFStreamDescriptor;

struct MediaPacket {
	int64_t timestamp = 0;
	int64_t duration = 0;
	Vector<uint8_t> data;
	bool is_keyframe = false;
};

struct StreamInfo {
	DWORD stream_index = 0;
	GUID major_type;
	GUID sub_type;
	
	// Video specific
	int width = 0;
	int height = 0;
	float fps = 0.0f;
	
	// Audio specific
	int sample_rate = 0;
	int channels = 0;
	int bits_per_sample = 0;
	
	// Common
	float duration = 0.0f;
	bool selected = false;
};

class WMFContainerDecoder {
private:
	IMFMediaSource *media_source = nullptr;
	IMFPresentationDescriptor *presentation_descriptor = nullptr;
	
	Vector<StreamInfo> stream_infos;
	Mutex container_mutex;
	
	bool is_initialized = false;
	String file_path;
	
	HRESULT create_media_source(const String &p_file);
	HRESULT analyze_streams();
	void cleanup();

public:
	WMFContainerDecoder();
	~WMFContainerDecoder();
	
	// Container operations
	Error open_file(const String &p_file);
	void close();
	
	// Stream information
	int get_stream_count() const;
	StreamInfo get_stream_info(int stream_index) const;
	Vector<int> get_video_streams() const;
	Vector<int> get_audio_streams() const;
	
	// Stream selection
	Error select_stream(int stream_index, bool selected = true);
	bool is_stream_selected(int stream_index) const;
	
	// Container-level seeking
	Error seek_to_time(double time_seconds);
	double get_duration() const;
	
	// Raw data access for decoders
	IMFMediaSource* get_media_source() const { return media_source; }
	IMFPresentationDescriptor* get_presentation_descriptor() const { return presentation_descriptor; }
	IMFStreamDescriptor* get_stream_descriptor(int stream_index) const;
	
	// Utility
	bool is_valid() const { return is_initialized && media_source != nullptr; }
	String get_file_path() const { return file_path; }
};
