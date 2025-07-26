/**************************************************************************/
/*  video_stream_wmf.h                                                    */
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
#include "core/os/mutex.h"
#include "scene/resources/video_stream.h"
#include "wmf_container_decoder.h"

class WMFVideoDecoder;
class WMFAudioDecoder;
class VideoStreamPlaybackWMF;
class AudioStreamWMF;

class VideoStreamWMF : public VideoStream {
	GDCLASS(VideoStreamWMF, VideoStream);

private:
	WMFContainerDecoder *container_decoder = nullptr;
	WMFVideoDecoder *video_decoder = nullptr;
	WMFAudioDecoder *audio_decoder = nullptr;
	Ref<AudioStreamWMF> audio_stream;
	
	// Container information
	Vector<int> video_streams;
	Vector<int> audio_streams;
	int selected_video_stream = -1;
	int selected_audio_stream = -1;
	
	bool is_initialized = false;
	Mutex container_mutex;
	String file_path;
	
	void cleanup();
	Error initialize_container();

protected:
	static void _bind_methods();

public:
	// VideoStream interface
	virtual Ref<VideoStreamPlayback> instantiate_playback() override;
	
	// File operations
	void set_file(const String &p_file);
	String get_file() const { return file_path; }
	
	// Stream information
	int get_video_stream_count() const;
	int get_audio_stream_count() const;
	Vector<int> get_video_streams() const { return video_streams; }
	Vector<int> get_audio_streams() const { return audio_streams; }
	
	// Stream selection
	void set_video_stream(int p_stream_index);
	void set_audio_stream(int p_stream_index);
	int get_selected_video_stream() const { return selected_video_stream; }
	int get_selected_audio_stream() const { return selected_audio_stream; }
	
	// Decoder access (for playback classes)
	WMFVideoDecoder* get_video_decoder() const { return video_decoder; }
	WMFAudioDecoder* get_audio_decoder() const { return audio_decoder; }
	WMFContainerDecoder* get_container_decoder() const { return container_decoder; }
	Ref<AudioStreamWMF> get_audio_stream() const { return audio_stream; }
	
	// Container operations
	Error seek_container(double time_seconds);
	double get_duration() const;
	bool is_container_valid() const { return is_initialized && container_decoder && container_decoder->is_valid(); }

	VideoStreamWMF();
	~VideoStreamWMF();
};

class ResourceFormatLoaderWMF : public ResourceFormatLoader {
public:
	virtual Ref<Resource> load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_use_sub_threads = false, float *r_progress = nullptr, CacheMode p_cache_mode = CACHE_MODE_REUSE) override;
	virtual void get_recognized_extensions(List<String> *p_extensions) const override;
	virtual bool handles_type(const String &p_type) const override;
	virtual String get_resource_type(const String &p_path) const override;
};
