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

#include <mfapi.h>
#include <mfidl.h>

// Undefine Windows macros that conflict with Godot
#ifdef CONNECT_DEFERRED
#undef CONNECT_DEFERRED
#endif
#ifdef CONNECT_ONESHOT
#undef CONNECT_ONESHOT
#endif

#define CHECK_HR(func)                                                           \
	if (SUCCEEDED(hr)) {                                                         \
		hr = (func);                                                             \
		if (FAILED(hr)) {                                                        \
			print_line("WMF function failed, HRESULT: " + itos(hr)); \
		}                                                                        \
	}

#define SafeRelease(p)      \
	{                       \
		if (p) {            \
			(p)->Release(); \
			(p) = nullptr;  \
		}                   \
	}

// Forward declarations
class WMFVideoDecoder;
class WMFAudioDecoder;
class WMFContainerDecoder;
class AudioVideoSynchronizer;
class AudioStreamWMF;
class SampleGrabberCallback;

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

struct VideoFrame {
	Vector<uint8_t> data;
	int64_t timestamp = 0;
	int width = 0;
	int height = 0;
	
	VideoFrame() {
		data.clear();
		timestamp = 0;
		width = 0;
		height = 0;
	}
};

struct WMFAudioSample {
	Vector<float> data;
	int64_t timestamp = 0;
	int channels = 0;
	int sample_rate = 0;
	
	WMFAudioSample() {
		data.clear();
		timestamp = 0;
		channels = 0;
		sample_rate = 0;
	}
};

// WMF Container Decoder Class
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
	Error load_file(const String &p_file);
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

// WMF Video Decoder Class
class WMFVideoDecoder {
private:
	WMFContainerDecoder *container_decoder = nullptr;
	int video_stream_index = -1;
	
	// Video properties
	int video_width = 0;
	int video_height = 0;
	float video_fps = 0.0f;
	
	// WMF objects
	IMFMediaSession *media_session = nullptr;
	IMFTopology *topology = nullptr;
	IMFPresentationClock *presentation_clock = nullptr;
	SampleGrabberCallback *sample_grabber_callback = nullptr;
	
	// Frame buffer
	Vector<VideoFrame> frame_buffer;
	int read_frame_idx = 0;
	int write_frame_idx = 0;
	mutable Mutex video_mutex;
	
	// State
	bool is_initialized = false;
	bool is_playing = false;
	bool is_paused = false;
	
	HRESULT create_video_topology();
	HRESULT create_video_topology_internal(IMFMediaSource *media_source);
	HRESULT create_sample_grabber_sink(IMFActivate **sink_activate);
	void cleanup();

public:
	WMFVideoDecoder();
	~WMFVideoDecoder();
	
	// Initialization
	Error initialize(WMFContainerDecoder *p_container, int p_video_stream_index);
	void shutdown();
	
	// Playback control
	Error start_decoding();
	Error stop_decoding();
	Error pause_decoding(bool p_paused);
	Error seek_to_time(double time_seconds);
	
	// Frame access
	bool has_frame_available() const;
	VideoFrame get_next_frame();
	VideoFrame *get_next_writable_frame();
	void write_frame_done();
	
	// Properties
	int get_width() const { return video_width; }
	int get_height() const { return video_height; }
	float get_fps() const { return video_fps; }
	bool is_valid() const { return is_initialized; }
	
	// Utility
	double get_frame_time(int64_t timestamp) const;
};

// WMF Audio Decoder Class
class WMFAudioDecoder {
private:
	WMFContainerDecoder *container_decoder = nullptr;
	int audio_stream_index = -1;
	
	// Audio properties
	int audio_channels = 0;
	int audio_sample_rate = 0;
	int audio_bits_per_sample = 0;
	
	// WMF objects
	IMFMediaSession *media_session = nullptr;
	IMFTopology *topology = nullptr;
	IMFPresentationClock *presentation_clock = nullptr;
	
	// Audio buffer
	Vector<WMFAudioSample> sample_buffer;
	int read_sample_idx = 0;
	int write_sample_idx = 0;
	mutable Mutex audio_mutex;
	
	// State
	bool is_initialized = false;
	bool is_playing = false;
	bool is_paused = false;
	
	HRESULT create_audio_topology();
	void cleanup();

public:
	WMFAudioDecoder();
	~WMFAudioDecoder();
	
	// Initialization
	Error initialize(WMFContainerDecoder *p_container, int p_audio_stream_index);
	void shutdown();
	
	// Playback control
	Error start_decoding();
	Error stop_decoding();
	Error pause_decoding(bool p_paused);
	Error seek_to_time(double time_seconds);
	
	// Sample access
	bool has_sample_available() const;
	WMFAudioSample get_next_sample();
	WMFAudioSample *get_next_writable_sample();
	void write_sample_done();
	
	// Properties
	int get_channels() const { return audio_channels; }
	int get_sample_rate() const { return audio_sample_rate; }
	int get_bits_per_sample() const { return audio_bits_per_sample; }
	bool is_valid() const { return is_initialized; }
};

// Main Video Stream Class
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
