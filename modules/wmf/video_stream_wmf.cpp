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
	
	// Get stream information
	int stream_count = container_decoder->get_stream_count();
	for (int i = 0; i < stream_count; i++) {
		StreamInfo info = container_decoder->get_stream_info(i);
		
		if (info.major_type == MFMediaType_Video) {
			video_streams.push_back(i);
			if (selected_video_stream == -1) {
				selected_video_stream = i; // Select first video stream by default
			}
		} else if (info.major_type == MFMediaType_Audio) {
			audio_streams.push_back(i);
			if (selected_audio_stream == -1) {
				selected_audio_stream = i; // Select first audio stream by default
			}
		}
	}
	
	// Create video decoder if we have a video stream
	if (selected_video_stream >= 0) {
		video_decoder = memnew(WMFVideoDecoder);
		Error err = video_decoder->initialize(container_decoder, selected_video_stream);
		if (err != OK) {
			print_line("VideoStreamWMF: Failed to initialize video decoder");
			memdelete(video_decoder);
			video_decoder = nullptr;
		}
	}
	
	// Create audio decoder if we have an audio stream
	if (selected_audio_stream >= 0) {
		audio_decoder = memnew(WMFAudioDecoder);
		Error err = audio_decoder->initialize(container_decoder, selected_audio_stream);
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
			audio_stream->set_synchronizer(synchronizer);
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

#include <mfapi.h>
#include <mferror.h>
#include <shlwapi.h>

// Undefine Windows macros that conflict with Godot
#ifdef CONNECT_DEFERRED
#undef CONNECT_DEFERRED
#endif
#ifdef CONNECT_ONESHOT
#undef CONNECT_ONESHOT
#endif

#include "core/config/project_settings.h"
#include "core/error/error_macros.h"
#include "core/io/file_access.h"
#include "core/string/print_string.h"


#define SafeRelease(p)      \
	{                       \
		if (p) {            \
			(p)->Release(); \
			(p) = nullptr;  \
		}                   \
	}

WMFContainerDecoder::WMFContainerDecoder() {
}

WMFContainerDecoder::~WMFContainerDecoder() {
	cleanup();
}

Error WMFContainerDecoder::load_file(const String &p_file) {
	MutexLock lock(container_mutex);
	
	cleanup();
	
	HRESULT hr = create_media_source(p_file);
	if (FAILED(hr)) {
		return ERR_CANT_OPEN;
	}
	
	hr = analyze_streams();
	if (FAILED(hr)) {
		cleanup();
		return ERR_CANT_OPEN;
	}
	
	file_path = p_file;
	is_initialized = true;
	
	print_line("WMF Container: Successfully opened " + p_file);
	print_line("WMF Container: Found " + itos(stream_infos.size()) + " streams");
	
	return OK;
}

void WMFContainerDecoder::close() {
	MutexLock lock(container_mutex);
	cleanup();
}

HRESULT WMFContainerDecoder::create_media_source(const String &p_file) {
	ERR_FAIL_COND_V(p_file.is_empty(), E_FAIL);

	IMFSourceResolver *source_resolver = nullptr;
	IUnknown *source = nullptr;

	HRESULT hr = S_OK;
	CHECK_HR(MFCreateSourceResolver(&source_resolver));

	print_line("WMF Container: Opening file: " + p_file);

	file_path = p_file;

	// Handle Godot resource paths
	if (p_file.begins_with("res://")) {
		file_path = ProjectSettings::get_singleton()->globalize_path(p_file);
		print_line("WMF Container: Globalized path: " + file_path);
	}

	// Verify the file exists
	Error e;
	Ref<FileAccess> fa = FileAccess::open(file_path, FileAccess::READ, &e);
	if (e != OK) {
		print_line("WMF Container: Failed to open file: " + file_path + " Error: " + itos(e));
		SafeRelease(source_resolver);
		return E_FAIL;
	}
	fa.unref();

	// Convert to Windows path format
	file_path = file_path.replace("/", "\\");
	print_line("WMF Container: Final path: " + file_path);

	MF_OBJECT_TYPE object_type;
	CHECK_HR(source_resolver->CreateObjectFromURL((LPCWSTR)file_path.utf16().ptrw(),
			MF_RESOLUTION_MEDIASOURCE, nullptr, &object_type, &source));
	CHECK_HR(source->QueryInterface(IID_PPV_ARGS(&media_source)));

	SafeRelease(source_resolver);
	SafeRelease(source);

	return hr;
}

HRESULT WMFContainerDecoder::analyze_streams() {
	ERR_FAIL_COND_V(!media_source, E_FAIL);

	HRESULT hr = S_OK;
	CHECK_HR(media_source->CreatePresentationDescriptor(&presentation_descriptor));

	DWORD stream_count = 0;
	CHECK_HR(presentation_descriptor->GetStreamDescriptorCount(&stream_count));

	stream_infos.clear();
	stream_infos.resize(stream_count);

	for (DWORD i = 0; i < stream_count; i++) {
		BOOL selected = FALSE;
		IMFStreamDescriptor *stream_descriptor = nullptr;
		IMFMediaTypeHandler *type_handler = nullptr;
		IMFMediaType *media_type = nullptr;

		CHECK_HR(presentation_descriptor->GetStreamDescriptorByIndex(i, &selected, &stream_descriptor));
		CHECK_HR(stream_descriptor->GetMediaTypeHandler(&type_handler));
		CHECK_HR(type_handler->GetMediaTypeByIndex(0, &media_type));

		StreamInfo &info = stream_infos.ptrw()[i];
		info.stream_index = i;
		info.selected = selected;

		CHECK_HR(type_handler->GetMajorType(&info.major_type));
		CHECK_HR(media_type->GetGUID(MF_MT_SUBTYPE, &info.sub_type));

		// Get duration
		UINT64 duration;
		if (SUCCEEDED(presentation_descriptor->GetUINT64(MF_PD_DURATION, &duration))) {
			info.duration = duration / 10000000.0f; // Convert from 100ns units to seconds
		}

		if (info.major_type == MFMediaType_Video) {
			// Video stream
			UINT32 width, height;
			if (SUCCEEDED(MFGetAttributeSize(media_type, MF_MT_FRAME_SIZE, &width, &height))) {
				info.width = width;
				info.height = height;
			}

			UINT32 numerator, denominator;
			if (SUCCEEDED(MFGetAttributeRatio(media_type, MF_MT_FRAME_RATE, &numerator, &denominator))) {
				if (denominator > 0) {
					info.fps = (float)numerator / (float)denominator;
				}
			}

			print_line("WMF Container: Video stream " + itos(i) + ": " + itos(info.width) + "x" + itos(info.height) + " @ " + rtos(info.fps) + "fps");
		} else if (info.major_type == MFMediaType_Audio) {
			// Audio stream
			UINT32 sample_rate, channels, bits_per_sample;
			if (SUCCEEDED(media_type->GetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, &sample_rate))) {
				info.sample_rate = sample_rate;
			}
			if (SUCCEEDED(media_type->GetUINT32(MF_MT_AUDIO_NUM_CHANNELS, &channels))) {
				info.channels = channels;
			}
			if (SUCCEEDED(media_type->GetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, &bits_per_sample))) {
				info.bits_per_sample = bits_per_sample;
			}

			print_line("WMF Container: Audio stream " + itos(i) + ": " + itos(info.sample_rate) + "Hz, " + itos(info.channels) + " channels, " + itos(info.bits_per_sample) + " bits");
		}

		SafeRelease(media_type);
		SafeRelease(type_handler);
		SafeRelease(stream_descriptor);
	}

	return hr;
}

void WMFContainerDecoder::cleanup() {
	if (media_source) {
		media_source->Shutdown();
	}
	
	SafeRelease(presentation_descriptor);
	SafeRelease(media_source);
	
	stream_infos.clear();
	is_initialized = false;
	file_path = "";
}

int WMFContainerDecoder::get_stream_count() const {
	return stream_infos.size();
}

StreamInfo WMFContainerDecoder::get_stream_info(int stream_index) const {
	ERR_FAIL_INDEX_V(stream_index, stream_infos.size(), StreamInfo());
	return stream_infos[stream_index];
}

Vector<int> WMFContainerDecoder::get_video_streams() const {
	Vector<int> video_streams;
	for (int i = 0; i < stream_infos.size(); i++) {
		if (stream_infos[i].major_type == MFMediaType_Video) {
			video_streams.push_back(i);
		}
	}
	return video_streams;
}

Vector<int> WMFContainerDecoder::get_audio_streams() const {
	Vector<int> audio_streams;
	for (int i = 0; i < stream_infos.size(); i++) {
		if (stream_infos[i].major_type == MFMediaType_Audio) {
			audio_streams.push_back(i);
		}
	}
	return audio_streams;
}

Error WMFContainerDecoder::select_stream(int stream_index, bool selected) {
	ERR_FAIL_INDEX_V(stream_index, stream_infos.size(), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(!presentation_descriptor, ERR_UNCONFIGURED);

	MutexLock lock(container_mutex);

	HRESULT hr = S_OK;
	if (selected) {
		CHECK_HR(presentation_descriptor->SelectStream(stream_index));
	} else {
		CHECK_HR(presentation_descriptor->DeselectStream(stream_index));
	}

	if (SUCCEEDED(hr)) {
		stream_infos.ptrw()[stream_index].selected = selected;
		print_line("WMF Container: Stream " + itos(stream_index) + (selected ? " selected" : " deselected"));
		return OK;
	}

	return ERR_CANT_ACQUIRE_RESOURCE;
}

bool WMFContainerDecoder::is_stream_selected(int stream_index) const {
	ERR_FAIL_INDEX_V(stream_index, stream_infos.size(), false);
	return stream_infos[stream_index].selected;
}

Error WMFContainerDecoder::seek_to_time(double time_seconds) {
	ERR_FAIL_COND_V(!media_source, ERR_UNCONFIGURED);

	MutexLock lock(container_mutex);

	// Convert seconds to 100-nanosecond units
	MFTIME seek_time = (MFTIME)(time_seconds * 10000000.0);

	PROPVARIANT var;
	PropVariantInit(&var);
	var.vt = VT_I8;
	var.hVal.QuadPart = seek_time;

	HRESULT hr = media_source->Start(presentation_descriptor, &GUID_NULL, &var);
	PropVariantClear(&var);

	if (SUCCEEDED(hr)) {
		print_line("WMF Container: Seeked to " + rtos(time_seconds) + " seconds");
		return OK;
	}

	print_line("WMF Container: Seek failed with HRESULT: " + itos(hr));
	return ERR_CANT_ACQUIRE_RESOURCE;
}

double WMFContainerDecoder::get_duration() const {
	if (stream_infos.size() > 0) {
		return stream_infos[0].duration;
	}
	return 0.0;
}

IMFStreamDescriptor* WMFContainerDecoder::get_stream_descriptor(int stream_index) const {
	ERR_FAIL_INDEX_V(stream_index, stream_infos.size(), nullptr);
	ERR_FAIL_COND_V(!presentation_descriptor, nullptr);

	BOOL selected = FALSE;
	IMFStreamDescriptor *stream_descriptor = nullptr;
	
	HRESULT hr = presentation_descriptor->GetStreamDescriptorByIndex(stream_index, &selected, &stream_descriptor);
	if (SUCCEEDED(hr)) {
		return stream_descriptor; // Caller must release
	}
	
	return nullptr;
}
