/**************************************************************************/
/*  wmf_container_decoder.cpp                                             */
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

#include "wmf_container_decoder.h"

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

#define CHECK_HR(func)                                                           \
	if (SUCCEEDED(hr)) {                                                         \
		hr = (func);                                                             \
		if (FAILED(hr)) {                                                        \
			print_line("WMF Container function failed, HRESULT: " + itos(hr)); \
		}                                                                        \
	}

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

Error WMFContainerDecoder::open_file(const String &p_file) {
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
