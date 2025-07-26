/**************************************************************************/
/*  wmf_video_decoder.cpp                                                 */
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

#include "wmf_video_decoder.h"

#include "sample_grabber_callback.h"
#include "core/error/error_macros.h"
#include "core/string/print_string.h"
#include <mfapi.h>
#include <mferror.h>

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
			print_line("WMF Video Decoder function failed, HRESULT: " + itos(hr)); \
		}                                                                        \
	}

#define SafeRelease(p)      \
	{                       \
		if (p) {            \
			(p)->Release(); \
			(p) = nullptr;  \
		}                   \
	}

WMFVideoDecoder::WMFVideoDecoder() {
	frame_buffer.resize(24); // Buffer for 24 video frames
}

WMFVideoDecoder::~WMFVideoDecoder() {
	shutdown();
}

Error WMFVideoDecoder::initialize(WMFContainerDecoder *p_container, int p_video_stream_index) {
	ERR_FAIL_COND_V(!p_container, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(!p_container->is_valid(), ERR_UNCONFIGURED);
	
	container_decoder = p_container;
	video_stream_index = p_video_stream_index;
	
	// Get video stream information
	StreamInfo stream_info = container_decoder->get_stream_info(video_stream_index);
	video_width = stream_info.width;
	video_height = stream_info.height;
	video_fps = stream_info.fps;
	
	print_line("WMF Video Decoder: Initializing for stream " + itos(video_stream_index));
	print_line("WMF Video Decoder: " + itos(video_width) + "x" + itos(video_height) + " @ " + rtos(video_fps) + " FPS");
	
	HRESULT hr = create_video_topology();
	if (FAILED(hr)) {
		cleanup();
		return ERR_CANT_CREATE;
	}
	
	is_initialized = true;
	return OK;
}

void WMFVideoDecoder::shutdown() {
	MutexLock lock(video_mutex);
	
	if (is_playing) {
		stop_decoding();
	}
	
	cleanup();
	
	container_decoder = nullptr;
	video_stream_index = -1;
	is_initialized = false;
}

HRESULT WMFVideoDecoder::create_video_topology() {
	HRESULT hr = S_OK;
	
	// Get the media source from container decoder
	IMFMediaSource *media_source = container_decoder->get_media_source();
	if (!media_source) {
		print_line("WMFVideoDecoder: No media source available");
		return E_FAIL;
	}
	
	// Create media session for video decoding
	CHECK_HR(MFCreateMediaSession(nullptr, &media_session));
	
	// Create sample grabber callback for video frames
	sample_grabber_callback = new SampleGrabberCallback(nullptr, video_mutex);
	if (!sample_grabber_callback) {
		print_line("WMFVideoDecoder: Failed to create sample grabber callback");
		return E_FAIL;
	}
	
	// Create topology for video stream
	CHECK_HR(create_video_topology_internal(media_source));
	
	// Set topology on media session
	CHECK_HR(media_session->SetTopology(0, topology));
	
	// Get presentation clock
	IMFClock *clock;
	if (SUCCEEDED(media_session->GetClock(&clock))) {
		CHECK_HR(clock->QueryInterface(IID_PPV_ARGS(&presentation_clock)));
		SafeRelease(clock);
	}
	
	// Set frame size for sample grabber
	sample_grabber_callback->set_frame_size(video_width, video_height);
	
	print_line("WMF Video Decoder: Video topology created successfully");
	
	return hr;
}

void WMFVideoDecoder::cleanup() {
	SafeRelease(presentation_clock);
	SafeRelease(topology);
	SafeRelease(media_session);
	
	if (sample_grabber_callback) {
		sample_grabber_callback->Release();
		sample_grabber_callback = nullptr;
	}
	
	frame_buffer.clear();
	frame_buffer.resize(24);
	read_frame_idx = 0;
	write_frame_idx = 0;
	
	is_playing = false;
	is_paused = false;
}

Error WMFVideoDecoder::start_decoding() {
	if (!is_initialized) {
		return ERR_UNCONFIGURED;
	}
	
	if (is_playing) {
		return OK; // Already playing
	}
	
	// TODO: Start video decoding session
	print_line("WMF Video Decoder: Starting video decoding");
	
	is_playing = true;
	is_paused = false;
	
	return OK;
}

Error WMFVideoDecoder::stop_decoding() {
	if (!is_playing) {
		return OK; // Already stopped
	}
	
	// TODO: Stop video decoding session
	print_line("WMF Video Decoder: Stopping video decoding");
	
	is_playing = false;
	is_paused = false;
	
	return OK;
}

Error WMFVideoDecoder::pause_decoding(bool p_paused) {
	if (!is_initialized) {
		return ERR_UNCONFIGURED;
	}
	
	// TODO: Pause/resume video decoding session
	print_line("WMF Video Decoder: " + String(p_paused ? "Pausing" : "Resuming") + " video decoding");
	
	is_paused = p_paused;
	
	return OK;
}

Error WMFVideoDecoder::seek_to_time(double time_seconds) {
	if (!is_initialized) {
		return ERR_UNCONFIGURED;
	}
	
	// TODO: Seek video decoder to specific time
	print_line("WMF Video Decoder: Seeking to " + rtos(time_seconds) + " seconds");
	
	// Clear video buffer after seek
	MutexLock lock(video_mutex);
	read_frame_idx = write_frame_idx = 0;
	
	return OK;
}

bool WMFVideoDecoder::has_frame_available() const {
	MutexLock lock(video_mutex);
	return read_frame_idx != write_frame_idx;
}

VideoFrame WMFVideoDecoder::get_next_frame() {
	MutexLock lock(video_mutex);
	
	if (read_frame_idx == write_frame_idx) {
		// No frames available
		return VideoFrame();
	}
	
	VideoFrame frame = frame_buffer[read_frame_idx];
	read_frame_idx = (read_frame_idx + 1) % frame_buffer.size();
	
	return frame;
}

VideoFrame *WMFVideoDecoder::get_next_writable_frame() {
	return &frame_buffer.write[write_frame_idx];
}

void WMFVideoDecoder::write_frame_done() {
	MutexLock lock(video_mutex);
	int next_write_idx = (write_frame_idx + 1) % frame_buffer.size();
	
	// Handle buffer overflow by advancing read index if needed
	if (read_frame_idx == next_write_idx) {
		read_frame_idx = (read_frame_idx + 1) % frame_buffer.size();
	}
	
	write_frame_idx = next_write_idx;
}

double WMFVideoDecoder::get_frame_time(int64_t timestamp) const {
	// Convert WMF timestamp to seconds
	// WMF timestamps are in 100-nanosecond units
	return static_cast<double>(timestamp) / 10000000.0;
}

HRESULT WMFVideoDecoder::create_video_topology_internal(IMFMediaSource *media_source) {
	HRESULT hr = S_OK;
	
	// Create topology
	CHECK_HR(MFCreateTopology(&topology));
	
	// Get presentation descriptor
	IMFPresentationDescriptor *presentation_descriptor = nullptr;
	CHECK_HR(media_source->CreatePresentationDescriptor(&presentation_descriptor));
	
	// Get stream descriptor for our video stream
	IMFStreamDescriptor *stream_descriptor = nullptr;
	BOOL selected = FALSE;
	CHECK_HR(presentation_descriptor->GetStreamDescriptorByIndex(video_stream_index, &selected, &stream_descriptor));
	
	if (!selected) {
		print_line("WMFVideoDecoder: Video stream is not selected");
		SafeRelease(presentation_descriptor);
		SafeRelease(stream_descriptor);
		return E_FAIL;
	}
	
	// Create source node
	IMFTopologyNode *source_node = nullptr;
	CHECK_HR(MFCreateTopologyNode(MF_TOPOLOGY_SOURCESTREAM_NODE, &source_node));
	CHECK_HR(source_node->SetUnknown(MF_TOPONODE_SOURCE, media_source));
	CHECK_HR(source_node->SetUnknown(MF_TOPONODE_PRESENTATION_DESCRIPTOR, presentation_descriptor));
	CHECK_HR(source_node->SetUnknown(MF_TOPONODE_STREAM_DESCRIPTOR, stream_descriptor));
	CHECK_HR(topology->AddNode(source_node));
	
	// Create sample grabber sink
	IMFActivate *sink_activate = nullptr;
	CHECK_HR(create_sample_grabber_sink(&sink_activate));
	
	// Create output node
	IMFTopologyNode *output_node = nullptr;
	CHECK_HR(MFCreateTopologyNode(MF_TOPOLOGY_OUTPUT_NODE, &output_node));
	CHECK_HR(output_node->SetObject(sink_activate));
	CHECK_HR(output_node->SetUINT32(MF_TOPONODE_STREAMID, 0));
	CHECK_HR(output_node->SetUINT32(MF_TOPONODE_NOSHUTDOWN_ON_REMOVE, FALSE));
	CHECK_HR(topology->AddNode(output_node));
	
	// Connect source to output
	CHECK_HR(source_node->ConnectOutput(0, output_node, 0));
	
	// Cleanup
	SafeRelease(presentation_descriptor);
	SafeRelease(stream_descriptor);
	SafeRelease(source_node);
	SafeRelease(output_node);
	SafeRelease(sink_activate);
	
	return hr;
}

HRESULT WMFVideoDecoder::create_sample_grabber_sink(IMFActivate **sink_activate) {
	HRESULT hr = S_OK;
	
	// Create media type for sample grabber
	IMFMediaType *media_type = nullptr;
	CHECK_HR(MFCreateMediaType(&media_type));
	CHECK_HR(media_type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video));
	CHECK_HR(media_type->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_NV12));
	CHECK_HR(MFSetAttributeSize(media_type, MF_MT_FRAME_SIZE, video_width, video_height));
	CHECK_HR(media_type->SetUINT32(MF_MT_FIXED_SIZE_SAMPLES, TRUE));
	CHECK_HR(media_type->SetUINT32(MF_MT_ALL_SAMPLES_INDEPENDENT, TRUE));
	CHECK_HR(MFSetAttributeRatio(media_type, MF_MT_PIXEL_ASPECT_RATIO, 1, 1));
	
	// Create sample grabber sink activate
	CHECK_HR(MFCreateSampleGrabberSinkActivate(media_type, sample_grabber_callback, sink_activate));
	
	SafeRelease(media_type);
	return hr;
}
